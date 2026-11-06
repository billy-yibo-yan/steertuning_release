"""Recording task demonstrations from the live Aria stream.

A session on disk:

    processed/rgb/000000.jpg           stream JPEG bytes as received
    processed/stereo_left/000000.png   rectified left image
    processed/stereo_right/000000.jpg  rectified right image, stream JPEG bytes
    raw/frames.npz                     per-frame timestamps and head poses at each camera's capture
    raw/hands.npz                      every hand tracking sample, in the device frame
    raw/calib.json                     rectified intrinsics and extrinsics
    raw/meta.json                      task, typed description, counts

The publisher is `aria2mano.stream_pub`, on the machine connected to the glasses.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from steertuning import ask
from steertuning.processing.recording import RAW_DIR

VIO_FROZEN_AFTER_S = 1.5
VIO_STALE_AFTER_S = 3.0

_FRAME_KEYS = ("t_pub", "t_wall", "ts_ns_rgb", "ts_ns_stereo_left", "ts_ns_stereo_right")
_POSE_KEYS = ("T_odom_device_stereo", "T_odom_device_rgb")
_STREAM_META_KEYS = ("profile_name", "interface", "has_rgb", "has_stereo", "has_hands", "has_vio")


def _pose_or_nan(value) -> np.ndarray:
    pose = np.full((4, 4), np.nan) if value is None else np.asarray(value, dtype=np.float64)
    return pose if pose.shape == (4, 4) and np.isfinite(pose).all() else np.full((4, 4), np.nan)


def _write_json(path: Path, value: dict) -> None:
    def plain(item):
        return item.tolist() if isinstance(item, np.ndarray) else item.item() if isinstance(item, np.generic) else item

    path.write_text(json.dumps(value, indent=2, default=plain))


class _SessionWriter:
    """One demonstration on disk. Images go straight to files, poses and hands stay in memory until the end."""

    def __init__(self, session_dir: Path, task: str, description: str, calib: dict, stream_meta: dict) -> None:
        self.session_dir = session_dir
        self.task = task
        self.description = description
        self.calib = calib
        self.stream_meta = {key: stream_meta.get(key) for key in _STREAM_META_KEYS}
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.frames: dict[str, list] = {key: [] for key in (*_FRAME_KEYS, *_POSE_KEYS)}
        self.hands: list[dict] = []
        self.dropped_frames = 0
        for name in ("rgb", "stereo_left", "stereo_right"):
            (session_dir / "processed" / name).mkdir(parents=True)
        (session_dir / RAW_DIR).mkdir()

    def add_frame(self, message: dict) -> None:
        rgb, left, right = (message.get(key) for key in ("rgb", "stereo_left", "stereo_right"))
        left_image = None
        if all(isinstance(image, (bytes, bytearray)) for image in (rgb, left, right)):
            left_image = cv2.imdecode(np.frombuffer(left, np.uint8), cv2.IMREAD_GRAYSCALE)
        with self.lock:
            if left_image is None:
                self.dropped_frames += 1
                return
            frame = len(self.frames["t_pub"])
            for key in _FRAME_KEYS:
                self.frames[key].append(message.get(key, -1))
            for key in _POSE_KEYS:
                self.frames[key].append(_pose_or_nan(message.get(key)))
        processed = self.session_dir / "processed"
        (processed / "rgb" / f"{frame:06d}.jpg").write_bytes(bytes(rgb))
        cv2.imwrite(str(processed / "stereo_left" / f"{frame:06d}.png"), left_image)
        (processed / "stereo_right" / f"{frame:06d}.jpg").write_bytes(bytes(right))

    def add_hands(self, message: dict) -> None:
        row = {
            "t_pub": message.get("t_pub", np.nan),
            "t_wall": message.get("t_wall", np.nan),
            "T_odom_device": _pose_or_nan(message.get("T_odom_device")),
        }
        tracked = message.get("htd") or {}
        for side in ("left", "right"):
            hand = tracked.get(side) if isinstance(tracked.get(side), dict) else {}
            wrist = _pose_or_nan(hand.get("T_device_wrist"))
            valid = hand.get("landmarks") is not None and np.isfinite(wrist).all()
            row[f"{side}_valid"] = float(valid)
            row[f"{side}_confidence"] = hand.get("confidence", np.nan) if valid else np.nan
            landmarks = np.asarray(hand["landmarks"]).reshape(21, 3) if valid else np.full((21, 3), np.nan)
            row[f"{side}_landmarks"] = landmarks
            row[f"{side}_wrist"] = wrist
        with self.lock:
            self.hands.append(row)

    def finish(self) -> dict:
        with self.lock:
            frames = {key: np.asarray(values) for key, values in self.frames.items()}
            hands = {key: np.asarray([row[key] for row in self.hands]) for key in self.hands[0]} if self.hands else {}
        raw = self.session_dir / RAW_DIR
        np.savez_compressed(raw / "frames.npz", **frames)
        np.savez_compressed(raw / "hands.npz", **hands)
        _write_json(raw / "calib.json", self.calib)
        duration = time.monotonic() - self.started
        count = len(frames["t_pub"])
        meta = {
            "task": self.task,
            "prompt": self.description,
            "session": self.session_dir.name,
            "duration_s": round(duration, 2),
            "n_frames": count,
            "n_hand_samples": len(self.hands),
            "n_dropped_frames": self.dropped_frames,
            "images_hz": round(count / duration, 2) if duration > 0 else 0.0,
            "stream_meta": self.stream_meta,
        }
        _write_json(raw / "meta.json", meta)
        return meta


class AriaStreamRecorder:
    """Subscribes to the stream and writes a session between `start` and `stop`."""

    def __init__(self, host: str, port: int) -> None:
        from commlink import Subscriber

        self.subscribers = {
            "meta": Subscriber(host, port=port, topics=["meta"]),
            "online_calib": Subscriber(host, port=port, topics=["online_calib"]),
            "images": Subscriber(host, port=port, topics=["images"], buffer=True, queue_size=120),
            "hands": Subscriber(host, port=port, topics=["hands"], buffer=True, queue_size=512),
        }
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.writer: _SessionWriter | None = None
        self.latest: dict[str, dict] = {}
        self.pose_last: bytes | None = None
        self.pose_seen_at: float | None = None
        self.pose_changed_at: float | None = None
        for target in (self._images_loop, self._hands_loop, self._state_loop):
            threading.Thread(target=target, daemon=True).start()

    def _next_message(self, topic: str) -> dict | None:
        try:
            message = self.subscribers[topic].get(topic)
        except Exception:
            message = None
        if not isinstance(message, dict):
            time.sleep(0.01)
            return None
        return message

    def _images_loop(self) -> None:
        while not self.stopping.is_set():
            if (message := self._next_message("images")) is None:
                continue
            with self.lock:
                self._note_pose(message.get("T_odom_device_rgb"))
                writer = self.writer
            if writer is not None:
                writer.add_frame(message)

    def _hands_loop(self) -> None:
        while not self.stopping.is_set():
            if (message := self._next_message("hands")) is None:
                continue
            with self.lock:
                writer = self.writer
            if writer is not None:
                writer.add_hands(message)

    def _state_loop(self) -> None:
        while not self.stopping.is_set():
            for topic in ("meta", "online_calib"):
                if (message := self._next_message(topic)) is not None:
                    with self.lock:
                        self.latest[topic] = message
            self.stopping.wait(1.0)

    def _note_pose(self, value) -> None:
        pose = _pose_or_nan(value)
        if not np.isfinite(pose).all():
            return
        now = time.monotonic()
        self.pose_seen_at = now
        if (key := pose.tobytes()) != self.pose_last:
            self.pose_last = key
            self.pose_changed_at = now

    def not_ready_reason(self) -> str | None:
        """Why recording should not start yet, or None. A dead head pose stream repeats one pose exactly."""
        now = time.monotonic()
        with self.lock:
            seen, changed, latest = self.pose_seen_at, self.pose_changed_at, dict(self.latest)
        if "meta" not in latest or "online_calib" not in latest:
            return "waiting for the stream publisher"
        if seen is None or now - seen > VIO_STALE_AFTER_S:
            return "no head pose in the image stream"
        if now - changed > VIO_FROZEN_AFTER_S:
            return "head pose is frozen: restart the stream publisher, then move your head"
        return None

    def wait_until_ready(self, timeout_s: float = 60.0) -> None:
        deadline = time.monotonic() + timeout_s
        while (reason := self.not_ready_reason()) is not None:
            if time.monotonic() > deadline:
                raise RuntimeError(f"Stream not ready: {reason}")
            time.sleep(0.2)

    def start(self, session_dir: Path, task: str, description: str) -> None:
        with self.lock:
            calib, meta = self.latest["online_calib"], self.latest["meta"]
        writer = _SessionWriter(session_dir, task, description, calib, meta)
        with self.lock:
            self.writer = writer

    def stop(self) -> dict:
        with self.lock:
            writer, self.writer = self.writer, None
        return writer.finish()

    def close(self) -> None:
        self.stopping.set()
        # Not the other two, which `_state_loop` reads: closing a socket during a read aborts the process.
        for topic in ("images", "hands"):
            self.subscribers[topic].stop()


def next_session_dir(task_dir: Path) -> Path:
    """The directory for the next demonstration, `<task>/<task>_N`."""
    pattern = re.compile(rf"^{re.escape(task_dir.name)}_(\d+)$")
    numbers = [int(match[1]) for path in task_dir.glob("*") if (match := pattern.match(path.name))]
    return task_dir / f"{task_dir.name}_{max(numbers, default=0) + 1}"


def recorded_description(task_dir: Path) -> str | None:
    """The task description typed when the demonstrations of a task were recorded."""
    for meta in sorted(task_dir.glob(f"*/{RAW_DIR}/meta.json")):
        if prompt := json.loads(meta.read_text()).get("prompt"):
            return prompt
    return None


def _ask_description() -> str:
    while not (description := input("Task description: ").strip()):
        print("A task needs a description: it is the prompt at deployment.")
    return description


def record_demo(task_dir: Path, description: str | None, host: str, port: int) -> dict:
    """Record one demonstration from the stream. Enter starts it and Enter stops it.

    The task description is the deployment prompt. It is asked for once per task.
    """
    description = description or recorded_description(task_dir) or _ask_description()
    recorder = AriaStreamRecorder(host, port)
    try:
        print(f"Connecting to the stream at {host}:{port}")
        recorder.wait_until_ready()
        session_dir = next_session_dir(task_dir)
        ask(f"Press Enter to start recording {session_dir.name}: ")
        recorder.start(session_dir, task_dir.name, description)
        ask("Recording. Press Enter to stop: ")
        meta = recorder.stop()
    finally:
        recorder.close()
    if meta["n_frames"] == 0:
        raise RuntimeError(f"No frames arrived during {session_dir.name}. Check the stream and record it again.")
    print(f"Saved {session_dir}: {meta['n_frames']} frames in {meta['duration_s']} s ({meta['images_hz']} Hz)")
    return meta
