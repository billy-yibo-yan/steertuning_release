"""Training windows for pretraining and fine-tuning.

A window is an anchor frame and the 15 target frames after it. Its text is the label of the segment
that holds the first target; later targets may run past the end of that segment. Observations come
from the anchor frame, and everything is expressed in the anchor frame's RGB camera frame.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataloader import default_collate

from steertuning.data.augmentation import (
    ANCHOR_ROTATION,
    ANCHOR_TRANSLATION,
    jitter_rgb,
    perturb_fingertips,
    sample_rigid,
)
from steertuning.data.normalizer import ActionNormalizer
from steertuning.data.representation import (
    CHUNK_STEPS,
    action_mask,
    chunk_frames,
    cloud_transform,
    crop_and_resample,
    fingertips,
    utonia_input,
)
from steertuning.data.session import (
    NO_WINDOWS_HINT,
    PARAPHRASE_FILE,
    Segment,
    num_frames,
    open_session,
    read_paraphrases,
    read_segments,
    session_directories,
    window_first_targets,
)


class TextTokenizer:
    """SigLIP token ids and attention mask of one instruction. The mask is ids != pad id."""

    def __init__(self, model_name: str, revision: str) -> None:
        from transformers import SiglipTokenizer

        self.tokenizer = SiglipTokenizer.from_pretrained(model_name, revision=revision)
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def __call__(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if text not in self._cache:
            ids = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=20)[
                "input_ids"
            ][0]
            self._cache[text] = ids, (ids != self.tokenizer.pad_token_id).long()
        return self._cache[text]


@dataclass(slots=True)
class _SegmentWindows:
    directory: Path
    segment: Segment
    anchors: np.ndarray


class WindowDataset(Dataset):
    """Windows over a dataset root, as pretraining reads them.

    Clouds and targets get no geometric or RGB augmentation. Instructions are drawn from paraphrases when
    a `text_prompt_augmentations.json` exists, and the proprioception noise is in the encoder.
    `dataset[i]` draws a window from segment `i`. `EndlessWindows` draws segments by `draw_segment`, and
    is what training iterates over, so every window counts the same.
    """

    def __init__(self, root: str | Path, normalizer: ActionNormalizer, *, tokenizer: TextTokenizer) -> None:
        self.root = Path(root)
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.cloud_transform = cloud_transform()
        self.paraphrases = read_paraphrases(self.root / PARAPHRASE_FILE)
        for path in sorted(self.root.glob(f"*/{PARAPHRASE_FILE}")):
            self.paraphrases.update(read_paraphrases(path))
        self.segments: list[_SegmentWindows] = []
        self._index_segments()
        self._cumulative_windows = np.cumsum([len(entry.anchors) for entry in self.segments])

    def _index_segments(self) -> None:
        for directory in session_directories(self.root):
            segments = read_segments(directory)
            if not segments:
                continue
            self.paraphrases.update(read_paraphrases(directory / PARAPHRASE_FILE))
            length = num_frames(directory)
            for segment in segments:
                anchors = window_first_targets(segment, length) - 1
                if anchors.size > 0:
                    self.segments.append(_SegmentWindows(directory, segment, anchors))
        if not self.segments:
            raise RuntimeError(f"No windows under {self.root}. {NO_WINDOWS_HINT}")

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> dict:
        return self.sample(index)

    def draw_segment(self) -> int:
        """Draw a segment index in proportion to its number of windows, so every window counts the same."""
        window = np.random.randint(self._cumulative_windows[-1])
        return int(np.searchsorted(self._cumulative_windows, window, side="right"))

    def sample(self, index: int, anchor: int | None = None, seed: int | None = None) -> dict:
        """Build a sample from segment `index`. A seed fixes every draw, including the anchor."""
        if seed is None:
            return self._build(index, anchor, np.random)
        state = np.random.get_state()
        np.random.seed(seed)  # the Utonia transform draws from the global stream
        try:
            return self._build(index, anchor, np.random.RandomState(seed))
        finally:
            np.random.set_state(state)

    def _build(self, index: int, anchor: int | None, rng) -> dict:
        entry = self.segments[index]
        session = open_session(entry.directory)
        if anchor is None:
            anchor = self._draw_anchor(entry, rng)
        tokens, text_mask = self.tokenizer(self._draw_text(entry.segment.instruction, rng))

        T_camera_world = session.camera_from_world(anchor, self._anchor_delta(rng))
        cloud = crop_and_resample(session.cloud(anchor, T_camera_world), rng)
        cloud = self._augment_cloud(cloud, rng)
        tips = fingertips(session.action_points(np.array([anchor]), T_camera_world)[0])
        tips = self._augment_fingertips(tips, session.wrist_poses(anchor, T_camera_world), rng)

        frames = chunk_frames(anchor)
        actions = self.normalizer.normalize(session.action_points(frames, T_camera_world))
        return {
            "point_cloud": self.cloud_transform(utonia_input(cloud)),
            "object_mask": torch.from_numpy(session.object_mask(anchor)).float()[None],
            "proprio": torch.from_numpy(self.normalizer.normalize(tips)).float(),
            "text_tokens": tokens,
            "text_mask": text_mask,
            "actions": torch.from_numpy(actions.reshape(CHUNK_STEPS, -1)).float(),
            "action_mask": torch.from_numpy(action_mask(session.hand_valid[frames])),
        }

    def _draw_anchor(self, entry: _SegmentWindows, rng) -> int:
        return int(rng.choice(entry.anchors))

    def _draw_text(self, instruction: str, rng) -> str:
        variants = self.paraphrases.get(instruction)
        if variants:
            return str(rng.choice(variants))
        return instruction

    def _anchor_delta(self, rng) -> np.ndarray | None:
        return None

    def _augment_cloud(self, cloud: np.ndarray, rng) -> np.ndarray:
        return cloud

    def _augment_fingertips(self, tips: np.ndarray, wrists: np.ndarray, rng) -> np.ndarray:
        return tips


class FinetuneDataset(WindowDataset):
    """The windows of a task directory, with the fine-tuning augmentations.

    The anchor frame moves rigidly, which moves the cloud and the targets with it. The fingertip
    proprioception moves on its own, per hand. Colors are jittered and instructions are drawn from
    `text_prompt_augmentations.json`.
    """

    def _anchor_delta(self, rng) -> np.ndarray:
        return sample_rigid(rng, ANCHOR_TRANSLATION, ANCHOR_ROTATION)

    def _augment_cloud(self, cloud: np.ndarray, rng) -> np.ndarray:
        return jitter_rgb(cloud, rng)

    def _augment_fingertips(self, tips: np.ndarray, wrists: np.ndarray, rng) -> np.ndarray:
        return perturb_fingertips(tips, wrists, rng)


class EndlessWindows(IterableDataset):
    """An endless training stream. Each DataLoader worker draws from its own seeded numpy stream."""

    def __init__(self, dataset: WindowDataset) -> None:
        self.dataset = dataset

    def __iter__(self) -> Iterator[dict]:
        while True:
            yield self.dataset[self.dataset.draw_segment()]


def collate(samples: list[dict]) -> dict:
    """Batch samples. `point_cloud` goes through Utonia's collate, which concatenates points."""
    from utonia.data import collate_fn

    batch = {key: default_collate([s[key] for s in samples]) for key in samples[0] if key != "point_cloud"}
    batch["point_cloud"] = collate_fn([s["point_cloud"] for s in samples])
    return batch


def to_batch(sample: dict, device: torch.device) -> dict:
    """A batch of one sample on `device`."""
    from accelerate.utils import send_to_device

    return send_to_device(collate([sample]), device)
