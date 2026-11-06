"""Processing a directory of recordings: its sessions, split file and a task's paraphrases."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from steertuning.data.session import (
    PARAPHRASE_FILE,
    SPLIT_FILE,
    read_paraphrases,
    read_segments,
    read_split,
)
from steertuning.processing.language import paraphrase_table
from steertuning.processing.pipeline import OUTPUT_DIR, Models, ProcessConfig, process_session
from steertuning.processing.recording import RAW_DIR, UnusableSession, is_recording
from steertuning.processing.stream_recorder import recorded_description

log = logging.getLogger(__name__)


def find_sessions(path: Path) -> list[Path]:
    """The recordings at or under a directory."""
    if is_recording(path):
        return [path]
    candidates = {p.parents[1] for p in path.rglob(f"{RAW_DIR}/frames.npz")}
    candidates |= {p.parents[1] for p in path.rglob("processed/aria_data.pkl")}
    return sorted(candidates)


def write_split(root: Path, sessions: list[Path], excluded: dict[str, str]) -> None:
    """`train_test_split.json` for the sessions that were processed.

    Every session trains, except those an existing split file lists under `test`.
    """
    ids = [str(session.relative_to(root)) for session in sessions]
    path = root / SPLIT_FILE
    _, test = read_split(root)
    split = {"train": sorted(set(ids) - set(test)), "test": sorted(test), "excluded": excluded}
    path.write_text(json.dumps(split, indent=2))


def write_paraphrases(
    root: Path, sessions: list[Path], description: str, config: ProcessConfig, models: Models
) -> dict:
    """Write `text_prompt_augmentations.json` for a task and return its contents."""
    path = root / PARAPHRASE_FILE
    instructions = [segment.instruction for session in sessions for segment in read_segments(session / OUTPUT_DIR)]
    existing = read_paraphrases(path)
    if existing and {description, *instructions} <= existing.keys():
        return json.loads(path.read_text())
    table = paraphrase_table(models.language, instructions, description, config.num_paraphrases)
    path.write_text(json.dumps(table, indent=2))
    return table


def process_directory(
    path: str | Path, config: ProcessConfig, description: str | None = None, models: Models | None = None
) -> dict:
    """Process every recording at or under `path`.

    A directory of recordings also gets `train_test_split.json`. A task gets the paraphrase table, and
    its demonstrations train unless the split lists them under `test`. A task is a directory with a
    description, typed here or when the demonstrations were recorded.
    """
    path = Path(path)
    sessions = find_sessions(path)
    if not sessions:
        raise FileNotFoundError(f"No recordings under {path}")
    models = models or Models(config)
    notes, usable, excluded = [], [], {}
    for session in sessions:
        try:
            notes.append(process_session(session, config, models))
            usable.append(session)
        except UnusableSession as error:
            excluded[session.name if session == path else str(session.relative_to(path))] = str(error)
            log.warning("Excluded %s", error)
    summary = {"sessions": notes, "excluded": excluded}
    if is_recording(path):
        return summary
    description = description or recorded_description(path)
    write_split(path, usable, excluded)
    if description and usable:
        summary["paraphrases"] = write_paraphrases(path, usable, description, config, models)
    return summary
