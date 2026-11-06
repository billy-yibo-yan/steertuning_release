"""Steertuning: policies pretrained on human video, adapted by fine-tuning and action steering."""

import logging
import sys

__version__ = "1.0.0"


def setup_logging() -> None:
    """Print this package's log messages to stderr. Entry points call this first."""
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger(__name__).setLevel(logging.INFO)


def ask(prompt: str) -> str:
    """The operator's answer in lowercase. Keys pressed before the prompt appeared do not count."""
    if sys.stdin.isatty():
        import termios

        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    return input(prompt).strip().lower()
