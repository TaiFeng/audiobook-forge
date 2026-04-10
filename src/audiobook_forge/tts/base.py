"""Abstract base class for TTS engines."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AnnotatedSentence:
    """A sentence with optional emotion/prosody metadata."""

    text: str
    speaker: str = "narrator"           # "narrator" or character name
    narration_mode: str = "prose"       # "prose", "dialogue", "thought"
    emotion: str = "neutral"            # calm, curious, tense, excited, sad, whispered, etc.
    intensity: float = 0.3              # 0.0 = flat, 1.0 = maximum
    pause_after_ms: int = 250           # Silence to insert after this sentence
    is_paragraph_end: bool = False      # Triggers paragraph-level pause


@dataclass
class TTSResult:
    """Result from a TTS synthesis call."""

    audio_path: Path
    duration_seconds: float = 0.0
    sample_rate: int = 24000


class BaseTTSEngine(abc.ABC):
    """Interface that all TTS backends must implement."""

    @abc.abstractmethod
    def initialize(self) -> None:
        """Load model weights, connect to API, etc."""

    @abc.abstractmethod
    def synthesize(
        self,
        sentences: list[AnnotatedSentence],
        output_path: Path,
        progress_callback: Any | None = None,
    ) -> TTSResult:
        """
        Synthesize a list of annotated sentences into a single audio file.

        The engine should:
        1. Translate emotion annotations into engine-specific controls
        2. Generate audio for each sentence
        3. Insert pauses per sentence metadata
        4. Concatenate into a single WAV at output_path

        Args:
            sentences: Annotated sentences to synthesize.
            output_path: Where to write the output WAV.
            progress_callback: Optional callable(current, total) for progress.

        Returns:
            TTSResult with the path and duration.
        """

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release resources (GPU memory, connections, etc.)."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""

    @property
    @abc.abstractmethod
    def sample_rate(self) -> int:
        """Native sample rate of this engine's output."""
