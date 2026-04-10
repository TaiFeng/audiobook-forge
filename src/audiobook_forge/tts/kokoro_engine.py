"""Kokoro TTS engine — wraps the `kokoro` Python package (v0.9+)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from audiobook_forge.config import TTSKokoroConfig
from audiobook_forge.tts.base import AnnotatedSentence, BaseTTSEngine, TTSResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Emotion → speed/voice modifier mappings
# ---------------------------------------------------------------------------

# Multiplicative speed adjustments on top of the configured base speed
_EMOTION_SPEED_FACTOR: dict[str, float] = {
    "neutral":   1.00,
    "calm":      0.95,
    "curious":   1.00,
    "tense":     1.05,
    "excited":   1.05,
    "sad":       0.95,
    "whispered": 0.85,
    "angry":     1.10,
}

# Kokoro voice suffix variants (appended when a variant exists).
# Kokoro voice names typically look like "af_heart", "af_sky", etc.
# We leave voice selection to the user config; we only adjust speed here.
_DEFAULT_SPEED_FACTOR = 1.00


class KokoroEngine(BaseTTSEngine):
    """TTS engine that uses the Kokoro neural TTS pipeline.

    The engine wraps ``KPipeline`` from the ``kokoro`` package and translates
    :class:`~audiobook_forge.tts.base.AnnotatedSentence` emotion/prosody
    annotations into Kokoro speed controls.

    Example usage::

        from audiobook_forge.config import TTSKokoroConfig
        from audiobook_forge.tts.kokoro_engine import KokoroEngine

        engine = KokoroEngine(TTSKokoroConfig(voice="af_heart", speed=1.0))
        engine.initialize()
        result = engine.synthesize(sentences, Path("chapter01.wav"))
        engine.shutdown()
    """

    def __init__(self, config: TTSKokoroConfig) -> None:
        """
        Args:
            config: Kokoro-specific configuration (voice, speed, lang_code).
        """
        self.config = config
        self.pipeline: Any = None  # KPipeline instance, set in initialize()

    # ------------------------------------------------------------------
    # BaseTTSEngine interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "Kokoro TTS"

    @property
    def sample_rate(self) -> int:
        return 24_000

    def initialize(self) -> None:
        """Load the Kokoro pipeline.

        Raises:
            RuntimeError: If the ``kokoro`` package is not installed.
        """
        try:
            from kokoro import KPipeline  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "The 'kokoro' package is required but not installed. "
                "Install it with: pip install kokoro soundfile"
            ) from exc

        logger.info(
            "Initialising Kokoro TTS pipeline (lang_code=%r, voice=%r)",
            self.config.lang_code,
            self.config.voice,
        )
        self.pipeline = KPipeline(lang_code=self.config.lang_code)
        logger.info("Kokoro pipeline ready.")

    def synthesize(
        self,
        sentences: list[AnnotatedSentence],
        output_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> TTSResult:
        """Synthesize *sentences* into a single WAV file at *output_path*.

        Each sentence is rendered using the configured voice and a per-emotion
        speed multiplier.  Silence is inserted after each sentence according to
        ``pause_after_ms``; paragraph endings receive an extra pause controlled
        by :attr:`~audiobook_forge.config.AudioConfig.paragraph_pause_ms`
        (hard-coded to 500 ms here because ``AudioConfig`` is not available at
        this layer — callers may provide it through the config chain).

        Args:
            sentences:        Annotated sentences to synthesise.
            output_path:      Destination path for the output WAV file.
            progress_callback: Optional ``callable(current_index, total)``
                              called after each sentence is synthesised.

        Returns:
            :class:`~audiobook_forge.tts.base.TTSResult` containing the output
            path, duration, and sample rate.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called first.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "KokoroEngine.initialize() must be called before synthesize()."
            )

        try:
            import soundfile as sf  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "The 'soundfile' package is required but not installed. "
                "Install it with: pip install soundfile"
            ) from exc

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total = len(sentences)
        all_audio: list[np.ndarray] = []

        for idx, sentence in enumerate(sentences):
            # --- Compute per-sentence speed ---
            emotion_factor = _EMOTION_SPEED_FACTOR.get(
                sentence.emotion, _DEFAULT_SPEED_FACTOR
            )
            # Blend the factor with intensity: at intensity=0 we keep base
            # speed; at intensity=1 we apply the full emotion factor.
            blended_factor = 1.0 + (emotion_factor - 1.0) * sentence.intensity
            speed = self.config.speed * blended_factor
            # Clamp to a sane range
            speed = max(0.5, min(speed, 2.0))

            voice = self.config.voice

            logger.debug(
                "Sentence %d/%d | emotion=%r intensity=%.2f speed=%.3f",
                idx + 1,
                total,
                sentence.emotion,
                sentence.intensity,
                speed,
            )

            # --- Generate audio chunks ---
            sentence_chunks: list[np.ndarray] = []
            try:
                for _gs, _ps, audio_chunk in self.pipeline(
                    sentence.text, voice=voice, speed=speed
                ):
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        sentence_chunks.append(
                            audio_chunk.astype(np.float32)
                        )
            except Exception:
                logger.exception(
                    "Kokoro pipeline error on sentence %d: %r",
                    idx + 1,
                    sentence.text[:80],
                )
                # Insert silence of approximately the expected sentence length
                # so that timing is not completely broken.
                fallback_secs = max(1.0, len(sentence.text) / 15.0)
                sentence_chunks = [
                    np.zeros(
                        int(fallback_secs * self.sample_rate), dtype=np.float32
                    )
                ]

            if sentence_chunks:
                sentence_audio = np.concatenate(sentence_chunks)
            else:
                sentence_audio = np.zeros(0, dtype=np.float32)

            all_audio.append(sentence_audio)

            # --- Insert pause after sentence ---
            pause_ms = sentence.pause_after_ms
            if sentence.is_paragraph_end:
                # Add extra paragraph-level silence (500 ms default)
                pause_ms += 500

            if pause_ms > 0:
                silence_samples = int(self.sample_rate * pause_ms / 1000.0)
                all_audio.append(
                    np.zeros(silence_samples, dtype=np.float32)
                )

            # --- Report progress ---
            if progress_callback is not None:
                progress_callback(idx + 1, total)

        # --- Concatenate everything ---
        if all_audio:
            final_audio = np.concatenate(all_audio)
        else:
            logger.warning("No audio was generated; writing empty WAV.")
            final_audio = np.zeros(0, dtype=np.float32)

        sf.write(str(output_path), final_audio, self.sample_rate, subtype="PCM_16")

        duration_seconds = len(final_audio) / self.sample_rate
        logger.info(
            "Wrote %s (%.1f s, %d samples)",
            output_path,
            duration_seconds,
            len(final_audio),
        )

        return TTSResult(
            audio_path=output_path,
            duration_seconds=duration_seconds,
            sample_rate=self.sample_rate,
        )

    def shutdown(self) -> None:
        """Release the Kokoro pipeline and free GPU/CPU resources."""
        if self.pipeline is not None:
            logger.info("Shutting down Kokoro TTS pipeline.")
            self.pipeline = None
