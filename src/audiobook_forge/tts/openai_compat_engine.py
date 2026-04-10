"""OpenAI-compatible TTS engine — works with OpenAI, LM Studio, vLLM, etc."""

from __future__ import annotations

import io
import logging
import time
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from audiobook_forge.config import TTSOpenAICompatConfig
from audiobook_forge.tts.base import AnnotatedSentence, BaseTTSEngine, TTSResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate-limit / retry constants
# ---------------------------------------------------------------------------

_MAX_RETRIES = 5
_BASE_BACKOFF_S = 1.0   # first retry waits 1 s, then 2 s, 4 s, 8 s, …
_MAX_BACKOFF_S = 60.0


class OpenAICompatEngine(BaseTTSEngine):
    """TTS engine for any OpenAI-compatible ``/v1/audio/speech`` endpoint.

    Tested against:
    * OpenAI API (``https://api.openai.com/v1``)
    * LM Studio local server (``http://localhost:1234/v1``)
    * vLLM OpenAI-compatible mode

    The engine does **not** translate emotion annotations into the request
    because the OpenAI TTS API does not yet expose per-sentence prosody
    controls.  Emotion information is therefore silently ignored.

    Example usage::

        from audiobook_forge.config import TTSOpenAICompatConfig
        from audiobook_forge.tts.openai_compat_engine import OpenAICompatEngine

        config = TTSOpenAICompatConfig(
            api_url="https://api.openai.com/v1",
            api_key="sk-...",
            model="tts-1-hd",
            voice="nova",
        )
        engine = OpenAICompatEngine(config)
        engine.initialize()
        result = engine.synthesize(sentences, Path("chapter01.wav"))
        engine.shutdown()
    """

    def __init__(self, config: TTSOpenAICompatConfig) -> None:
        """
        Args:
            config: OpenAI-compatible TTS configuration.
        """
        self.config = config
        self._client = None  # openai.OpenAI instance

    # ------------------------------------------------------------------
    # BaseTTSEngine interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "OpenAI Compatible"

    @property
    def sample_rate(self) -> int:
        return 24_000

    def initialize(self) -> None:
        """Create the OpenAI client pointed at the configured base URL.

        Raises:
            RuntimeError: If the ``openai`` package is not installed.
        """
        try:
            import openai  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required but not installed. "
                "Install it with: pip install openai"
            ) from exc

        logger.info(
            "Initialising OpenAI-compatible TTS client (base_url=%r, model=%r, voice=%r)",
            self.config.api_url,
            self.config.model,
            self.config.voice,
        )

        self._client = openai.OpenAI(
            base_url=self.config.api_url,
            api_key=self.config.api_key or "not-required",
        )
        logger.info("OpenAI-compatible TTS client ready.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _speech_with_backoff(self, text: str) -> bytes:
        """Call ``audio.speech.create`` with exponential back-off on rate limits.

        Args:
            text: The input text to synthesise.

        Returns:
            Raw WAV audio bytes from the API response.

        Raises:
            RuntimeError: If all retries are exhausted.
            openai.APIError: On non-rate-limit API errors.
        """
        import openai  # type: ignore[import]

        assert self._client is not None, "Call initialize() first."

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.audio.speech.create(
                    model=self.config.model,
                    voice=self.config.voice,  # type: ignore[arg-type]
                    input=text,
                    response_format="wav",
                )
                # The response content can be iterated or accessed as bytes
                return b"".join(response.iter_bytes())
            except openai.RateLimitError as exc:
                last_exc = exc
                backoff = min(_BASE_BACKOFF_S * (2 ** attempt), _MAX_BACKOFF_S)
                logger.warning(
                    "Rate limit hit (attempt %d/%d); retrying in %.1f s …",
                    attempt + 1,
                    _MAX_RETRIES,
                    backoff,
                )
                time.sleep(backoff)
            except openai.APIStatusError as exc:
                # Retry on 5xx server errors; re-raise client errors immediately
                if exc.status_code is not None and exc.status_code >= 500:
                    last_exc = exc
                    backoff = min(_BASE_BACKOFF_S * (2 ** attempt), _MAX_BACKOFF_S)
                    logger.warning(
                        "Server error %d (attempt %d/%d); retrying in %.1f s …",
                        exc.status_code,
                        attempt + 1,
                        _MAX_RETRIES,
                        backoff,
                    )
                    time.sleep(backoff)
                else:
                    raise
            except openai.APIConnectionError as exc:
                last_exc = exc
                backoff = min(_BASE_BACKOFF_S * (2 ** attempt), _MAX_BACKOFF_S)
                logger.warning(
                    "Connection error (attempt %d/%d); retrying in %.1f s …",
                    attempt + 1,
                    _MAX_RETRIES,
                    backoff,
                )
                time.sleep(backoff)

        raise RuntimeError(
            f"OpenAI TTS endpoint unavailable after {_MAX_RETRIES} attempts: {last_exc}"
        )

    @staticmethod
    def _wav_bytes_to_array(wav_bytes: bytes) -> tuple[np.ndarray, int]:
        """Decode WAV bytes into a float32 numpy array.

        Args:
            wav_bytes: Raw WAV file content.

        Returns:
            Tuple of ``(audio_array, sample_rate)``.
        """
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(n_frames)

        if sampwidth == 2:
            dtype = np.int16
            max_val = 32768.0
        elif sampwidth == 4:
            dtype = np.int32
            max_val = 2_147_483_648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

        samples = np.frombuffer(raw, dtype=dtype).astype(np.float32) / max_val
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)
        return samples, sr

    @staticmethod
    def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
        """Write float32 *audio* as a 16-bit PCM WAV to *path*.

        Args:
            path:        Destination file path.
            audio:       Float32 audio samples, values in ``[-1.0, 1.0]``.
            sample_rate: Sample rate in Hz.
        """
        pcm = np.clip(audio, -1.0, 1.0)
        pcm_int16 = (pcm * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_int16.tobytes())

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(
        self,
        sentences: list[AnnotatedSentence],
        output_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> TTSResult:
        """Synthesise *sentences* into a single WAV via the OpenAI speech API.

        Each sentence is sent as a separate API call (the API does not accept
        batch requests).  Rate-limit errors are handled with exponential
        back-off.  Silence is inserted after each sentence according to
        ``pause_after_ms``; paragraph ends receive an additional 500 ms.

        Args:
            sentences:         Annotated sentences to synthesise.
            output_path:       Destination path for the output WAV file.
            progress_callback: Optional ``callable(current_index, total)``
                              called after each sentence finishes.

        Returns:
            :class:`~audiobook_forge.tts.base.TTSResult` with path, duration,
            and sample rate.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called first.
        """
        if self._client is None:
            raise RuntimeError(
                "OpenAICompatEngine.initialize() must be called before synthesize()."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total = len(sentences)
        all_audio: list[np.ndarray] = []
        actual_sample_rate = self.sample_rate

        for idx, sentence in enumerate(sentences):
            logger.debug(
                "Sentence %d/%d | text=%r",
                idx + 1,
                total,
                sentence.text[:80],
            )

            try:
                wav_bytes = self._speech_with_backoff(sentence.text)
                audio, sr = self._wav_bytes_to_array(wav_bytes)
                actual_sample_rate = sr
            except Exception:
                logger.exception(
                    "Failed to synthesise sentence %d; substituting silence.",
                    idx + 1,
                )
                fallback_secs = max(1.0, len(sentence.text) / 15.0)
                audio = np.zeros(
                    int(actual_sample_rate * fallback_secs), dtype=np.float32
                )

            all_audio.append(audio)

            # Inter-sentence / paragraph pause
            pause_ms = sentence.pause_after_ms
            if sentence.is_paragraph_end:
                pause_ms += 500

            if pause_ms > 0:
                silence_samples = int(actual_sample_rate * pause_ms / 1000.0)
                all_audio.append(np.zeros(silence_samples, dtype=np.float32))

            if progress_callback is not None:
                progress_callback(idx + 1, total)

        # Concatenate & write
        if all_audio:
            final_audio = np.concatenate(all_audio)
        else:
            logger.warning("No audio generated; writing empty WAV.")
            final_audio = np.zeros(0, dtype=np.float32)

        self._write_wav(output_path, final_audio, actual_sample_rate)

        duration_seconds = (
            len(final_audio) / actual_sample_rate if actual_sample_rate else 0.0
        )
        logger.info(
            "Wrote %s (%.1f s, %d samples @ %d Hz)",
            output_path,
            duration_seconds,
            len(final_audio),
            actual_sample_rate,
        )
        return TTSResult(
            audio_path=output_path,
            duration_seconds=duration_seconds,
            sample_rate=actual_sample_rate,
        )

    def shutdown(self) -> None:
        """Release the OpenAI client."""
        if self._client is not None:
            logger.info("Shutting down OpenAI-compatible TTS engine.")
            self._client = None
