"""Fish Audio S2 TTS engine — connects to a locally-running Fish Audio HTTP server."""

from __future__ import annotations

import base64
import io
import logging
import struct
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests

from audiobook_forge.config import TTSFishAudioConfig
from audiobook_forge.tts.base import AnnotatedSentence, BaseTTSEngine, TTSResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Emotion → inline tag mapping
# ---------------------------------------------------------------------------

_EMOTION_TAG: dict[str, str] = {
    "neutral":   "",
    "calm":      "calm narrator voice",
    "curious":   "curious tone",
    "tense":     "tense, urgent",
    "excited":   "excited",
    "sad":       "sad, melancholic",
    "whispered": "whispering",
    "angry":     "angry",
}

_INTENSITY_PREFIX_THRESHOLD = 0.6  # intensity above this → prepend "very"

# Retry settings
_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 2.0

# Fish Audio native sample rate
_SAMPLE_RATE = 44_100


class FishAudioEngine(BaseTTSEngine):
    """TTS engine that forwards synthesis requests to a Fish Audio S2 HTTP API.

    The engine inserts emotion-specific inline tags into the text before sending
    it to the server, then assembles the returned WAV fragments into a single
    output file.

    Example usage::

        from audiobook_forge.config import TTSFishAudioConfig
        from audiobook_forge.tts.fish_audio_engine import FishAudioEngine

        config = TTSFishAudioConfig(
            api_url="http://localhost:8080",
            reference_audio="/path/to/reference.wav",
            reference_text="Reference transcript.",
        )
        engine = FishAudioEngine(config)
        engine.initialize()
        result = engine.synthesize(sentences, Path("chapter01.wav"))
        engine.shutdown()
    """

    def __init__(self, config: TTSFishAudioConfig) -> None:
        """
        Args:
            config: Fish Audio-specific configuration.
        """
        self.config = config
        self._session: requests.Session | None = None
        # Cache for base64-encoded reference audio so we only read the file once
        self._reference_audio_b64: str | None = None

    # ------------------------------------------------------------------
    # BaseTTSEngine interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "Fish Audio S2"

    @property
    def sample_rate(self) -> int:
        return _SAMPLE_RATE

    def initialize(self) -> None:
        """Verify connectivity to the Fish Audio API and load reference audio.

        Raises:
            RuntimeError: If the API is unreachable or returns an unexpected
                response.
        """
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        # Health check
        health_url = self.config.api_url.rstrip("/")
        logger.info("Connecting to Fish Audio API at %s …", health_url)
        try:
            resp = self._session.get(health_url + "/", timeout=10)
            resp.raise_for_status()
            logger.info("Fish Audio API reachable (status %d).", resp.status_code)
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Cannot reach Fish Audio API at {health_url}: {exc}"
            ) from exc

        # Pre-load reference audio
        if self.config.reference_audio:
            ref_path = Path(self.config.reference_audio)
            if ref_path.is_file():
                logger.info("Loading reference audio from %s …", ref_path)
                audio_bytes = ref_path.read_bytes()
                self._reference_audio_b64 = base64.b64encode(audio_bytes).decode()
                logger.debug(
                    "Reference audio loaded (%d bytes → %d b64 chars).",
                    len(audio_bytes),
                    len(self._reference_audio_b64),
                )
            else:
                logger.warning(
                    "reference_audio path %r does not exist; continuing without it.",
                    self.config.reference_audio,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_tag(self, emotion: str, intensity: float) -> str:
        """Return the Fish Audio inline tag string for *emotion* and *intensity*.

        When ``intensity`` exceeds :data:`_INTENSITY_PREFIX_THRESHOLD`, the
        adjective "very" is prepended to strengthen the tag (e.g.
        ``"very excited"``).

        Args:
            emotion:   Emotion label (e.g. ``"excited"``).
            intensity: Intensity in ``[0.0, 1.0]``.

        Returns:
            The raw tag text (without surrounding brackets), or an empty string
            for neutral/unrecognised emotions.
        """
        base = _EMOTION_TAG.get(emotion, "")
        if not base:
            return ""
        if intensity > _INTENSITY_PREFIX_THRESHOLD:
            return f"very {base}"
        return base

    def _tag_text(self, sentence: AnnotatedSentence) -> str:
        """Combine emotion tag and sentence text.

        Args:
            sentence: The annotated sentence.

        Returns:
            Tagged text ready to send to the API, e.g.
            ``"[very excited] The crowd went wild."``
        """
        tag = self._build_tag(sentence.emotion, sentence.intensity)
        if tag:
            return f"[{tag}] {sentence.text}"
        return sentence.text

    def _post_tts(self, text: str) -> bytes:
        """Send *text* to the Fish Audio TTS endpoint and return raw audio bytes.

        Retries up to :data:`_MAX_RETRIES` times with exponential back-off on
        connection/timeout errors.

        Args:
            text: Tagged text to synthesise.

        Returns:
            Raw WAV audio bytes from the server.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        assert self._session is not None, "Call initialize() first."

        endpoint = self.config.api_url.rstrip("/") + "/v1/tts"
        payload: dict[str, Any] = {
            "text": text,
            "format": "wav",
        }
        if self._reference_audio_b64:
            payload["reference_audio"] = self._reference_audio_b64
            payload["reference_text"] = self.config.reference_text

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = self._session.post(endpoint, json=payload, timeout=120)
                resp.raise_for_status()
                return resp.content
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "Fish Audio API attempt %d/%d failed: %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_S * attempt)

        raise RuntimeError(
            f"Fish Audio API unavailable after {_MAX_RETRIES} attempts: {last_exc}"
        )

    @staticmethod
    def _wav_bytes_to_array(wav_bytes: bytes) -> tuple[np.ndarray, int]:
        """Decode raw WAV bytes into a float32 numpy array.

        Args:
            wav_bytes: Raw WAV file content.

        Returns:
            Tuple of ``(audio_array, sample_rate)`` where *audio_array* is
            ``float32`` in the range ``[-1.0, 1.0]``.
        """
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(n_frames)

        # Convert raw bytes to int samples
        if sampwidth == 2:
            dtype = np.int16
            max_val = 32768.0
        elif sampwidth == 4:
            dtype = np.int32
            max_val = 2_147_483_648.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

        samples = np.frombuffer(raw, dtype=dtype).astype(np.float32) / max_val

        # Mix down to mono if stereo
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        return samples, sr

    @staticmethod
    def _make_silence(n_samples: int) -> np.ndarray:
        return np.zeros(n_samples, dtype=np.float32)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(
        self,
        sentences: list[AnnotatedSentence],
        output_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> TTSResult:
        """Synthesise *sentences* into a single WAV file via Fish Audio API.

        For each sentence the method:

        1. Builds an inline emotion tag and prepends it to the text.
        2. Posts the tagged text to the ``/v1/tts`` endpoint with retry logic.
        3. Decodes the returned WAV and appends the audio to a running buffer.
        4. Inserts inter-sentence and paragraph silence.
        5. Writes the final concatenated audio as a 16-bit PCM WAV.

        Args:
            sentences:         Annotated sentences to synthesise.
            output_path:       Destination path for the output WAV file.
            progress_callback: Optional ``callable(current_index, total)``
                              called after each sentence is synthesised.

        Returns:
            :class:`~audiobook_forge.tts.base.TTSResult` with path, duration,
            and sample rate.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called first.
        """
        if self._session is None:
            raise RuntimeError(
                "FishAudioEngine.initialize() must be called before synthesize()."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total = len(sentences)
        all_audio: list[np.ndarray] = []
        actual_sample_rate = _SAMPLE_RATE  # will be updated from first response

        for idx, sentence in enumerate(sentences):
            tagged_text = self._tag_text(sentence)
            logger.debug(
                "Sentence %d/%d | emotion=%r | text=%r",
                idx + 1,
                total,
                sentence.emotion,
                tagged_text[:80],
            )

            try:
                wav_bytes = self._post_tts(tagged_text)
                audio, sr = self._wav_bytes_to_array(wav_bytes)
                actual_sample_rate = sr
            except Exception:
                logger.exception(
                    "Failed to synthesise sentence %d; substituting silence.",
                    idx + 1,
                )
                fallback_secs = max(1.0, len(sentence.text) / 15.0)
                audio = self._make_silence(int(actual_sample_rate * fallback_secs))

            all_audio.append(audio)

            # Inter-sentence pause
            pause_ms = sentence.pause_after_ms
            if sentence.is_paragraph_end:
                pause_ms += 500  # paragraph-level extra silence

            if pause_ms > 0:
                silence_samples = int(actual_sample_rate * pause_ms / 1000.0)
                all_audio.append(self._make_silence(silence_samples))

            if progress_callback is not None:
                progress_callback(idx + 1, total)

        # --- Write final WAV ---
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

    @staticmethod
    def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
        """Write *audio* (float32) as a 16-bit PCM WAV file to *path*.

        Args:
            path:        Destination file path.
            audio:       Float32 audio array, values in ``[-1.0, 1.0]``.
            sample_rate: Sample rate in Hz.
        """
        pcm = np.clip(audio, -1.0, 1.0)
        pcm_int16 = (pcm * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_int16.tobytes())

    def shutdown(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            logger.info("Shutting down Fish Audio engine.")
            self._session.close()
            self._session = None
        self._reference_audio_b64 = None
