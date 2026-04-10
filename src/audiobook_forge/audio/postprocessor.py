"""Audio post-processing utilities using ffmpeg and ffprobe subprocesses."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

from audiobook_forge.config import AudioConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_ffmpeg() -> str:
    """Return the absolute path to the ``ffmpeg`` binary.

    Raises:
        RuntimeError: If ``ffmpeg`` is not found on ``PATH``.
    """
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. "
            "Install it via your system package manager "
            "(e.g. 'apt install ffmpeg' or 'brew install ffmpeg')."
        )
    return path


def _require_ffprobe() -> str:
    """Return the absolute path to the ``ffprobe`` binary.

    Raises:
        RuntimeError: If ``ffprobe`` is not found on ``PATH``.
    """
    path = shutil.which("ffprobe")
    if path is None:
        raise RuntimeError(
            "ffprobe is not installed or not on PATH. "
            "Install ffmpeg (which includes ffprobe) via your system package manager."
        )
    return path


def _run(
    cmd: list[str],
    *,
    capture_output: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command with consistent error handling.

    Args:
        cmd:            Command list passed to :func:`subprocess.run`.
        capture_output: Whether to capture stdout/stderr.
        check:          Whether to raise on non-zero exit codes.

    Returns:
        Completed process result.

    Raises:
        RuntimeError: Wraps :exc:`subprocess.CalledProcessError` for clarity.
    """
    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=check,
        )
        return result
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        raise RuntimeError(
            f"Command failed (exit {exc.returncode}): {' '.join(cmd)}\n{stderr}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_loudness(
    input_path: Path | str,
    output_path: Path | str,
    target_lufs: float = -19.0,
) -> None:
    """Normalise audio loudness using the ffmpeg ``loudnorm`` filter (two-pass).

    The two-pass approach measures the actual integrated loudness in the first
    pass and applies a precise correction in the second pass — which is more
    accurate than single-pass normalisation.

    Args:
        input_path:   Path to the input audio file.
        output_path:  Path where the normalised output will be written.
        target_lufs:  Target integrated loudness in LUFS (default ``-19.0``).
                      Typical audiobook targets: ``-18`` to ``-20`` LUFS.

    Raises:
        RuntimeError: If ``ffmpeg`` is not installed or the command fails.
    """
    ffmpeg = _require_ffmpeg()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: measure loudness statistics ---
    logger.info("loudnorm pass 1 (measuring) on %s …", input_path)
    pass1_cmd = [
        ffmpeg, "-nostdin", "-i", str(input_path),
        "-af", (
            f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json"
        ),
        "-f", "null", "-",
    ]
    pass1_result = _run(pass1_cmd)

    # loudnorm JSON is written to stderr
    stderr = pass1_result.stderr or ""
    json_match = re.search(r"\{[^}]*\}", stderr, re.DOTALL)
    if not json_match:
        raise RuntimeError(
            f"loudnorm pass 1 did not return JSON stats.\n"
            f"ffmpeg stderr:\n{stderr}"
        )
    stats = json.loads(json_match.group())

    measured_I    = stats.get("input_i",    "-70")
    measured_LRA  = stats.get("input_lra",  "0")
    measured_TP   = stats.get("input_tp",   "-70")
    measured_thresh = stats.get("input_thresh", "-70")
    offset        = stats.get("target_offset", "0")

    # --- Pass 2: apply normalisation ---
    logger.info(
        "loudnorm pass 2 (applying) on %s → %s …", input_path, output_path
    )
    af = (
        f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
        f":measured_I={measured_I}"
        f":measured_LRA={measured_LRA}"
        f":measured_TP={measured_TP}"
        f":measured_thresh={measured_thresh}"
        f":offset={offset}"
        f":linear=true:print_format=summary"
    )
    pass2_cmd = [
        ffmpeg, "-nostdin", "-i", str(input_path),
        "-af", af,
        "-y", str(output_path),
    ]
    _run(pass2_cmd)
    logger.info("Loudness normalisation complete → %s", output_path)


def trim_silence(
    input_path: Path | str,
    output_path: Path | str,
    threshold_db: float = -40.0,
    min_silence_ms: int = 500,
) -> None:
    """Trim leading and trailing silence using ffmpeg ``silenceremove``.

    Args:
        input_path:      Path to the input audio file.
        output_path:     Path where the trimmed output will be written.
        threshold_db:    Silence threshold in dBFS (default ``-40``).
                         Samples quieter than this level are considered silent.
        min_silence_ms:  Minimum duration of a silence region to be trimmed,
                         in milliseconds (default ``500``).

    Raises:
        RuntimeError: If ``ffmpeg`` is not installed or the command fails.
    """
    ffmpeg = _require_ffmpeg()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    min_silence_s = min_silence_ms / 1000.0
    # Two silenceremove filters: one for the start, one reversed for the end
    af = (
        f"silenceremove=start_periods=1:start_silence={min_silence_s}"
        f":start_threshold={threshold_db}dB"
        f",areverse"
        f",silenceremove=start_periods=1:start_silence={min_silence_s}"
        f":start_threshold={threshold_db}dB"
        f",areverse"
    )
    cmd = [
        ffmpeg, "-nostdin", "-i", str(input_path),
        "-af", af,
        "-y", str(output_path),
    ]
    logger.info("Trimming silence from %s → %s …", input_path, output_path)
    _run(cmd)
    logger.info("Silence trim complete → %s", output_path)


def resample(
    input_path: Path | str,
    output_path: Path | str,
    target_rate: int = 22050,
    channels: int = 1,
) -> None:
    """Resample and optionally convert channel count using ffmpeg.

    Args:
        input_path:   Path to the input audio file.
        output_path:  Path where the resampled output will be written.
        target_rate:  Target sample rate in Hz (default ``22050``).
        channels:     Number of output channels; ``1`` for mono (default).

    Raises:
        RuntimeError: If ``ffmpeg`` is not installed or the command fails.
    """
    ffmpeg = _require_ffmpeg()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg, "-nostdin", "-i", str(input_path),
        "-ar", str(target_rate),
        "-ac", str(channels),
        "-y", str(output_path),
    ]
    logger.info(
        "Resampling %s → %s (%d Hz, %d ch) …",
        input_path, output_path, target_rate, channels,
    )
    _run(cmd)
    logger.info("Resample complete → %s", output_path)


def get_duration(file_path: Path | str) -> float:
    """Return the duration of an audio file in seconds using ``ffprobe``.

    Args:
        file_path: Path to the audio file.

    Returns:
        Duration in seconds as a float.

    Raises:
        RuntimeError: If ``ffprobe`` is not installed, the file is unreadable,
                      or duration information cannot be extracted.
    """
    ffprobe = _require_ffprobe()
    file_path = Path(file_path)

    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]
    result = _run(cmd)
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"ffprobe returned non-JSON output for {file_path}: {exc}"
        ) from exc

    # Prefer format-level duration (most reliable)
    fmt = info.get("format", {})
    if "duration" in fmt:
        return float(fmt["duration"])

    # Fall back to the first audio stream
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio" and "duration" in stream:
            return float(stream["duration"])

    raise RuntimeError(
        f"Could not extract duration from {file_path}. "
        f"ffprobe output:\n{result.stdout}"
    )


def process_chapter(
    input_path: Path | str,
    output_path: Path | str,
    config: AudioConfig,
) -> None:
    """Apply the full audio post-processing chain to a single chapter file.

    Processing order:

    1. **Trim silence** (if :attr:`~audiobook_forge.config.AudioConfig.trim_silence`
       is enabled).
    2. **Normalise loudness** (if
       :attr:`~audiobook_forge.config.AudioConfig.normalize_loudness` is enabled).
    3. **Resample** to
       :attr:`~audiobook_forge.config.AudioConfig.sample_rate`.

    Intermediate files are written to a ``.tmp`` sibling directory of
    *output_path* and cleaned up on completion.

    Args:
        input_path:  Path to the raw chapter WAV produced by a TTS engine.
        output_path: Destination path for the fully processed chapter audio.
        config:      :class:`~audiobook_forge.config.AudioConfig` instance
                     controlling which steps to apply.

    Raises:
        RuntimeError: If ``ffmpeg`` / ``ffprobe`` are missing or any step
                      fails.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a temporary directory alongside the output file
    tmp_dir = output_path.parent / ".postproc_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    current = input_path

    try:
        step_n = 0

        # Step 1: trim silence
        if config.trim_silence:
            step_n += 1
            trimmed = tmp_dir / f"{output_path.stem}_step{step_n}_trimmed.wav"
            trim_silence(current, trimmed)
            current = trimmed

        # Step 2: normalise loudness
        if config.normalize_loudness:
            step_n += 1
            normed = tmp_dir / f"{output_path.stem}_step{step_n}_normed.wav"
            normalize_loudness(
                current, normed, target_lufs=config.loudness_target_lufs
            )
            current = normed

        # Step 3: resample
        step_n += 1
        resampled = tmp_dir / f"{output_path.stem}_step{step_n}_resampled.wav"
        resample(current, resampled, target_rate=config.sample_rate, channels=1)
        current = resampled

        # Move final result into place
        import shutil as _shutil
        _shutil.copy2(str(current), str(output_path))
        logger.info("process_chapter complete → %s", output_path)

    finally:
        # Clean up intermediates
        for tmp_file in tmp_dir.glob(f"{output_path.stem}_step*"):
            try:
                tmp_file.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            tmp_dir.rmdir()  # only succeeds if empty
        except OSError:
            pass
