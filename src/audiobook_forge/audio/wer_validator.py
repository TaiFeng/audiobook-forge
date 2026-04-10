"""WER (Word Error Rate) validation — transcribe generated audio and compare
against source text to detect TTS errors (skipped words, hallucinations,
mispronunciations).

Uses `faster-whisper` for GPU-accelerated local transcription and `jiwer` for
alignment-based WER/CER scoring.

Public API
----------
* :func:`validate_chapter`  — score a single chapter audio file against its
  source text and return a :class:`ValidationResult`.
* :func:`validate_book`     — score all chapter audio files and return a
  :class:`BookValidationReport` with per-chapter and aggregate metrics.
* :func:`format_report`     — render a :class:`BookValidationReport` as a
  human-readable string.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """WER / CER metrics for a single chapter."""

    chapter_index: int
    chapter_title: str
    wer: float                  # Word Error Rate (0.0 = perfect)
    mer: float                  # Match Error Rate
    wil: float                  # Word Information Lost
    cer: float                  # Character Error Rate
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0
    reference_words: int = 0
    hypothesis_words: int = 0
    duration_seconds: float = 0.0
    transcript: str = ""        # Raw Whisper transcript for review
    flagged: bool = False       # True if WER exceeds threshold
    flag_reason: str = ""


@dataclass
class BookValidationReport:
    """Aggregate validation report for an entire book."""

    book_title: str
    chapters: list[ValidationResult] = field(default_factory=list)
    aggregate_wer: float = 0.0
    aggregate_cer: float = 0.0
    total_reference_words: int = 0
    total_errors: int = 0
    flagged_chapters: int = 0
    whisper_model: str = ""


# ---------------------------------------------------------------------------
# Text normalization for WER comparison
# ---------------------------------------------------------------------------

def _normalize_for_wer(text: str) -> str:
    """Normalize text for fair WER comparison.

    Both the reference (source text) and hypothesis (Whisper transcript) go
    through this so that trivial formatting differences don't inflate WER.
    """
    text = text.lower()
    # Remove common TTS / emotion annotation artifacts
    text = re.sub(r"\[.*?\]", "", text)           # [tag] inline tags
    text = re.sub(r"\(.*?\)", "", text)            # (parenthetical tags)
    # Expand common contractions Whisper may produce differently
    # (intentionally minimal — jiwer's alignment handles most cases)
    # Remove punctuation
    text = re.sub(r"[^\w\s']", " ", text)          # keep apostrophes
    text = re.sub(r"'(?!\w)|(?<!\w)'", " ", text)  # strip stray quotes
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def _transcribe(
    audio_path: Path,
    model_size: str = "base",
    device: str = "auto",
    compute_type: str = "auto",
    language: str = "en",
) -> tuple[str, float]:
    """Transcribe an audio file using faster-whisper.

    Args:
        audio_path:   Path to the audio file (WAV, MP3, etc.).
        model_size:   Whisper model size — ``tiny``, ``base``, ``small``,
                      ``medium``, ``large-v3``, ``large-v3-turbo``.
        device:       ``"auto"`` (GPU if available, else CPU), ``"cuda"``,
                      or ``"cpu"``.
        compute_type: ``"auto"``, ``"float16"``, ``"int8"``, ``"int8_float16"``.
        language:     ISO 639-1 language code.

    Returns:
        (transcript_text, duration_seconds)

    Raises:
        RuntimeError: If faster-whisper is not installed.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper is not installed.  Install it with:\n"
            "  pip install faster-whisper\n"
            "For GPU acceleration, ensure CUDA is available."
        )

    # Resolve device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # Resolve compute type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    logger.info(
        "Transcribing %s with Whisper %s (%s / %s) …",
        audio_path.name, model_size, device, compute_type,
    )

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language=language,
        vad_filter=True,               # skip silence
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
    )

    # Collect transcript
    full_text_parts: list[str] = []
    for segment in segments:
        full_text_parts.append(segment.text.strip())

    transcript = " ".join(full_text_parts)
    duration = info.duration or 0.0

    logger.info(
        "Transcription complete: %d words, %.1f s audio.",
        len(transcript.split()), duration,
    )

    return transcript, duration


# ---------------------------------------------------------------------------
# WER computation
# ---------------------------------------------------------------------------

def _compute_wer(reference: str, hypothesis: str) -> dict[str, Any]:
    """Compute WER and related metrics using jiwer.

    Args:
        reference:  Normalized source text.
        hypothesis: Normalized Whisper transcript.

    Returns:
        Dict with keys: wer, mer, wil, cer, substitutions, deletions,
        insertions, hits, ref_words, hyp_words.

    Raises:
        RuntimeError: If jiwer is not installed.
    """
    try:
        import jiwer
    except ImportError:
        raise RuntimeError(
            "jiwer is not installed.  Install it with:\n"
            "  pip install jiwer"
        )

    # Word-level metrics
    word_output = jiwer.process_words(reference, hypothesis)

    # Character-level metric
    cer_val = 0.0
    try:
        char_output = jiwer.process_characters(reference, hypothesis)
        cer_val = char_output.cer
    except Exception:
        # jiwer < 3.0 may not have process_characters
        try:
            cer_val = jiwer.cer(reference, hypothesis)
        except Exception:
            pass

    ref_words = len(reference.split())
    hyp_words = len(hypothesis.split())

    return {
        "wer": word_output.wer,
        "mer": word_output.mer,
        "wil": word_output.wil,
        "cer": cer_val,
        "substitutions": word_output.substitutions,
        "deletions": word_output.deletions,
        "insertions": word_output.insertions,
        "hits": word_output.hits,
        "ref_words": ref_words,
        "hyp_words": hyp_words,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_chapter(
    audio_path: Path | str,
    source_text: str,
    chapter_index: int = 0,
    chapter_title: str = "",
    model_size: str = "base",
    device: str = "auto",
    compute_type: str = "auto",
    language: str = "en",
    wer_threshold: float = 0.15,
) -> ValidationResult:
    """Transcribe a chapter audio file and compute WER against source text.

    Args:
        audio_path:     Path to the chapter WAV/audio file.
        source_text:    Original text that was sent to TTS.
        chapter_index:  Chapter number (for reporting).
        chapter_title:  Chapter name (for reporting).
        model_size:     Whisper model size (default ``"base"`` — fast, ~1 GB).
        device:         ``"auto"``, ``"cuda"``, or ``"cpu"``.
        compute_type:   ``"auto"``, ``"float16"``, ``"int8"``.
        language:       Language code.
        wer_threshold:  Flag chapters with WER above this value.

    Returns:
        :class:`ValidationResult` with all metrics.
    """
    audio_path = Path(audio_path)

    # Transcribe
    transcript, duration = _transcribe(
        audio_path,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    # Normalize both sides
    ref_norm = _normalize_for_wer(source_text)
    hyp_norm = _normalize_for_wer(transcript)

    # Handle edge cases
    if not ref_norm.strip():
        logger.warning("Chapter %d has empty reference text — skipping WER.", chapter_index)
        return ValidationResult(
            chapter_index=chapter_index,
            chapter_title=chapter_title,
            wer=0.0, mer=0.0, wil=0.0, cer=0.0,
            duration_seconds=duration,
            transcript=transcript,
        )

    if not hyp_norm.strip():
        logger.warning("Chapter %d produced empty transcript.", chapter_index)
        return ValidationResult(
            chapter_index=chapter_index,
            chapter_title=chapter_title,
            wer=1.0, mer=1.0, wil=1.0, cer=1.0,
            reference_words=len(ref_norm.split()),
            deletions=len(ref_norm.split()),
            duration_seconds=duration,
            transcript="",
            flagged=True,
            flag_reason="Empty transcript — TTS may have produced silence.",
        )

    # Compute metrics
    metrics = _compute_wer(ref_norm, hyp_norm)

    flagged = metrics["wer"] > wer_threshold
    flag_reason = ""
    if flagged:
        flag_reason = (
            f"WER {metrics['wer']:.1%} exceeds threshold {wer_threshold:.0%} "
            f"({metrics['substitutions']}S/{metrics['deletions']}D/"
            f"{metrics['insertions']}I)"
        )

    result = ValidationResult(
        chapter_index=chapter_index,
        chapter_title=chapter_title,
        wer=metrics["wer"],
        mer=metrics["mer"],
        wil=metrics["wil"],
        cer=metrics["cer"],
        substitutions=metrics["substitutions"],
        deletions=metrics["deletions"],
        insertions=metrics["insertions"],
        hits=metrics["hits"],
        reference_words=metrics["ref_words"],
        hypothesis_words=metrics["hyp_words"],
        duration_seconds=duration,
        transcript=transcript,
        flagged=flagged,
        flag_reason=flag_reason,
    )

    level = logging.WARNING if flagged else logging.INFO
    logger.log(
        level,
        "Chapter %d %r — WER: %.1f%% CER: %.1f%% (%d words)%s",
        chapter_index + 1,
        chapter_title,
        metrics["wer"] * 100,
        metrics["cer"] * 100,
        metrics["ref_words"],
        f"  *** FLAGGED: {flag_reason}" if flagged else "",
    )

    return result


def validate_book(
    chapter_audio_files: list[dict[str, Any]],
    chapter_source_texts: list[str],
    book_title: str = "",
    model_size: str = "base",
    device: str = "auto",
    compute_type: str = "auto",
    language: str = "en",
    wer_threshold: float = 0.15,
) -> BookValidationReport:
    """Validate all chapter audio files against their source texts.

    Args:
        chapter_audio_files: List of dicts with ``"path"`` and ``"title"``
                             keys, in chapter order.
        chapter_source_texts: Source text for each chapter, matching order.
        book_title:          Book name for the report.
        model_size:          Whisper model size.
        device:              ``"auto"``, ``"cuda"``, or ``"cpu"``.
        compute_type:        ``"auto"``, ``"float16"``, ``"int8"``.
        language:            Language code.
        wer_threshold:       Flag chapters exceeding this WER.

    Returns:
        :class:`BookValidationReport` with per-chapter and aggregate scores.
    """
    report = BookValidationReport(
        book_title=book_title,
        whisper_model=model_size,
    )

    total_sub = 0
    total_del = 0
    total_ins = 0
    total_ref_words = 0
    total_ref_chars = 0
    total_cer_weighted = 0.0

    for i, (ch_info, src_text) in enumerate(
        zip(chapter_audio_files, chapter_source_texts)
    ):
        audio_path = ch_info.get("path", "")
        ch_title = ch_info.get("title", f"Chapter {i + 1}")

        if not audio_path or not Path(audio_path).exists():
            logger.warning("Chapter %d audio not found: %s — skipping.", i, audio_path)
            continue

        result = validate_chapter(
            audio_path=audio_path,
            source_text=src_text,
            chapter_index=i,
            chapter_title=ch_title,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            wer_threshold=wer_threshold,
        )

        report.chapters.append(result)
        total_sub += result.substitutions
        total_del += result.deletions
        total_ins += result.insertions
        total_ref_words += result.reference_words

        ref_chars = len(_normalize_for_wer(src_text))
        total_ref_chars += ref_chars
        total_cer_weighted += result.cer * ref_chars

        if result.flagged:
            report.flagged_chapters += 1

    # Aggregate WER = total errors / total reference words
    total_errors = total_sub + total_del + total_ins
    report.total_reference_words = total_ref_words
    report.total_errors = total_errors
    report.aggregate_wer = total_errors / total_ref_words if total_ref_words > 0 else 0.0
    report.aggregate_cer = total_cer_weighted / total_ref_chars if total_ref_chars > 0 else 0.0

    logger.info(
        "Book validation complete: aggregate WER %.1f%%, CER %.1f%%, "
        "%d/%d chapters flagged.",
        report.aggregate_wer * 100,
        report.aggregate_cer * 100,
        report.flagged_chapters,
        len(report.chapters),
    )

    return report


def format_report(report: BookValidationReport) -> str:
    """Render a :class:`BookValidationReport` as a human-readable string.

    Suitable for logging, printing to console, or saving to a text file.
    """
    lines: list[str] = [
        "=" * 70,
        f"  WER Validation Report: {report.book_title}",
        f"  Whisper model: {report.whisper_model}",
        "=" * 70,
        "",
        f"  Aggregate WER : {report.aggregate_wer:.1%}",
        f"  Aggregate CER : {report.aggregate_cer:.1%}",
        f"  Total words   : {report.total_reference_words:,}",
        f"  Total errors  : {report.total_errors:,}",
        f"  Flagged       : {report.flagged_chapters} / {len(report.chapters)} chapters",
        "",
        "-" * 70,
        f"  {'Ch':>3}  {'Title':<35}  {'WER':>6}  {'CER':>6}  {'Words':>6}  {'Flag':>4}",
        "-" * 70,
    ]

    for ch in report.chapters:
        flag_marker = " ***" if ch.flagged else ""
        title_trunc = (ch.chapter_title[:33] + "..") if len(ch.chapter_title) > 35 else ch.chapter_title
        lines.append(
            f"  {ch.chapter_index + 1:>3}  {title_trunc:<35}  "
            f"{ch.wer:>5.1%}  {ch.cer:>5.1%}  "
            f"{ch.reference_words:>6}{flag_marker}"
        )

    lines.append("-" * 70)

    # List flagged chapters with details
    flagged = [ch for ch in report.chapters if ch.flagged]
    if flagged:
        lines.append("")
        lines.append("  Flagged chapters (consider regenerating):")
        for ch in flagged:
            lines.append(f"    Ch {ch.chapter_index + 1}: {ch.flag_reason}")

    # Quality assessment
    lines.append("")
    wer_pct = report.aggregate_wer * 100
    if wer_pct <= 5:
        quality = "Excellent — near-perfect transcription fidelity."
    elif wer_pct <= 10:
        quality = "Good — minor deviations, likely acceptable for most listeners."
    elif wer_pct <= 15:
        quality = "Fair — noticeable errors; review flagged chapters."
    elif wer_pct <= 25:
        quality = "Poor — significant errors; consider re-generating affected chapters."
    else:
        quality = "Unacceptable — major issues; check TTS engine and text normalization."

    lines.append(f"  Quality: {quality}")
    lines.append("=" * 70)

    return "\n".join(lines)
