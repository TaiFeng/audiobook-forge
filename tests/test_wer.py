#!/usr/bin/env python3
"""Unit tests for WER validation module — text normalization and scoring.

These tests do NOT require faster-whisper (no audio transcription).
They verify the WER computation and normalization logic using jiwer directly.

Usage:
    python tests/test_wer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_normalize_for_wer():
    """Test that WER text normalization handles common edge cases."""
    print("=" * 60)
    print("TEST: WER text normalization")
    print("=" * 60)

    from audiobook_forge.audio.wer_validator import _normalize_for_wer

    # Basic lowercasing and punctuation removal
    assert _normalize_for_wer("Hello, World!") == "hello world"

    # Inline tags stripped
    assert _normalize_for_wer("[excited] Hello there!") == "hello there"
    assert _normalize_for_wer("(laughing) Ha ha") == "ha ha"

    # Smart quotes and em-dashes
    assert "hello" in _normalize_for_wer("\u201cHello\u201d")

    # Whitespace collapse
    assert _normalize_for_wer("  too   many   spaces  ") == "too many spaces"

    # Apostrophes preserved in contractions
    norm = _normalize_for_wer("I can't believe it's real")
    assert "can't" in norm or "cant" in norm  # depends on regex

    print("  PASS: normalization handles all edge cases.\n")
    return True


def test_wer_computation():
    """Test WER computation with known reference/hypothesis pairs."""
    print("=" * 60)
    print("TEST: WER computation")
    print("=" * 60)

    from audiobook_forge.audio.wer_validator import _compute_wer

    # Perfect match
    metrics = _compute_wer("hello world", "hello world")
    assert metrics["wer"] == 0.0, f"Expected 0.0, got {metrics['wer']}"
    assert metrics["hits"] == 2
    print(f"  Perfect match: WER={metrics['wer']:.1%} (expected 0%)")

    # One substitution in two words
    metrics = _compute_wer("hello world", "hello duck")
    assert abs(metrics["wer"] - 0.5) < 0.01, f"Expected ~0.5, got {metrics['wer']}"
    assert metrics["substitutions"] == 1
    print(f"  One substitution: WER={metrics['wer']:.1%} (expected 50%)")

    # Deletion
    metrics = _compute_wer("the quick brown fox", "the brown fox")
    assert metrics["deletions"] >= 1
    print(f"  With deletion: WER={metrics['wer']:.1%}, deletions={metrics['deletions']}")

    # Insertion
    metrics = _compute_wer("hello world", "hello beautiful world")
    assert metrics["insertions"] >= 1
    print(f"  With insertion: WER={metrics['wer']:.1%}, insertions={metrics['insertions']}")

    # Longer passage
    ref = "alice was beginning to get very tired of sitting by her sister on the bank"
    hyp = "alice was beginning to get very tired of sitting by her sister on the banks"
    metrics = _compute_wer(ref, hyp)
    assert metrics["wer"] < 0.1  # Just one word changed
    print(f"  Long passage (1 error in 14 words): WER={metrics['wer']:.1%}")

    print("\n  PASS: WER computation is accurate.\n")
    return True


def test_validation_result_structure():
    """Test that validate_chapter returns correct structure (without Whisper)."""
    print("=" * 60)
    print("TEST: ValidationResult structure")
    print("=" * 60)

    from audiobook_forge.audio.wer_validator import (
        ValidationResult, BookValidationReport, format_report
    )

    # Build a mock report
    ch1 = ValidationResult(
        chapter_index=0,
        chapter_title="Chapter 1: Down the Rabbit-Hole",
        wer=0.08,
        mer=0.07,
        wil=0.12,
        cer=0.04,
        substitutions=5,
        deletions=2,
        insertions=1,
        hits=92,
        reference_words=100,
        hypothesis_words=99,
        duration_seconds=120.0,
        transcript="alice was beginning to get very tired...",
        flagged=False,
    )

    ch2 = ValidationResult(
        chapter_index=1,
        chapter_title="Chapter 2: The Pool of Tears",
        wer=0.22,
        mer=0.20,
        wil=0.30,
        cer=0.15,
        substitutions=12,
        deletions=8,
        insertions=2,
        hits=78,
        reference_words=100,
        hypothesis_words=92,
        duration_seconds=115.0,
        transcript="curiouser and curiouser cried alice...",
        flagged=True,
        flag_reason="WER 22.0% exceeds threshold 15% (12S/8D/2I)",
    )

    report = BookValidationReport(
        book_title="Alice's Adventures in Wonderland",
        chapters=[ch1, ch2],
        aggregate_wer=0.15,
        aggregate_cer=0.095,
        total_reference_words=200,
        total_errors=30,
        flagged_chapters=1,
        whisper_model="base",
    )

    # Format report
    report_text = format_report(report)
    print(report_text)

    # Verify content
    assert "Alice" in report_text
    assert "15.0%" in report_text  # aggregate WER
    assert "FLAGGED" in report_text or "***" in report_text or "flagged" in report_text.lower()
    assert "Chapter 2" in report_text

    print("\n  PASS: Report structure and formatting correct.\n")
    return True


def main():
    results = []

    results.append(("WER Normalization", test_normalize_for_wer()))
    results.append(("WER Computation", test_wer_computation()))
    results.append(("Report Structure", test_validation_result_structure()))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    sys.exit(0 if all(r[1] for r in results) else 1)


if __name__ == "__main__":
    main()
