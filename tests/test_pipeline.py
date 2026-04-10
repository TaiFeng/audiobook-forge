#!/usr/bin/env python3
"""End-to-end test: process a short public-domain text sample.

Usage:
    python tests/test_pipeline.py [--engine kokoro|fish_audio|openai_compat] [--emotion]

Requirements:
    - ffmpeg on PATH
    - For Kokoro: pip install kokoro soundfile
    - For Fish Audio: running Fish Audio server on localhost:8080
    - For OpenAI-compat: running server on localhost:1234/v1
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Add project src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_text_processing():
    """Test text normalization, segmentation, and dialogue detection."""
    print("=" * 60)
    print("TEST: Text processing pipeline")
    print("=" * 60)

    sample = Path(PROJECT_ROOT / "samples" / "alice_chapter1.txt")
    if not sample.exists():
        print(f"SKIP: Sample file not found at {sample}")
        return False

    from audiobook_forge.ingestion.reader import read_book
    from audiobook_forge.processing.text_normalizer import normalize_text
    from audiobook_forge.processing.sentence_segmenter import segment_sentences, chunk_sentences
    from audiobook_forge.processing.dialogue_detector import detect_dialogue

    # Read book
    book = read_book(str(sample))
    print(f"  Title   : {book.title}")
    print(f"  Author  : {book.author}")
    print(f"  Chapters: {len(book.chapters)}")
    assert len(book.chapters) >= 1, "Expected at least 1 chapter"

    for ch in book.chapters:
        print(f"  Chapter {ch.index}: {ch.title!r} ({len(ch.text)} chars)")

    # Normalize
    text = normalize_text(book.chapters[0].text)
    print(f"\n  Normalized text length: {len(text)} chars")
    assert len(text) > 50, "Normalized text too short"

    # Segment
    sentences = segment_sentences(text)
    print(f"  Sentences: {len(sentences)}")
    assert len(sentences) >= 3, "Expected at least 3 sentences"

    for s in sentences[:5]:
        para_marker = " [P]" if s.is_paragraph_end else ""
        print(f"    [{s.index}]{para_marker} {s.text[:80]}...")

    # Chunk
    chunks = chunk_sentences(sentences, max_chars=400)
    print(f"  Chunks (max 400 chars): {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        total_chars = sum(len(s.text) for s in chunk)
        print(f"    Chunk {i}: {len(chunk)} sentences, {total_chars} chars")

    # Dialogue detection
    annotations = detect_dialogue(sentences)
    dialogue_count = sum(1 for a in annotations if a.contains_dialogue)
    print(f"  Dialogue sentences: {dialogue_count}/{len(annotations)}")

    for a in annotations[:5]:
        if a.contains_dialogue:
            print(f"    [{a.sentence_index}] mode={a.narration_mode} verb={a.attribution_verb!r}")

    print("\n  PASS: Text processing pipeline works correctly.\n")
    return True


def test_emotion_tagging():
    """Test the emotion tagging system (rules mode only)."""
    print("=" * 60)
    print("TEST: Emotion tagging (rules mode)")
    print("=" * 60)

    sample = Path(PROJECT_ROOT / "samples" / "alice_chapter1.txt")
    if not sample.exists():
        print(f"SKIP: Sample file not found at {sample}")
        return False

    from audiobook_forge.ingestion.reader import read_book
    from audiobook_forge.processing.text_normalizer import normalize_text
    from audiobook_forge.processing.sentence_segmenter import segment_sentences
    from audiobook_forge.processing.dialogue_detector import detect_dialogue
    from audiobook_forge.processing.emotion_tagger import tag_emotions
    from audiobook_forge.config import EmotionConfig, AudioConfig

    book = read_book(str(sample))
    text = normalize_text(book.chapters[0].text)
    sentences = segment_sentences(text)
    dialogue_annotations = detect_dialogue(sentences)

    emotion_cfg = EmotionConfig(enabled=True, mode="rules", max_intensity=0.7)
    audio_cfg = AudioConfig()

    annotated = tag_emotions(sentences, dialogue_annotations, emotion_cfg, audio_cfg)
    print(f"  Total annotated sentences: {len(annotated)}")

    non_neutral = [a for a in annotated if a.emotion != "neutral"]
    print(f"  Non-neutral emotions: {len(non_neutral)}/{len(annotated)}")

    for a in annotated[:10]:
        if a.emotion != "neutral":
            print(f"    [{a.narration_mode}] emotion={a.emotion} intensity={a.intensity:.2f}: {a.text[:60]}...")

    # Verify guardrails
    non_neutral_ratio = len(non_neutral) / len(annotated) if annotated else 0
    print(f"  Non-neutral ratio: {non_neutral_ratio:.1%}")
    assert non_neutral_ratio <= 0.35, f"Anti-melodrama guardrail failed: {non_neutral_ratio:.1%} > 35%"

    max_intensity = max((a.intensity for a in annotated), default=0)
    print(f"  Max intensity: {max_intensity:.2f}")
    assert max_intensity <= 0.7, f"Max intensity cap failed: {max_intensity} > 0.7"

    print("\n  PASS: Emotion tagging works correctly.\n")
    return True


def test_full_pipeline(engine: str = "kokoro", emotion: bool = False):
    """Test the full pipeline end-to-end with a short sample."""
    print("=" * 60)
    print(f"TEST: Full pipeline (engine={engine}, emotion={emotion})")
    print("=" * 60)

    sample = Path(PROJECT_ROOT / "samples" / "alice_chapter1.txt")
    if not sample.exists():
        print(f"SKIP: Sample file not found at {sample}")
        return False

    with tempfile.TemporaryDirectory(prefix="audiobook_forge_test_") as tmpdir:
        from audiobook_forge.config import load_config

        config_path = PROJECT_ROOT / "config.yaml"
        overrides = {
            "input": {"file": str(sample)},
            "project": {
                "name": "Alice Test",
                "output_dir": str(Path(tmpdir) / "output"),
                "temp_dir": str(Path(tmpdir) / "tmp"),
            },
            "tts": {"engine": engine},
            "emotion": {"enabled": emotion, "mode": "rules"},
            "resume": {"checkpoint_file": str(Path(tmpdir) / ".checkpoint.json")},
            "logging": {"log_file": str(Path(tmpdir) / "test.log")},
        }

        cfg = load_config(config_path, overrides)
        cfg.input_file = str(sample)

        from audiobook_forge.pipeline import AudiobookPipeline
        pipeline = AudiobookPipeline(cfg)

        try:
            m4b_path = pipeline.run(str(sample))
            print(f"\n  Output: {m4b_path}")
            assert m4b_path.exists(), "M4B file not created"
            assert m4b_path.stat().st_size > 0, "M4B file is empty"
            print(f"  Size: {m4b_path.stat().st_size / 1024:.1f} KB")
            print("\n  PASS: Full pipeline completed successfully.\n")
            return True
        except Exception as e:
            print(f"\n  FAIL: {e}\n")
            return False


def main():
    parser = argparse.ArgumentParser(description="Audiobook Forge test suite")
    parser.add_argument("--engine", default="kokoro", choices=["kokoro", "fish_audio", "openai_compat"])
    parser.add_argument("--emotion", action="store_true", help="Enable emotion tagging")
    parser.add_argument("--text-only", action="store_true", help="Only run text processing tests (no TTS)")
    args = parser.parse_args()

    results = []

    # Always run text processing tests
    results.append(("Text Processing", test_text_processing()))
    results.append(("Emotion Tagging", test_emotion_tagging()))

    # Run full pipeline test unless --text-only
    if not args.text_only:
        results.append(("Full Pipeline", test_full_pipeline(args.engine, args.emotion)))

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL/SKIP"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
