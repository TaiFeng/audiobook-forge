"""Command-line interface for Audiobook Forge.

Usage examples::

    audiobook-forge forge --input book.epub --engine kokoro --emotion
    audiobook-forge status
    audiobook-forge reset --confirm

Can also be invoked as ``python -m audiobook_forge``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Subcommand: forge
# ---------------------------------------------------------------------------

def _cmd_forge(args: argparse.Namespace) -> int:
    """Run the full conversion pipeline."""
    from audiobook_forge.config import load_config

    overrides: dict[str, Any] = {}

    # --- Input file --------------------------------------------------------
    if args.input:
        overrides["input"] = {"file": args.input}

    # --- Output directory --------------------------------------------------
    if args.output:
        overrides.setdefault("project", {})["output_dir"] = args.output

    # --- TTS engine --------------------------------------------------------
    if args.engine:
        overrides.setdefault("tts", {})["engine"] = args.engine

    # --- Voice override (applied to all backends that support it) ----------
    if args.voice:
        overrides.setdefault("tts", {}).setdefault("kokoro", {})["voice"] = args.voice
        overrides["tts"].setdefault("openai_compat", {})["voice"] = args.voice

    # --- Emotion tagging ---------------------------------------------------
    if args.emotion:
        overrides.setdefault("emotion", {})["enabled"] = True

    # --- WER validation ----------------------------------------------------
    if args.validate:
        overrides.setdefault("validation", {})["enabled"] = True
    if args.whisper_model:
        overrides.setdefault("validation", {})["whisper_model"] = args.whisper_model

    # --- Resume behaviour --------------------------------------------------
    if args.no_resume:
        overrides.setdefault("resume", {})["enabled"] = False

    # --- Metadata overrides ------------------------------------------------
    if args.title:
        overrides.setdefault("project", {})["name"] = args.title
    if args.author:
        overrides.setdefault("project", {})["author"] = args.author

    config_path = args.config or "config.yaml"
    cfg = load_config(config_path, overrides=overrides)

    # If --no-resume, wipe checkpoint before starting
    if args.no_resume:
        from audiobook_forge.checkpoint import CheckpointManager
        cp = CheckpointManager(cfg.resume.checkpoint_file)
        cp.reset()

    from audiobook_forge.pipeline import AudiobookPipeline

    pipeline = AudiobookPipeline(cfg)
    try:
        output_path = pipeline.run(input_file=args.input)
        print(f"Done: {output_path}")
        return 0
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 2


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------

def _cmd_status(args: argparse.Namespace) -> int:
    """Print current checkpoint / progress status."""
    from audiobook_forge.config import load_config
    from audiobook_forge.checkpoint import CheckpointManager

    config_path = args.config or "config.yaml"
    cfg = load_config(config_path)
    cp  = CheckpointManager(cfg.resume.checkpoint_file)

    progress = cp.get_progress()

    print(f"Book     : {progress['book_title'] or '(not started)'}")
    print(f"Chapters : {progress['completed_chapters']} / {progress['total_chapters']} "
          f"({progress['percent']}%)")
    print(f"M4B done : {'yes' if progress['m4b_assembled'] else 'no'}")

    # Per-chapter details
    if cp.state.chapters:
        print("\nChapter details:")
        for ch in cp.state.chapters:
            status = "✔" if ch.get("completed") else "…"
            chunks_done  = ch.get("completed_chunks", 0)
            chunks_total = ch.get("total_chunks", "?")
            print(
                f"  [{status}] Ch {ch['chapter_index'] + 1:>3}: "
                f"{ch.get('chapter_title', '')[:50]:<50} "
                f"({chunks_done}/{chunks_total} chunks)"
            )

    return 0


# ---------------------------------------------------------------------------
# Subcommand: validate
# ---------------------------------------------------------------------------

def _cmd_validate(args: argparse.Namespace) -> int:
    """Run standalone WER validation on existing chapter audio files."""
    from audiobook_forge.config import load_config
    from audiobook_forge.checkpoint import CheckpointManager

    config_path = args.config or "config.yaml"
    cfg = load_config(config_path)
    cp = CheckpointManager(cfg.resume.checkpoint_file)

    if not cp.state.chapters:
        print("No chapters found in checkpoint. Run 'forge' first.", file=sys.stderr)
        return 1

    # Gather completed chapter audio files and source texts
    completed_chapters = [
        ch for ch in cp.state.chapters if ch.get("completed")
    ]
    if not completed_chapters:
        print("No completed chapters to validate.", file=sys.stderr)
        return 1

    # Read book to get source texts
    input_path = cp.state.input_file
    if not input_path or not Path(input_path).exists():
        if args.input:
            input_path = args.input
        else:
            print(
                "Cannot find original input file. Provide it with --input.",
                file=sys.stderr,
            )
            return 1

    from audiobook_forge.ingestion.reader import read_book
    book = read_book(input_path)

    chapter_infos = []
    chapter_texts = []
    for ch_data in completed_chapters:
        audio_file = ch_data.get("audio_file", "")
        ch_idx = ch_data.get("chapter_index", 0)
        if audio_file and Path(audio_file).exists() and ch_idx < len(book.chapters):
            chapter_infos.append({
                "path": audio_file,
                "title": ch_data.get("chapter_title", f"Chapter {ch_idx + 1}"),
            })
            chapter_texts.append(book.chapters[ch_idx].text)

    if not chapter_infos:
        print("No chapter audio files found on disk.", file=sys.stderr)
        return 1

    whisper_model = args.whisper_model or cfg.validation.whisper_model
    wer_threshold = cfg.validation.wer_threshold

    try:
        from audiobook_forge.audio.wer_validator import validate_book, format_report
    except ImportError as exc:
        print(f"Missing dependency: {exc}\nInstall: pip install faster-whisper jiwer", file=sys.stderr)
        return 1

    print(f"Validating {len(chapter_infos)} chapters with Whisper {whisper_model}...\n")

    report = validate_book(
        chapter_audio_files=chapter_infos,
        chapter_source_texts=chapter_texts,
        book_title=cp.state.book_title,
        model_size=whisper_model,
        device=cfg.validation.device,
        compute_type=cfg.validation.compute_type,
        language=cfg.validation.language,
        wer_threshold=wer_threshold,
    )

    report_text = format_report(report)
    print(report_text)

    # Save report
    report_file = cfg.validation.report_file
    if report_file:
        report_path = Path(report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")
        print(f"\nReport saved to {report_path}")

    return 0 if report.flagged_chapters == 0 else 1


# ---------------------------------------------------------------------------
# Subcommand: reset
# ---------------------------------------------------------------------------

def _cmd_reset(args: argparse.Namespace) -> int:
    """Clear the checkpoint so the pipeline starts fresh."""
    from audiobook_forge.config import load_config
    from audiobook_forge.checkpoint import CheckpointManager

    config_path = args.config or "config.yaml"
    cfg = load_config(config_path)
    cp  = CheckpointManager(cfg.resume.checkpoint_file)

    if not args.confirm:
        answer = input(
            "This will delete all checkpoint data and force a full re-run.\n"
            "Type 'yes' to confirm: "
        ).strip().lower()
        if answer != "yes":
            print("Reset cancelled.")
            return 0

    cp.reset()
    print(f"Checkpoint cleared: {cfg.resume.checkpoint_file}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiobook-forge",
        description="Convert EPUB/TXT books to M4B audiobooks using a local TTS engine.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # -----------------------------------------------------------------------
    # forge
    # -----------------------------------------------------------------------
    forge_p = sub.add_parser(
        "forge",
        help="Run the full conversion pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    forge_p.add_argument(
        "--input", "-i",
        metavar="FILE",
        required=False,
        help="Input .epub or .txt file (required unless set in config.yaml).",
    )
    forge_p.add_argument(
        "--config", "-c",
        metavar="FILE",
        default="config.yaml",
        help="Path to YAML config file.",
    )
    forge_p.add_argument(
        "--output", "-o",
        metavar="DIR",
        help="Output directory override.",
    )
    forge_p.add_argument(
        "--engine",
        choices=["kokoro", "fish_audio", "openai_compat"],
        help="TTS engine to use.",
    )
    forge_p.add_argument(
        "--voice",
        metavar="VOICE",
        help="Voice name (engine-specific).",
    )
    forge_p.add_argument(
        "--emotion",
        action="store_true",
        default=False,
        help="Enable emotion tagging.",
    )
    forge_p.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Run WER validation after generation (requires faster-whisper + jiwer).",
    )
    forge_p.add_argument(
        "--whisper-model",
        metavar="MODEL",
        default=None,
        help="Whisper model for validation: tiny, base, small, medium, large-v3.",
    )
    forge_p.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Disable resume — start fresh even if a checkpoint exists.",
    )
    forge_p.add_argument(
        "--title",
        metavar="TITLE",
        help="Override book title (used in M4B metadata and output filename).",
    )
    forge_p.add_argument(
        "--author",
        metavar="AUTHOR",
        help="Override author name.",
    )

    # -----------------------------------------------------------------------
    # status
    # -----------------------------------------------------------------------
    status_p = sub.add_parser(
        "status",
        help="Show current checkpoint / progress status.",
    )
    status_p.add_argument(
        "--config", "-c",
        metavar="FILE",
        default="config.yaml",
        help="Path to YAML config file.",
    )

    # -----------------------------------------------------------------------
    # validate
    # -----------------------------------------------------------------------
    validate_p = sub.add_parser(
        "validate",
        help="Run WER validation on existing chapter audio files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    validate_p.add_argument(
        "--config", "-c",
        metavar="FILE",
        default="config.yaml",
        help="Path to YAML config file.",
    )
    validate_p.add_argument(
        "--input", "-i",
        metavar="FILE",
        default=None,
        help="Path to original input file (if not in checkpoint).",
    )
    validate_p.add_argument(
        "--whisper-model",
        metavar="MODEL",
        default=None,
        help="Whisper model: tiny, base, small, medium, large-v3 (default: from config).",
    )

    # -----------------------------------------------------------------------
    # reset
    # -----------------------------------------------------------------------
    reset_p = sub.add_parser(
        "reset",
        help="Clear checkpoint (force a fresh run next time).",
    )
    reset_p.add_argument(
        "--config", "-c",
        metavar="FILE",
        default="config.yaml",
        help="Path to YAML config file.",
    )
    reset_p.add_argument(
        "--confirm",
        action="store_true",
        default=False,
        help="Skip the interactive confirmation prompt.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments, apply config overrides, and dispatch to subcommand."""
    parser = _build_parser()
    args   = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "forge":    _cmd_forge,
        "status":   _cmd_status,
        "validate": _cmd_validate,
        "reset":    _cmd_reset,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
