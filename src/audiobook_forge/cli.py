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
        "forge":  _cmd_forge,
        "status": _cmd_status,
        "reset":  _cmd_reset,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
