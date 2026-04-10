"""Main pipeline orchestrator for Audiobook Forge.

Coordinates every stage from raw input file to final .m4b output:
  ingestion → text normalisation → sentence segmentation →
  dialogue detection → emotion tagging → TTS synthesis →
  post-processing → M4B assembly.

The pipeline is fully resumable via :class:`~audiobook_forge.checkpoint.CheckpointManager`.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from audiobook_forge.checkpoint import CheckpointManager
from audiobook_forge.config import ForgeConfig
from audiobook_forge.tts import get_engine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _setup_logging(config: ForgeConfig) -> None:
    """Configure root logger to emit to console and optionally a log file."""
    level = getattr(logging, config.logging.level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    log_file = config.logging.log_file
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------

def _sha256_file(file_path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Audio concatenation helper (ffmpeg)
# ---------------------------------------------------------------------------

def _concat_audio_files(input_paths: list[Path], output_path: Path) -> None:
    """Concatenate *input_paths* into *output_path* using ffmpeg concat demuxer."""
    import subprocess

    list_file = output_path.with_suffix(".concat_list.txt")
    with open(list_file, "w", encoding="utf-8") as fh:
        for p in input_paths:
            fh.write(f"file '{p.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_file.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg concat failed for {output_path}:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class AudiobookPipeline:
    """End-to-end audiobook generation pipeline.

    Args:
        config: Fully populated :class:`~audiobook_forge.config.ForgeConfig`.
    """

    def __init__(self, config: ForgeConfig) -> None:
        self.config = config

        _setup_logging(config)

        self.output_dir = Path(config.project.output_dir)
        self.temp_dir   = Path(config.project.temp_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint = CheckpointManager(config.resume.checkpoint_file)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, input_file: str | None = None) -> Path:
        """Execute the full pipeline and return the path to the .m4b file.

        Args:
            input_file: Override for ``config.input_file``.

        Returns:
            Path to the assembled .m4b audiobook.

        Raises:
            ValueError: If no input file is specified.
            SystemExit: On :class:`KeyboardInterrupt` (checkpoint is saved first).
        """
        pipeline_start = time.monotonic()

        input_path = Path(input_file or self.config.input_file)
        if not input_path.exists():
            raise ValueError(f"Input file not found: {input_path}")

        logger.info("=== Audiobook Forge ===")
        logger.info("Input : %s", input_path)
        logger.info("Output: %s", self.output_dir)

        try:
            return self._run(input_path, pipeline_start)
        except KeyboardInterrupt:
            logger.warning("Interrupted — checkpoint saved. Re-run to resume.")
            sys.exit(0)

    # ------------------------------------------------------------------
    # Internal orchestration
    # ------------------------------------------------------------------

    def _run(self, input_path: Path, pipeline_start: float) -> Path:
        # --- Step 1: Read book -------------------------------------------
        logger.info("Reading book from %s …", input_path)
        from audiobook_forge.ingestion.reader import read_book

        book = read_book(str(input_path))

        # Override title/author from config if set
        title  = self.config.project.name or book.title or input_path.stem
        author = self.config.project.author or book.author or ""

        logger.info("Book: %r by %r (%d chapters)", title, author, len(book.chapters))

        # --- Step 2: SHA-256 hash for change detection -------------------
        logger.debug("Computing SHA-256 of input file …")
        input_hash = _sha256_file(input_path)

        # --- Step 3: Initialise / restore checkpoint ---------------------
        chapter_titles = [ch.title or f"Chapter {ch.index + 1}" for ch in book.chapters]
        self.checkpoint.initialize(title, str(input_path), input_hash, chapter_titles)

        if self.checkpoint.state.m4b_assembled:
            # Already finished — find the .m4b and return it
            existing = list(self.output_dir.glob("*.m4b"))
            if existing:
                logger.info("M4B already assembled: %s", existing[0])
                return existing[0]

        # --- Step 4: Extract cover image ----------------------------------
        cover_image_path: Path | None = None
        if book.cover_image_data and book.cover_image_ext:
            cover_image_path = self.temp_dir / f"cover{book.cover_image_ext}"
            cover_image_path.write_bytes(book.cover_image_data)
            logger.debug("Cover image saved to %s", cover_image_path)

        # --- Step 5: Initialise TTS engine --------------------------------
        engine = get_engine(self.config.tts.engine, self.config)
        logger.info("Initialising TTS engine: %s", engine.name)
        engine.initialize()

        # --- Step 6: Process chapters ------------------------------------
        chapter_audio_files: list[Path] = []
        chapter_times: list[float] = []
        total_chapters = len(book.chapters)

        for chapter in book.chapters:
            ch_idx   = chapter.index
            ch_title = chapter_titles[ch_idx]

            # Recover already-completed chapter audio
            if self.checkpoint.is_chapter_done(ch_idx):
                existing_audio = self.checkpoint.state.chapters[ch_idx].get("audio_file", "")
                if existing_audio and Path(existing_audio).exists():
                    chapter_audio_files.append(Path(existing_audio))
                    logger.info(
                        "Skipping chapter %d/%d (already done): %s",
                        ch_idx + 1, total_chapters, ch_title,
                    )
                    continue

            logger.info(
                "Processing chapter %d/%d: %s",
                ch_idx + 1, total_chapters, ch_title,
            )
            ch_start = time.monotonic()

            try:
                chapter_audio = self._process_chapter(chapter, ch_idx, engine)
            except Exception:
                logger.exception(
                    "Error processing chapter %d (%s) — skipping.",
                    ch_idx + 1, ch_title,
                )
                continue

            chapter_audio_files.append(chapter_audio)
            ch_elapsed = time.monotonic() - ch_start
            chapter_times.append(ch_elapsed)

            # Progress report
            done = self.checkpoint.state.completed_chapters
            pct  = done / total_chapters * 100 if total_chapters else 0.0
            avg  = sum(chapter_times) / len(chapter_times)
            remaining = avg * (total_chapters - done)
            logger.info(
                "Progress: %.1f%% | ~%.0f s remaining",
                pct, remaining,
            )

        # --- Step 7: Assemble M4B -----------------------------------------
        if not chapter_audio_files:
            raise RuntimeError("No chapter audio files produced — cannot assemble M4B.")

        safe_title = re.sub(r'[<>:"/\\|?*]', "_", title)  # type: ignore[name-defined]
        m4b_path = self.output_dir / f"{safe_title}.m4b"

        logger.info("Assembling M4B → %s …", m4b_path)

        metadata: dict[str, Any] = {
            "title":  title,
            "author": author,
            "cover":  str(cover_image_path) if cover_image_path else "",
        }

        from audiobook_forge.audio.m4b_assembler import assemble_m4b
        from audiobook_forge.audio.postprocessor import get_duration

        # Build chapter info list expected by assemble_m4b
        chapter_infos = []
        for i, ch_audio in enumerate(chapter_audio_files):
            ch_title = chapter_titles[i] if i < len(chapter_titles) else f"Chapter {i+1}"
            try:
                dur = get_duration(ch_audio)
            except Exception:
                dur = 0.0
            chapter_infos.append({
                "path": str(ch_audio),
                "title": ch_title,
                "duration": dur,
            })

        # Pass cover image via the M4B config
        m4b_config = self.config.m4b
        if cover_image_path and not m4b_config.cover_image:
            m4b_config.cover_image = str(cover_image_path)

        assemble_m4b(chapter_infos, m4b_path, metadata, m4b_config)

        self.checkpoint.mark_m4b_done()

        # --- Step 8: Shutdown TTS engine -----------------------------------
        engine.shutdown()

        # --- Step 9: WER Validation (optional) ----------------------------
        validation_report = None
        if self.config.validation.enabled:
            validation_report = self._validate_chapters(
                chapter_infos, book, title
            )

        elapsed = time.monotonic() - pipeline_start
        logger.info(
            "=== Done in %.1f s — output: %s ===",
            elapsed, m4b_path,
        )

        return m4b_path

    # ------------------------------------------------------------------
    # Chapter processing
    # ------------------------------------------------------------------

    def _process_chapter(
        self,
        chapter: Any,
        ch_idx: int,
        engine: Any,
    ) -> Path:
        """Process a single chapter from raw text to a normalised WAV.

        Returns:
            Path to the post-processed chapter WAV.
        """
        from audiobook_forge.processing.text_normalizer   import normalize_text
        from audiobook_forge.processing.sentence_segmenter import segment_sentences, chunk_sentences
        from audiobook_forge.processing.dialogue_detector  import detect_dialogue
        from audiobook_forge.audio.postprocessor           import process_chapter as _process_chapter_audio

        cfg  = self.config
        proc = cfg.processing

        # 8a. Normalize text
        text = normalize_text(
            chapter.text,
            expand_numbers=proc.normalize_numbers,
            expand_abbreviations=proc.expand_abbreviations,
        )

        # 8b. Segment into sentences
        sentences = segment_sentences(text, method=proc.segmenter)

        # 8c. Detect dialogue
        dialogue_annotations = detect_dialogue(sentences)

        # 8d. Emotion tagging (if enabled)
        if cfg.emotion.enabled:
            from audiobook_forge.processing.emotion_tagger import tag_emotions
            from dataclasses import replace as _dc_replace

            emotion_cfg = cfg.emotion

            # Engines that lack rich prosody controls gain nothing from LLM
            # refinement — force rules-only mode to avoid wasteful API calls.
            _RICH_EMOTION_ENGINES = {"fish_audio"}
            if (
                emotion_cfg.mode == "llm"
                and cfg.tts.engine not in _RICH_EMOTION_ENGINES
            ):
                logger.info(
                    "Emotion mode downgraded to 'rules' — %s engine only "
                    "supports speed-based emotion hints; LLM refinement skipped.",
                    cfg.tts.engine,
                )
                emotion_cfg = _dc_replace(emotion_cfg, mode="rules")

            annotated_sentences = tag_emotions(
                sentences, dialogue_annotations, emotion_cfg, cfg.audio
            )
        else:
            # Build plain AnnotatedSentence list without emotion metadata
            from audiobook_forge.tts.base import AnnotatedSentence
            ann_map = {a.sentence_index: a for a in dialogue_annotations}
            annotated_sentences = []
            for s in sentences:
                ann = ann_map.get(s.index)
                pause = (
                    cfg.audio.paragraph_pause_ms
                    if s.is_paragraph_end
                    else cfg.audio.sentence_pause_ms
                )
                annotated_sentences.append(
                    AnnotatedSentence(
                        text=s.text,
                        speaker=ann.speaker if ann else "narrator",
                        narration_mode=ann.narration_mode if ann else "prose",
                        pause_after_ms=pause,
                        is_paragraph_end=s.is_paragraph_end,
                    )
                )

        # 8e. Chunk sentences
        chunks = chunk_sentences(sentences, max_chars=proc.max_chunk_chars)

        # Build index → AnnotatedSentence map for quick lookup
        ann_by_idx = {i: s for i, s in enumerate(annotated_sentences)}

        # 8f. Synthesize each chunk
        chunk_audio_files: list[Path] = []
        ch_temp = self.temp_dir / f"chapter_{ch_idx:04d}"
        ch_temp.mkdir(parents=True, exist_ok=True)

        for chunk_idx, sentence_group in enumerate(chunks):
            if self.checkpoint.is_chunk_done(ch_idx, chunk_idx):
                # Retrieve cached path
                chap_data = self.checkpoint.state.chapters[ch_idx]
                for c in chap_data.get("chunks", []):
                    if c.get("chunk_index") == chunk_idx:
                        cached = Path(c.get("audio_file", ""))
                        if cached.exists():
                            chunk_audio_files.append(cached)
                        break
                continue

            chunk_out = ch_temp / f"chunk_{chunk_idx:05d}.wav"

            # Map sentence group to annotated sentences
            start_idx = sentence_group[0].index
            end_idx   = sentence_group[-1].index
            chunk_annotated = [
                ann_by_idx[i]
                for i in range(start_idx, end_idx + 1)
                if i in ann_by_idx
            ]

            if not chunk_annotated:
                logger.warning(
                    "Chapter %d chunk %d has no annotated sentences — skipping.",
                    ch_idx, chunk_idx,
                )
                continue

            logger.debug(
                "Chapter %d chunk %d/%d (%d sentences) → %s",
                ch_idx + 1, chunk_idx + 1, len(chunks),
                len(chunk_annotated), chunk_out.name,
            )

            engine.synthesize(chunk_annotated, chunk_out)
            self.checkpoint.mark_chunk_done(
                ch_idx, chunk_idx, str(chunk_out), (start_idx, end_idx)
            )
            chunk_audio_files.append(chunk_out)

        # 8g. Concatenate chunks into chapter audio
        raw_chapter_audio = ch_temp / f"chapter_{ch_idx:04d}_raw.wav"
        if len(chunk_audio_files) == 1:
            shutil.copy2(chunk_audio_files[0], raw_chapter_audio)
        elif chunk_audio_files:
            _concat_audio_files(chunk_audio_files, raw_chapter_audio)
        else:
            raise RuntimeError(f"No audio chunks produced for chapter {ch_idx}.")

        # 8h. Post-process (normalise, trim silence)
        processed_chapter_audio = self.output_dir / f"chapter_{ch_idx:04d}.wav"
        _process_chapter_audio(raw_chapter_audio, processed_chapter_audio, cfg.audio)

        self.checkpoint.mark_chapter_done(ch_idx, str(processed_chapter_audio))
        return processed_chapter_audio

    # ------------------------------------------------------------------
    # WER Validation
    # ------------------------------------------------------------------

    def _validate_chapters(
        self,
        chapter_infos: list[dict[str, Any]],
        book: Any,
        title: str,
    ) -> Any:
        """Run WER validation on all generated chapter audio files.

        Transcribes each chapter using Whisper, computes WER against the
        original source text, and writes a report.

        Returns:
            :class:`BookValidationReport` or *None* if validation deps missing.
        """
        try:
            from audiobook_forge.audio.wer_validator import (
                validate_book, format_report,
            )
        except Exception as exc:
            logger.warning("Could not load WER validator: %s", exc)
            return None

        vcfg = self.config.validation
        logger.info("Running WER validation (Whisper %s) …", vcfg.whisper_model)

        # Build source texts matching chapter_infos order
        chapter_source_texts = [ch.text for ch in book.chapters]

        try:
            report = validate_book(
                chapter_audio_files=chapter_infos,
                chapter_source_texts=chapter_source_texts,
                book_title=title,
                model_size=vcfg.whisper_model,
                device=vcfg.device,
                compute_type=vcfg.compute_type,
                language=vcfg.language,
                wer_threshold=vcfg.wer_threshold,
            )
        except RuntimeError as exc:
            logger.warning("WER validation failed: %s", exc)
            return None

        # Print report
        report_text = format_report(report)
        print(report_text)
        logger.info("\n%s", report_text)

        # Save report file
        if vcfg.report_file:
            report_path = Path(vcfg.report_file)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report_text, encoding="utf-8")
            logger.info("WER report saved to %s", report_path)

        # Save per-chapter transcripts if requested
        if vcfg.save_transcripts:
            transcript_dir = self.output_dir / "transcripts"
            transcript_dir.mkdir(parents=True, exist_ok=True)
            for ch_result in report.chapters:
                t_path = transcript_dir / f"chapter_{ch_result.chapter_index:04d}_transcript.txt"
                t_path.write_text(
                    f"# {ch_result.chapter_title}\n"
                    f"# WER: {ch_result.wer:.1%} | CER: {ch_result.cer:.1%}\n\n"
                    f"{ch_result.transcript}\n",
                    encoding="utf-8",
                )
            logger.info("Transcripts saved to %s", transcript_dir)

        return report


