"""M4B audiobook assembler — concatenates chapter WAV files into a single M4B.

The assembly pipeline:

1. Write an FFMETADATA file with book-level tags and chapter markers.
2. Write a concat list file for ffmpeg's ``concat`` demuxer.
3. Encode the concatenated audio to AAC-LC in an M4A container with
   embedded chapter metadata.
4. Optionally embed cover art.
5. Set the iTunes ``stik`` media kind to *Audiobook* (``2``) using mutagen
   if available, or log a notice that the flag was skipped.
6. Rename the ``.m4a`` file to ``.m4b``.

The public API consists of two functions:

* :func:`assemble_m4b` — full pipeline
* :func:`embed_cover`   — add/replace cover art in an existing M4B
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from audiobook_forge.config import M4BConfig

logger = logging.getLogger(__name__)

# Type alias for chapter info dicts supplied by the caller
ChapterInfo = dict  # {"path": str, "title": str, "duration": float}
BookMetadata = dict  # {"title": str, "author": str, "narrator": str, "year": str}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. "
            "Install it via your system package manager "
            "(e.g. 'apt install ffmpeg' or 'brew install ffmpeg')."
        )
    return path


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Execute *cmd*, raise :exc:`RuntimeError` on failure."""
    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        raise RuntimeError(
            f"Command failed (exit {exc.returncode}): {' '.join(cmd)}\n{stderr}"
        ) from exc


def _escape_ffmetadata_value(value: str) -> str:
    """Escape special characters in FFMETADATA values per the ffmpeg spec.

    The characters ``=``, ``#``, ``;``, ``\\``, and newlines must be
    backslash-escaped in FFMETADATA files.

    Args:
        value: Raw metadata string.

    Returns:
        Escaped string safe for embedding in an FFMETADATA file.
    """
    value = value.replace("\\", "\\\\")
    for ch in ("=", "#", ";", "\n"):
        value = value.replace(ch, f"\\{ch}")
    return value


def _write_ffmetadata(
    path: Path,
    metadata: BookMetadata,
    chapters: list[ChapterInfo],
) -> None:
    """Write an FFMETADATA1 file with chapter markers.

    Chapter timing is derived from the ``duration`` field of each entry in
    *chapters*.  Durations are accumulated to produce absolute
    ``START``/``END`` timestamps in milliseconds (``TIMEBASE=1/1000``).

    Args:
        path:      Destination path for the FFMETADATA file.
        metadata:  Book-level metadata dict.
        chapters:  Ordered list of chapter info dicts.
    """
    lines: list[str] = [
        ";FFMETADATA1",
        f"title={_escape_ffmetadata_value(metadata.get('title', ''))}",
        f"artist={_escape_ffmetadata_value(metadata.get('author', ''))}",
        f"album_artist={_escape_ffmetadata_value(metadata.get('author', ''))}",
        f"album={_escape_ffmetadata_value(metadata.get('title', ''))}",
        f"composer={_escape_ffmetadata_value(metadata.get('narrator', ''))}",
        f"date={_escape_ffmetadata_value(metadata.get('year', ''))}",
        "genre=Audiobook",
        "comment=Created by Audiobook Forge",
        "",
    ]

    start_ms = 0
    for chapter in chapters:
        duration_s = float(chapter.get("duration", 0.0))
        end_ms = start_ms + int(round(duration_s * 1000))
        title = _escape_ffmetadata_value(chapter.get("title", ""))
        lines.extend([
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start_ms}",
            f"END={end_ms}",
            f"title={title}",
            "",
        ])
        start_ms = end_ms

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.debug("Wrote FFMETADATA to %s", path)


def _write_concat_list(path: Path, chapters: list[ChapterInfo]) -> None:
    """Write an ffmpeg concat demuxer file listing all chapter WAV paths.

    Paths are written with ``file`` directives; single-quotes are escaped to
    allow special characters in filenames.

    Args:
        path:     Destination path for the concat list file.
        chapters: Ordered list of chapter info dicts containing ``path`` keys.
    """
    lines: list[str] = []
    for chapter in chapters:
        # Escape single-quotes for the concat file format
        chapter_path = str(Path(chapter["path"]).resolve())
        escaped = chapter_path.replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.debug("Wrote concat list to %s with %d entries", path, len(lines))


def _set_audiobook_stik(m4b_path: Path) -> None:
    """Set the iTunes ``stik`` atom to *Audiobook* (``2``) using mutagen.

    If mutagen is not installed the function logs a notice and returns without
    error — the M4B will still be valid, just without the explicit media-kind
    flag.

    Args:
        m4b_path: Path to the ``.m4b`` file to modify in-place.
    """
    try:
        from mutagen.mp4 import MP4  # type: ignore[import]
    except ImportError:
        logger.info(
            "mutagen not installed; skipping iTunes stik=2 (Audiobook) tag. "
            "Install mutagen with: pip install mutagen"
        )
        return

    try:
        audio = MP4(str(m4b_path))
        audio["stik"] = [2]  # 2 = Audiobook
        audio.save()
        logger.info("Set iTunes stik=2 (Audiobook) on %s", m4b_path)
    except Exception:
        logger.warning(
            "Failed to set iTunes stik tag on %s; skipping.",
            m4b_path,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assemble_m4b(
    chapter_files: list[ChapterInfo],
    output_path: Path | str,
    metadata: BookMetadata,
    config: M4BConfig,
) -> Path:
    """Assemble chapter WAV files into a single M4B audiobook.

    The function performs the following steps:

    1. Validate inputs.
    2. Write an FFMETADATA file with chapter markers.
    3. Write an ffmpeg concat list file.
    4. Encode concatenated audio to AAC-LC (M4A container).
    5. Embed chapter metadata.
    6. Optionally embed cover art from :attr:`~audiobook_forge.config.M4BConfig.cover_image`.
    7. Set iTunes ``stik=2`` (Audiobook) via mutagen.
    8. Rename ``.m4a`` → ``.m4b``.

    Args:
        chapter_files: Ordered list of chapter descriptors.  Each dict must
                       contain at least:

                       * ``"path"``     — absolute or relative WAV file path
                       * ``"title"``    — chapter display name
                       * ``"duration"`` — duration in seconds (float)

        output_path:   Desired path for the finished ``.m4b`` file.
        metadata:      Book-level metadata dict with keys ``title``, ``author``,
                       ``narrator``, and ``year`` (all strings).
        config:        :class:`~audiobook_forge.config.M4BConfig` controlling
                       bitrate, sample rate, channels, and cover image path.

    Returns:
        The final ``.m4b`` file path.

    Raises:
        ValueError:   If *chapter_files* is empty or a chapter path is missing.
        RuntimeError: If ``ffmpeg`` is not installed or encoding fails.
    """
    ffmpeg = _require_ffmpeg()

    # --- Validate inputs ---
    if not chapter_files:
        raise ValueError("chapter_files must not be empty.")

    for i, ch in enumerate(chapter_files):
        if not ch.get("path"):
            raise ValueError(f"Chapter {i} is missing a 'path' field.")
        if not Path(ch["path"]).is_file():
            raise ValueError(
                f"Chapter {i} path does not exist: {ch['path']!r}"
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Work in a temporary directory to avoid polluting the output directory
    with tempfile.TemporaryDirectory(prefix="audiobook_forge_m4b_") as tmpdir:
        tmp = Path(tmpdir)

        # --- Write helper files ---
        ffmeta_path  = tmp / "metadata.ffmeta"
        concat_path  = tmp / "filelist.txt"
        m4a_path     = tmp / "output.m4a"

        _write_ffmetadata(ffmeta_path, metadata, chapter_files)
        _write_concat_list(concat_path, chapter_files)

        # --- Encode ---
        bitrate = f"{config.bitrate}k"
        logger.info(
            "Encoding %d chapters → %s (aac, %s, %d Hz, %d ch) …",
            len(chapter_files),
            m4a_path.name,
            bitrate,
            config.sample_rate,
            config.channels,
        )

        encode_cmd = [
            ffmpeg, "-nostdin",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_path),
            "-i", str(ffmeta_path),
            "-map_metadata", "1",
            "-c:a", "aac",
            "-b:a", bitrate,
            "-ar", str(config.sample_rate),
            "-ac", str(config.channels),
            "-movflags", "+faststart",
            "-y",
            str(m4a_path),
        ]
        _run(encode_cmd)
        logger.info("AAC encoding complete.")

        # --- Embed cover art (optional) ---
        cover_path_str = config.cover_image or ""
        if cover_path_str:
            cover_path = Path(cover_path_str)
            if cover_path.is_file():
                logger.info("Embedding cover art from %s …", cover_path)
                covered_path = tmp / "output_cover.m4a"
                cover_cmd = [
                    ffmpeg, "-nostdin",
                    "-i", str(m4a_path),
                    "-i", str(cover_path),
                    "-map", "0:a",
                    "-map", "1",
                    "-c", "copy",
                    "-disposition:v:0", "attached_pic",
                    "-y",
                    str(covered_path),
                ]
                _run(cover_cmd)
                m4a_path = covered_path
                logger.info("Cover art embedded.")
            else:
                logger.warning(
                    "cover_image %r not found; skipping cover art.", cover_path_str
                )

        # --- Move to final location (still as .m4a) ---
        m4b_tmp = output_path.with_suffix(".m4a")
        shutil.copy2(str(m4a_path), str(m4b_tmp))

    # --- Set iTunes stik tag (mutagen) ---
    _set_audiobook_stik(m4b_tmp)

    # --- Rename to .m4b ---
    final_path = output_path.with_suffix(".m4b")
    m4b_tmp.rename(final_path)
    logger.info("M4B assembly complete → %s", final_path)
    return final_path


def embed_cover(m4b_path: Path | str, cover_image_path: Path | str) -> Path:
    """Embed or replace cover art in an existing M4B file using ffmpeg.

    The original M4B is replaced in-place.

    Args:
        m4b_path:          Path to the existing ``.m4b`` file.
        cover_image_path:  Path to the cover image (JPEG or PNG).

    Returns:
        The M4B path (same as *m4b_path*).

    Raises:
        FileNotFoundError: If either *m4b_path* or *cover_image_path* does
                           not exist.
        RuntimeError:      If ``ffmpeg`` is not installed or the command fails.
    """
    ffmpeg = _require_ffmpeg()
    m4b_path = Path(m4b_path)
    cover_image_path = Path(cover_image_path)

    if not m4b_path.is_file():
        raise FileNotFoundError(f"M4B file not found: {m4b_path}")
    if not cover_image_path.is_file():
        raise FileNotFoundError(f"Cover image not found: {cover_image_path}")

    tmp_path = m4b_path.with_suffix(".tmp.m4b")
    cmd = [
        ffmpeg, "-nostdin",
        "-i", str(m4b_path),
        "-i", str(cover_image_path),
        "-map", "0",
        "-map", "1",
        "-c", "copy",
        "-disposition:v:0", "attached_pic",
        "-y",
        str(tmp_path),
    ]
    logger.info("Embedding cover art into %s …", m4b_path)
    _run(cmd)
    tmp_path.replace(m4b_path)
    logger.info("Cover art embedded successfully → %s", m4b_path)
    return m4b_path
