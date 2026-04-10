"""Plain-text book reader with heuristic chapter detection.

Reads UTF-8 (with optional BOM) text files and attempts to split them into
chapters using a hierarchy of regex-based heuristics.  Falls back to treating
the entire file as a single chapter when no boundaries are detected.

Returned data uses the same :class:`BookData` / :class:`Chapter` types as the
EPUB reader so that the rest of the pipeline is format-agnostic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .epub_reader import BookData, Chapter  # re-export; imported for type use

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chapter-detection patterns (ordered from most specific to most generic)
# ---------------------------------------------------------------------------

# Roman numeral helper — matches I through MMMCMXCIX (1–3999).
_ROMAN_RE = (
    r"M{0,3}"
    r"(?:CM|CD|D?C{0,3})"
    r"(?:XC|XL|L?X{0,3})"
    r"(?:IX|IV|V?I{0,3})"
)

# Pattern list: each entry is ``(name, compiled_pattern)``.
# A chapter heading must appear as its own line (possibly with surrounding
# whitespace) preceded by at least one blank line (or the start of the file).
_HEADING_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # "Chapter 12", "CHAPTER 12", "Chapter 12: A Title"
    (
        "chapter_numeric",
        re.compile(
            r"^\s*chapter\s+(\d+)(?:\s*[:.\-–—]\s*(.+))?\s*$",
            re.IGNORECASE,
        ),
    ),
    # "Chapter IV", "CHAPTER IV: Sub-title"
    (
        "chapter_roman",
        re.compile(
            rf"^\s*chapter\s+({_ROMAN_RE})(?:\s*[:.\-–—]\s*(.+))?\s*$",
            re.IGNORECASE,
        ),
    ),
    # "Part Three", "Part 3", "PART THREE"
    (
        "part",
        re.compile(
            r"^\s*part\s+(\w+)\s*$",
            re.IGNORECASE,
        ),
    ),
    # Stand-alone roman numeral on its own line: "IV." or "IV"
    (
        "roman_standalone",
        re.compile(
            rf"^\s*({_ROMAN_RE})\.?\s*$",
            re.IGNORECASE,
        ),
    ),
    # All-caps short title preceded by blank lines.
    # (Matched separately in the scan loop, not via a single multiline regex.)
    (
        "allcaps_short",
        re.compile(r"^\s*[A-Z][A-Z\s\d\-:''\"]{0,48}[A-Z\d]\s*$"),
    ),
]

# Section-break patterns (treated as sub-section markers, *not* new chapters).
_SECTION_BREAK_RE = re.compile(r"^\s*(\*\s*){3,}$|^\s*-{3,}\s*$|^\s*#{3,}\s*$")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_raw(path: Path) -> str:
    """Read *path* with UTF-8 encoding, stripping an optional BOM."""
    raw = path.read_bytes()
    # Strip UTF-8 BOM (EF BB BF) if present.
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    # Attempt UTF-8; fall back to latin-1 (no decode errors).
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed for %s; retrying with latin-1", path)
        return raw.decode("latin-1")


def _is_chapter_heading(line: str, prev_blank: bool) -> tuple[bool, str]:
    """Return ``(True, title)`` if *line* looks like a chapter heading.

    *prev_blank* should be ``True`` when the preceding non-empty context was a
    blank line (or the start of the file).

    Only the ``allcaps_short`` heuristic requires *prev_blank* to be True; all
    others are unconditional (they are already specific enough).
    """
    stripped = line.strip()
    if not stripped:
        return False, ""

    for name, pattern in _HEADING_PATTERNS:
        m = pattern.fullmatch(stripped)
        if m:
            if name == "allcaps_short":
                # Require blank line before AND the line to be short enough.
                if not prev_blank or len(stripped) >= 50:
                    continue
                return True, stripped.title()  # normalise caps for readability
            if name in ("chapter_numeric", "chapter_roman"):
                title_suffix = m.group(2) or ""
                num = m.group(1)
                title = f"Chapter {num}" + (f": {title_suffix.strip()}" if title_suffix else "")
                return True, title
            if name == "part":
                return True, f"Part {m.group(1).capitalize()}"
            if name == "roman_standalone":
                return True, stripped.rstrip(".")
            return True, stripped

    return False, ""


def _split_into_chapters(text: str) -> list[tuple[str, str]]:
    """Split *text* into ``[(title, body), ...]`` using heading heuristics.

    Returns an empty list if no chapter boundaries are detected (caller should
    fall back to treating the whole text as one chapter).
    """
    lines = text.splitlines(keepends=True)
    # Track which line indices are headings.
    heading_positions: list[tuple[int, str]] = []  # (line_index, title)

    prev_blank = True  # treat start-of-file as preceded by blank

    for i, raw_line in enumerate(lines):
        line = raw_line.rstrip("\n")
        is_heading, title = _is_chapter_heading(line, prev_blank)
        if is_heading:
            heading_positions.append((i, title))
        prev_blank = not line.strip()

    if not heading_positions:
        return []

    # Build chapter bodies.
    chapters: list[tuple[str, str]] = []
    for idx, (line_idx, title) in enumerate(heading_positions):
        start = line_idx + 1  # body starts after the heading line
        end = heading_positions[idx + 1][0] if idx + 1 < len(heading_positions) else len(lines)
        body_lines = lines[start:end]
        body = "".join(body_lines).strip()
        if body:
            chapters.append((title, body))

    # If there is text before the first heading, prepend it as a prologue.
    first_heading_line = heading_positions[0][0]
    prologue = "".join(lines[:first_heading_line]).strip()
    if prologue:
        chapters.insert(0, ("Prologue", prologue))

    return chapters


def _normalise_body(text: str) -> str:
    """Normalise whitespace within a chapter body.

    * Collapses 3+ consecutive blank lines to 2 (one paragraph break).
    * Strips trailing whitespace from each line.
    * Converts Windows line endings to Unix.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse excess blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces on each line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_txt(file_path: str | Path) -> BookData:
    """Parse a plain-text file and return structured :class:`BookData`.

    Parameters
    ----------
    file_path:
        Filesystem path to the ``.txt`` file.

    Returns
    -------
    BookData
        Populated data container (``cover_image_data`` is always *None*).

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TXT file not found: {path}")

    logger.info("Reading TXT: %s", path)
    raw_text = _read_raw(path)

    # Derive a title from the filename.
    book_title = path.stem.replace("_", " ").replace("-", " ").title()

    chapter_pairs = _split_into_chapters(raw_text)

    if chapter_pairs:
        logger.info("Detected %d chapter(s) via heuristics", len(chapter_pairs))
        chapters = [
            Chapter(
                index=i,
                title=title,
                text=_normalise_body(body),
                html="",  # no HTML source for plain-text files
            )
            for i, (title, body) in enumerate(chapter_pairs)
        ]
    else:
        logger.info("No chapter boundaries detected — treating file as single chapter")
        chapters = [
            Chapter(
                index=0,
                title=book_title,
                text=_normalise_body(raw_text),
                html="",
            )
        ]

    return BookData(
        title=book_title,
        author="Unknown",
        language="",
        chapters=chapters,
        cover_image_data=None,
        cover_image_ext="jpg",
    )
