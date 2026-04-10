"""EPUB reader — extracts chapters and metadata from EPUB files.

Uses ``ebooklib`` for EPUB parsing and ``beautifulsoup4`` for HTML-to-text
conversion.  Produces a :class:`BookData` instance that is consumed by the
rest of the Audiobook Forge pipeline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import warnings

import ebooklib
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import epub

# Suppress the XMLParsedAsHTMLWarning: EPUB XHTML content is most reliably
# handled by the lxml HTML parser despite technically being XML.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class Chapter:
    """A single chapter extracted from a book."""

    index: int
    """Zero-based chapter index within the book."""

    title: str
    """Chapter title, derived from the EPUB TOC or a generated fallback."""

    text: str
    """Plain-text content with paragraph breaks preserved as double newlines."""

    html: str
    """Raw HTML source for the chapter (unmodified EPUB item content)."""


@dataclass
class BookData:
    """Container for all data extracted from a book file."""

    title: str
    """Book title."""

    author: str
    """Primary author / creator."""

    language: str
    """IETF language tag (e.g. ``"en"``), or empty string if unknown."""

    chapters: list[Chapter] = field(default_factory=list)
    """Ordered list of chapters."""

    cover_image_data: Optional[bytes] = None
    """Raw bytes of the cover image, or *None* if no cover was found."""

    cover_image_ext: str = "jpg"
    """File extension for the cover image (``"jpg"``, ``"png"``, …)."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Media types treated as cover images, in preference order.
_COVER_MEDIA_TYPES: tuple[str, ...] = (
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
)

# Map MIME type → extension.
_MIME_TO_EXT: dict[str, str] = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}

# Tags whose *entire* subtree we want to discard when extracting text.
_DROP_TAGS: frozenset[str] = frozenset(
    {"script", "style", "head", "nav", "aside", "footer", "figure", "figcaption"}
)

# Tags that mark the end of a paragraph / block.
_BLOCK_TAGS: frozenset[str] = frozenset(
    {
        "p", "br", "div", "section", "article", "blockquote",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "dd", "dt", "tr",
    }
)


def _html_to_text(html: str) -> str:
    """Convert an HTML string to clean plain text.

    * Strips all HTML tags.
    * Converts HTML entities (``&amp;`` → ``&``, etc.).
    * Preserves paragraph / block-level breaks as double newlines.
    * Normalises internal whitespace.

    Parameters
    ----------
    html:
        Raw HTML string.

    Returns
    -------
    str
        Plain-text representation with paragraph breaks as ``\\n\\n``.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove unwanted subtrees entirely.
    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()

    # Insert a double-newline sentinel after each block-level element so that
    # paragraph breaks survive ``get_text()``.
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_after("\n\n")

    text = soup.get_text(separator=" ")

    # Normalise line endings → LF.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse runs of spaces / tabs within a single line.
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)

    # Collapse runs of blank lines to a single blank line (= paragraph break).
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line:
            current.append(line)
        else:
            if current:
                paragraphs.append(" ".join(current))
                current = []
    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs)


def _extract_cover(book: epub.EpubBook) -> tuple[Optional[bytes], str]:
    """Try several strategies to locate and return the cover image.

    Returns
    -------
    tuple[bytes | None, str]
        Raw image bytes (or *None*) and the file extension string.
    """
    # Strategy 1 — explicit ``cover-image`` manifest property.
    for item in book.get_items():
        if (
            hasattr(item, "properties")
            and "cover-image" in (item.properties or "")
            and item.media_type in _MIME_TO_EXT
        ):
            logger.debug("Cover found via manifest property: %s", item.get_name())
            return item.get_content(), _MIME_TO_EXT[item.media_type]

    # Strategy 2 — ``<meta name="cover">`` in the OPF metadata.
    cover_id: Optional[str] = None
    for meta in book.metadata.get(epub.NAMESPACES["OPF"], {}).get("meta", []):
        # ebooklib stores metadata as (value, attrs) tuples.
        if isinstance(meta, tuple) and len(meta) == 2:
            attrs = meta[1] if isinstance(meta[1], dict) else {}
            if attrs.get("name") == "cover":
                cover_id = attrs.get("content")
                break

    if cover_id:
        item = book.get_item_with_id(cover_id)
        if item is not None and item.media_type in _MIME_TO_EXT:
            logger.debug("Cover found via OPF meta id: %s", cover_id)
            return item.get_content(), _MIME_TO_EXT[item.media_type]

    # Strategy 3 — heuristic: first image item whose name contains "cover".
    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        name_lower = item.get_name().lower()
        if "cover" in name_lower and item.media_type in _MIME_TO_EXT:
            logger.debug("Cover found via heuristic name: %s", item.get_name())
            return item.get_content(), _MIME_TO_EXT[item.media_type]

    # Strategy 4 — first image in the book.
    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        if item.media_type in _MIME_TO_EXT:
            logger.debug("Cover fallback — first image: %s", item.get_name())
            return item.get_content(), _MIME_TO_EXT[item.media_type]

    return None, "jpg"


def _get_metadata_value(book: epub.EpubBook, namespace: str, name: str) -> str:
    """Return the first metadata value for *name* in *namespace*, or ``""``."""
    try:
        values = book.get_metadata(namespace, name)
        if values:
            # ebooklib returns list of (value, {attrs}) tuples.
            first = values[0]
            if isinstance(first, tuple):
                return str(first[0]).strip()
            return str(first).strip()
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# TOC / spine traversal
# ---------------------------------------------------------------------------


def _toc_to_hrefs(toc: list) -> list[tuple[str, str]]:
    """Flatten the (potentially nested) EPUB TOC into ``(href, title)`` pairs.

    Parameters
    ----------
    toc:
        ``book.toc`` as returned by ebooklib.

    Returns
    -------
    list[tuple[str, str]]
        Ordered list of ``(href_without_fragment, title)`` pairs.
    """
    result: list[tuple[str, str]] = []

    def _walk(items: list) -> None:
        for item in items:
            if isinstance(item, epub.Link):
                href = item.href.split("#")[0]  # strip anchor fragments
                result.append((href, item.title or ""))
            elif isinstance(item, tuple) and len(item) == 2:
                section, children = item
                if isinstance(section, epub.Link):
                    href = section.href.split("#")[0]
                    result.append((href, section.title or ""))
                elif isinstance(section, epub.Section):
                    result.append(("", section.title or ""))
                _walk(list(children))

    _walk(list(toc))
    return result


def _build_chapters_from_toc(
    book: epub.EpubBook,
    toc_hrefs: list[tuple[str, str]],
) -> list[Chapter]:
    """Build chapters by matching TOC entries to EPUB spine items.

    Parameters
    ----------
    book:
        Parsed EPUB book.
    toc_hrefs:
        Ordered ``(href, title)`` list from the TOC.

    Returns
    -------
    list[Chapter]
        Chapters in TOC order (items not referenced by TOC are appended).
    """
    # Build a lookup: bare filename → EpubHtml item.
    items_by_name: dict[str, epub.EpubHtml] = {}
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        name = Path(item.get_name()).name
        items_by_name[name] = item  # type: ignore[assignment]
        items_by_name[item.get_name()] = item  # type: ignore[assignment]

    seen_names: set[str] = set()
    chapters: list[Chapter] = []

    for href, title in toc_hrefs:
        if not href:
            continue
        bare = Path(href).name
        item = items_by_name.get(bare) or items_by_name.get(href)
        if item is None:
            continue
        key = item.get_name()
        if key in seen_names:
            continue
        seen_names.add(key)

        html_content = item.get_content().decode("utf-8", errors="replace")
        text = _html_to_text(html_content)
        if not text.strip():
            continue

        chapters.append(
            Chapter(
                index=len(chapters),
                title=title or f"Chapter {len(chapters) + 1}",
                text=text,
                html=html_content,
            )
        )

    # Append spine items not covered by the TOC.
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        key = item.get_name()
        if key in seen_names:
            continue
        html_content = item.get_content().decode("utf-8", errors="replace")
        text = _html_to_text(html_content)
        if not text.strip():
            continue
        seen_names.add(key)
        chapters.append(
            Chapter(
                index=len(chapters),
                title=f"Chapter {len(chapters) + 1}",
                text=text,
                html=html_content,
            )
        )

    return chapters


def _build_chapters_from_spine(book: epub.EpubBook) -> list[Chapter]:
    """Fallback: treat each spine document as a chapter.

    Parameters
    ----------
    book:
        Parsed EPUB book.

    Returns
    -------
    list[Chapter]
        One chapter per readable spine document.
    """
    chapters: list[Chapter] = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html_content = item.get_content().decode("utf-8", errors="replace")
        text = _html_to_text(html_content)
        if not text.strip():
            continue

        # Try to infer a title from the first heading in the document.
        soup = BeautifulSoup(html_content, "lxml")
        heading = soup.find(["h1", "h2", "h3"])
        title = heading.get_text(strip=True) if heading else f"Chapter {len(chapters) + 1}"

        chapters.append(
            Chapter(
                index=len(chapters),
                title=title,
                text=text,
                html=html_content,
            )
        )

    return chapters


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_epub(file_path: str | Path) -> BookData:
    """Parse an EPUB file and return structured :class:`BookData`.

    Extraction strategy:

    1. Read EPUB metadata (title, author, language).
    2. Attempt to build chapter list from the EPUB Table of Contents (TOC).
    3. Fall back to iterating spine documents if TOC is empty.
    4. Extract the cover image (several heuristics tried in order).

    Parameters
    ----------
    file_path:
        Filesystem path to the ``.epub`` file.

    Returns
    -------
    BookData
        Fully populated data container.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If the file cannot be parsed as an EPUB.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"EPUB file not found: {path}")

    logger.info("Reading EPUB: %s", path)

    try:
        book = epub.read_epub(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to parse EPUB '{path}': {exc}") from exc

    # --- Metadata -----------------------------------------------------------
    title = (
        _get_metadata_value(book, "DC", "title")
        or _get_metadata_value(book, "dc", "title")
        or path.stem
    )
    author = (
        _get_metadata_value(book, "DC", "creator")
        or _get_metadata_value(book, "dc", "creator")
        or "Unknown"
    )
    language = (
        _get_metadata_value(book, "DC", "language")
        or _get_metadata_value(book, "dc", "language")
        or ""
    )

    logger.debug("Metadata — title=%r author=%r language=%r", title, author, language)

    # --- Chapters -----------------------------------------------------------
    toc_hrefs = _toc_to_hrefs(list(book.toc))
    if toc_hrefs:
        logger.debug("Building chapters from TOC (%d entries)", len(toc_hrefs))
        chapters = _build_chapters_from_toc(book, toc_hrefs)
    else:
        logger.debug("No TOC found — falling back to spine documents")
        chapters = _build_chapters_from_spine(book)

    logger.info("Extracted %d chapter(s)", len(chapters))

    # --- Cover image --------------------------------------------------------
    cover_data, cover_ext = _extract_cover(book)
    if cover_data:
        logger.debug("Cover image extracted (%d bytes, .%s)", len(cover_data), cover_ext)

    return BookData(
        title=title,
        author=author,
        language=language,
        chapters=chapters,
        cover_image_data=cover_data,
        cover_image_ext=cover_ext,
    )
