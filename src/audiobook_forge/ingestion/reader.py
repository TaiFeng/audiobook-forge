"""Factory module: auto-detect book format and dispatch to the correct reader.

Usage::

    from audiobook_forge.ingestion.reader import read_book

    book = read_book("path/to/book.epub")
    # or
    book = read_book("path/to/book.txt")
"""

from __future__ import annotations

import logging
from pathlib import Path

from .epub_reader import BookData, read_epub
from .txt_reader import read_txt

__all__ = ["read_book"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported format registry
# ---------------------------------------------------------------------------

# Maps lowercase file extensions (without the leading dot) to reader callables.
_READERS: dict[str, object] = {
    "epub": read_epub,
    "txt": read_txt,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_book(file_path: str | Path) -> BookData:
    """Auto-detect the book format from the file extension and read it.

    Dispatches to :func:`~audiobook_forge.ingestion.epub_reader.read_epub`
    for ``.epub`` files and to
    :func:`~audiobook_forge.ingestion.txt_reader.read_txt` for ``.txt``
    files.

    Parameters
    ----------
    file_path:
        Path (string or :class:`~pathlib.Path`) to the book file.

    Returns
    -------
    BookData
        Structured book data ready for downstream processing.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist on disk.
    ValueError
        If the file extension is not among the supported formats.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Book file not found: {path}")

    ext = path.suffix.lstrip(".").lower()

    reader = _READERS.get(ext)
    if reader is None:
        supported = ", ".join(f".{e}" for e in _READERS)
        raise ValueError(
            f"Unsupported book format '.{ext}' (file: {path}). "
            f"Supported formats: {supported}"
        )

    logger.info("Dispatching '%s' to %s reader", path.name, ext.upper())
    return reader(path)  # type: ignore[operator]
