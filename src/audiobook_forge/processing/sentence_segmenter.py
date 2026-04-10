"""Sentence segmenter with paragraph-boundary tracking.

Provides two segmentation modes:

``"regex"``
    A hand-crafted regex-based splitter that handles common edge cases
    (abbreviations, decimal numbers, quoted speech, ellipses) without any
    external dependencies.

``"nltk"``
    Delegates to :func:`nltk.tokenize.sent_tokenize` when the ``nltk``
    package and its ``punkt`` model are available; gracefully falls back to
    the regex mode if not.

Both modes track paragraph boundaries so that downstream code can insert
appropriate pauses between paragraphs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK availability check
# ---------------------------------------------------------------------------

try:
    import nltk  # type: ignore[import]
    # Attempt to use the tokenizer; download punkt_tab if needed.
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
    _NLTK_AVAILABLE = True
    logger.debug("nltk available for sentence tokenization")
except ImportError:
    _NLTK_AVAILABLE = False
    logger.debug("nltk not available — regex segmenter will be used")


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class Sentence:
    """A single sentence extracted from the text."""

    text: str
    """The sentence string (stripped of leading/trailing whitespace)."""

    index: int
    """Zero-based position of the sentence in the full sequence."""

    is_paragraph_end: bool = False
    """*True* when this sentence is the last sentence in its paragraph."""


# ---------------------------------------------------------------------------
# Regex-based segmenter internals
# ---------------------------------------------------------------------------

# Abbreviations that should never trigger a sentence split.
# Keep sorted and lowercase for readability; matching is case-insensitive.
_ABBREV_PREFIXES: frozenset[str] = frozenset(
    {
        # Titles
        "mr", "mrs", "ms", "dr", "prof", "rev", "gen", "sgt", "cpl",
        "pvt", "lt", "capt", "cmdr", "adm", "pres", "gov", "sen", "rep",
        "atty", "supt", "insp", "sr", "jr",
        # Academic / Latin
        "vs", "etc", "fig", "vol", "no", "op", "pp", "est", "approx",
        "dept", "co", "corp", "inc", "ltd", "llc",
        # Months / days
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
        # Directions / geography
        "st", "ave", "blvd", "rd", "ln", "ct", "pl", "sq",
        # Misc
        "no", "vol", "ch", "p", "pg", "fig", "ref", "ed", "eds",
    }
)

# A sentence terminal is one of . ! ? possibly followed by closing brackets
# or quotes, then whitespace.
# We split ONLY when we can confirm it is a real sentence boundary.
#
# Strategy: tokenise by finding candidate split points, then filter them.

# Finds a candidate sentence-end: punctuation + possible closing chars + whitespace.
_SPLIT_RE = re.compile(
    r"""
    (?<!\s)               # not preceded by whitespace (avoids matching mid-word)
    (?:
        (?<=\.)           # after a period
      | (?<=\!)           # after exclamation mark
      | (?<=\?)           # after question mark
    )
    (?:                   # optional trailing close-quote / bracket
        ['"\u201d\u2019)\]]*
    )
    \s+                   # whitespace (the actual split point)
    (?=[A-Z\u00C0-\u00DC\u0400-\u04FF"'\u201c\u2018\(\[])
                          # next token starts with capital or open-quote/bracket
    """,
    re.VERBOSE,
)

# Pattern to detect abbreviations: "Word." where "Word" is a known abbreviation.
_ABBREV_DOT_RE = re.compile(r"\b([A-Za-z]+)\.\s*$")

# Detects decimal numbers like "3.14" so we don't split at the period.
_DECIMAL_RE = re.compile(r"\d+\.\d")

# Detects ellipsis patterns.
_ELLIPSIS_RE = re.compile(r"\.{2,}$")

# Detects single-letter initials: "J." or "J.K."
_INITIAL_RE = re.compile(r"^[A-Z]\.$")


def _is_abbreviation_boundary(left: str) -> bool:
    """Return *True* if the right-hand context of a candidate split looks like
    it follows an abbreviation (i.e. is *not* a real sentence boundary).

    Parameters
    ----------
    left:
        The text to the left of the whitespace split point.
    """
    stripped = left.rstrip()

    # Ellipsis — not a sentence end (treat as a pause, not a boundary).
    if _ELLIPSIS_RE.search(stripped):
        return True

    # Decimal number: "price is 3." — unlikely, but safe to keep.
    if _DECIMAL_RE.search(stripped):
        return True

    # Check for a known abbreviation.
    m = _ABBREV_DOT_RE.search(stripped)
    if m:
        word = m.group(1).lower()
        if word in _ABBREV_PREFIXES:
            return True
        # Single capital letter = initial.
        if _INITIAL_RE.match(m.group(1) + "."):
            return True

    return False


def _split_sentences_regex(paragraph: str) -> list[str]:
    """Split a single paragraph into sentences using the regex approach.

    Parameters
    ----------
    paragraph:
        A single paragraph (no ``\\n\\n``).

    Returns
    -------
    list[str]
        List of sentence strings.
    """
    # Find all candidate split positions (indices into the string).
    candidates: list[int] = []
    for m in _SPLIT_RE.finditer(paragraph):
        split_pos = m.start()  # end of punctuation group
        left_text = paragraph[:split_pos + 1]  # +1 to include the punctuation char
        if not _is_abbreviation_boundary(left_text):
            # The split position is the start of the whitespace run.
            candidates.append(m.start())

    if not candidates:
        return [paragraph.strip()] if paragraph.strip() else []

    sentences: list[str] = []
    prev = 0
    for pos in candidates:
        # pos is right after the punctuation; include punctuation in the left sentence.
        # The actual whitespace is between pos and the next real char.
        m = _SPLIT_RE.search(paragraph, pos)
        if m is None:
            break
        end_of_sentence = m.end()  # start of next sentence
        sentence = paragraph[prev : m.start() + 1].strip()  # include punctuation
        # Handle trailing close-quotes that belong to the sentence.
        close_quote_m = re.match(r"['\")\]]*", paragraph[m.start() + 1 :])
        if close_quote_m:
            sentence = (paragraph[prev : m.start() + 1 + close_quote_m.end()]).strip()
        if sentence:
            sentences.append(sentence)
        prev = end_of_sentence - (end_of_sentence - m.end()) + m.end() - m.end()
        prev = m.end()

    # Append the final fragment.
    tail = paragraph[prev:].strip()
    if tail:
        sentences.append(tail)

    return sentences


def _split_sentences_regex_v2(paragraph: str) -> list[str]:
    """Cleaner implementation using a split-and-rejoin strategy.

    Splits the paragraph on sentence-terminal punctuation and then reassembles
    sentences that were incorrectly broken (abbreviations, decimals).
    """
    # Tokenise into word-like tokens preserving whitespace positions.
    # We work with spans so we can accurately reconstruct sentences.

    # Insert a special marker after every real sentence-ending punctuation.
    MARKER = "\x00"

    # First protect abbreviations and decimals by replacing their periods
    # with a placeholder.
    _PERIOD_PLACEHOLDER = "\x01"

    protected = paragraph

    # Protect decimal numbers: 3.14 → 3\x011\x014
    def _protect_decimals(m: re.Match[str]) -> str:
        return m.group(0).replace(".", _PERIOD_PLACEHOLDER)

    protected = re.sub(r"\d+\.\d+", _protect_decimals, protected)

    # Protect known abbreviations: replace their period with a placeholder.
    # Group 1 = the word, group 2 = the whitespace AFTER the period (not the period).
    def _protect_abbrev(m: re.Match[str]) -> str:
        word = m.group(1)
        space_after = m.group(2)  # whitespace character(s) after the period
        if word.lower() in _ABBREV_PREFIXES or _INITIAL_RE.match(word + "."):
            # word + placeholder (replaces '.') + original whitespace
            return word + _PERIOD_PLACEHOLDER + space_after
        return m.group(0)

    # Capture: word + literal period + following whitespace (group 2 = whitespace only)
    protected = re.sub(r"\b([A-Za-z]+)\.( +)", _protect_abbrev, protected)

    # Protect ellipses.
    protected = protected.replace("...", "\x02")

    # Now insert MARKER after sentence-terminal punctuation followed by space+capital.
    protected = re.sub(
        r"([.!?]['\"\u201d\u2019)\]]*)\s+(?=[A-Z\u00C0-\u00DC\u0400-\u04FF\"'\u201c\u2018\(\[])",
        r"\1" + MARKER,
        protected,
    )

    # Split on markers.
    parts = protected.split(MARKER)

    # Restore placeholders.
    sentences: list[str] = []
    for part in parts:
        restored = (
            part
            .replace(_PERIOD_PLACEHOLDER, ".")
            .replace("\x02", "...")
            .strip()
        )
        if restored:
            sentences.append(restored)

    return sentences


# Use the cleaner v2 implementation as the primary regex splitter.
_split_sentences_regex = _split_sentences_regex_v2  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# NLTK segmenter
# ---------------------------------------------------------------------------


def _split_sentences_nltk(paragraph: str) -> list[str]:
    """Split using NLTK's Punkt tokenizer.

    Falls back to regex mode if NLTK is unavailable.

    Parameters
    ----------
    paragraph:
        A single paragraph string.

    Returns
    -------
    list[str]
        Sentence strings.
    """
    if not _NLTK_AVAILABLE:
        logger.debug("nltk unavailable — falling back to regex segmenter")
        return _split_sentences_regex(paragraph)
    try:
        from nltk.tokenize import sent_tokenize  # type: ignore[import]
        return [s.strip() for s in sent_tokenize(paragraph) if s.strip()]
    except Exception as exc:
        logger.warning("NLTK sent_tokenize failed (%s) — falling back to regex", exc)
        return _split_sentences_regex(paragraph)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segment_sentences(text: str, method: str = "regex") -> list[Sentence]:
    """Split *text* into individual :class:`Sentence` objects.

    Paragraph boundaries (double newlines) are preserved as
    ``is_paragraph_end=True`` on the last sentence of each paragraph.

    Parameters
    ----------
    text:
        Input text.  Paragraphs are separated by ``\\n\\n``.
    method:
        Segmentation backend.  One of ``"regex"`` or ``"nltk"``.
        Unknown values fall back to ``"regex"`` with a warning.

    Returns
    -------
    list[Sentence]
        Ordered list of sentences with metadata.
    """
    if method not in ("regex", "nltk"):
        logger.warning("Unknown segmentation method %r — using 'regex'", method)
        method = "regex"

    splitter = _split_sentences_nltk if method == "nltk" else _split_sentences_regex

    paragraphs = text.split("\n\n")
    all_sentences: list[Sentence] = []
    global_index = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_sentences = splitter(para)
        para_sentences = [s for s in para_sentences if s.strip()]

        for i, sent_text in enumerate(para_sentences):
            is_last = i == len(para_sentences) - 1
            all_sentences.append(
                Sentence(
                    text=sent_text.strip(),
                    index=global_index,
                    is_paragraph_end=is_last,
                )
            )
            global_index += 1

    return all_sentences


def chunk_sentences(
    sentences: list[Sentence],
    max_chars: int = 400,
) -> list[list[Sentence]]:
    """Group *sentences* into chunks that do not exceed *max_chars*.

    Sentences are never split across chunks.  If a single sentence is longer
    than *max_chars* it occupies its own chunk (possibly exceeding the limit).

    Chunks respect paragraph boundaries: when ``sentence.is_paragraph_end`` is
    *True* and adding the next sentence would not exceed the limit, the
    paragraph end is still honoured as a natural chunk boundary when the
    current chunk is already half-full (heuristic to avoid very short chunks).

    Parameters
    ----------
    sentences:
        Ordered list of :class:`Sentence` objects.
    max_chars:
        Soft character limit per chunk.

    Returns
    -------
    list[list[Sentence]]
        Ordered list of chunks; each chunk is a list of :class:`Sentence`.
    """
    if not sentences:
        return []

    chunks: list[list[Sentence]] = []
    current: list[Sentence] = []
    current_chars = 0

    for sentence in sentences:
        sent_len = len(sentence.text)

        if not current:
            current.append(sentence)
            current_chars = sent_len
        elif current_chars + 1 + sent_len > max_chars:
            # Current chunk is full — flush and start a new one.
            chunks.append(current)
            current = [sentence]
            current_chars = sent_len
        else:
            current.append(sentence)
            current_chars += 1 + sent_len  # +1 for the space between sentences

            # Honour paragraph boundaries as natural chunk breaks when the
            # chunk has used at least half its budget.
            if sentence.is_paragraph_end and current_chars >= max_chars // 2:
                chunks.append(current)
                current = []
                current_chars = 0

    if current:
        chunks.append(current)

    return chunks
