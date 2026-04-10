"""Text normaliser for TTS pre-processing.

Converts raw prose into a form that TTS engines read aloud correctly and
naturally.  Key transformations:

* Unicode NFC normalisation.
* Smart-quote / curly-quote → straight-quote conversion.
* Em-dash / en-dash spacing normalisation.
* Ellipsis normalisation (Unicode ``…`` and literal ``...``).
* Number expansion to words (cardinals, ordinals, currency, percentages,
  years) via ``num2words`` when available, with a lightweight built-in fallback.
* Common abbreviation expansion (Mr., Dr., St., etc.).
* Zero-width character removal.
* Whitespace normalisation (collapses spaces; preserves paragraph breaks).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# num2words — optional dependency
# ---------------------------------------------------------------------------

try:
    from num2words import num2words as _num2words  # type: ignore[import]
    _NUM2WORDS_AVAILABLE = True
    logger.debug("num2words available — using for number expansion")
except ImportError:
    _NUM2WORDS_AVAILABLE = False
    logger.debug("num2words not available — using built-in fallback")

# ---------------------------------------------------------------------------
# Abbreviation map
# ---------------------------------------------------------------------------

# Ordered so that longer keys are checked first (prevents partial matches).
_ABBREVIATIONS: list[tuple[re.Pattern[str], str]] = []

_ABBREV_MAP: list[tuple[str, str]] = [
    # Titles (case-sensitive — keep original capitalisation intent)
    (r"Mr\.", "Mister"),
    (r"Mrs\.", "Missus"),
    (r"Ms\.", "Miss"),
    (r"Dr\.", "Doctor"),
    (r"Prof\.", "Professor"),
    (r"Rev\.", "Reverend"),
    (r"Gen\.", "General"),
    (r"Sgt\.", "Sergeant"),
    (r"Cpl\.", "Corporal"),
    (r"Pvt\.", "Private"),
    (r"Lt\.", "Lieutenant"),
    (r"Capt\.", "Captain"),
    (r"Cmdr\.", "Commander"),
    (r"Adm\.", "Admiral"),
    (r"Pres\.", "President"),
    (r"Gov\.", "Governor"),
    (r"Sen\.", "Senator"),
    (r"Rep\.", "Representative"),
    (r"Atty\.", "Attorney"),
    (r"Supt\.", "Superintendent"),
    (r"Insp\.", "Inspector"),
    # Common words (case-insensitive)
    (r"(?i)\bSt\.\s+(?=[A-Z])", "Saint "),  # St. followed by capitalised word
    (r"(?i)\bvs\.", "versus"),
    (r"(?i)\betc\.", "et cetera"),
    (r"(?i)\bi\.e\.", "that is"),
    (r"(?i)\be\.g\.", "for example"),
    (r"(?i)\bapprox\.", "approximately"),
    (r"(?i)\bdept\.", "department"),
    (r"(?i)\bgov\.", "government"),
    (r"(?i)\bmax\.", "maximum"),
    (r"(?i)\bmin\.", "minimum"),
    (r"(?i)\bno\.\s*(?=\d)", "number "),   # "No.5" → "number 5"
    (r"(?i)\bvol\.", "volume"),
    (r"(?i)\bch\.\s*(?=\d)", "chapter "),
    (r"(?i)\bfig\.", "figure"),
    (r"(?i)\bp\.\s*(?=\d)", "page "),
    (r"(?i)\bpp\.\s*(?=\d)", "pages "),
    (r"(?i)\bjan\.", "January"),
    (r"(?i)\bfeb\.", "February"),
    (r"(?i)\bapr\.", "April"),
    (r"(?i)\baug\.", "August"),
    (r"(?i)\bsept?\.", "September"),
    (r"(?i)\boct\.", "October"),
    (r"(?i)\bnov\.", "November"),
    (r"(?i)\bdec\.", "December"),
    (r"(?i)\bsq\.", "square"),
    (r"(?i)\bave\.", "Avenue"),
    (r"(?i)\bblvd\.", "Boulevard"),
    (r"(?i)\brd\.", "Road"),
]

# Pre-compile patterns; keep track of replacement strings.
for _pat_str, _repl in _ABBREV_MAP:
    _ABBREVIATIONS.append((re.compile(_pat_str), _repl))


# ---------------------------------------------------------------------------
# Built-in number-to-words fallback
# ---------------------------------------------------------------------------

_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety",
]
_ORDINAL_EXCEPTIONS: dict[str, str] = {
    "one": "first", "two": "second", "three": "third",
    "four": "fourth", "five": "fifth", "six": "sixth",
    "seven": "seventh", "eight": "eighth", "nine": "ninth",
    "twelve": "twelfth",
}
_ORDINAL_SUFFIX_RE = re.compile(r"(st|nd|rd|th)$", re.IGNORECASE)


def _int_to_words(n: int) -> str:
    """Convert a non-negative integer to its English word representation.

    Handles numbers up to 999,999,999,999.

    Parameters
    ----------
    n:
        Non-negative integer.

    Returns
    -------
    str
        English word form, e.g. ``42`` → ``"forty-two"``.
    """
    if n < 0:
        return "negative " + _int_to_words(-n)
    if n == 0:
        return "zero"
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("-" + _ONES[ones] if ones else "")
    if n < 1_000:
        hundreds, rest = divmod(n, 100)
        return _ONES[hundreds] + " hundred" + (" " + _int_to_words(rest) if rest else "")
    if n < 1_000_000:
        thousands, rest = divmod(n, 1_000)
        return _int_to_words(thousands) + " thousand" + (" " + _int_to_words(rest) if rest else "")
    if n < 1_000_000_000:
        millions, rest = divmod(n, 1_000_000)
        return _int_to_words(millions) + " million" + (" " + _int_to_words(rest) if rest else "")
    billions, rest = divmod(n, 1_000_000_000)
    return _int_to_words(billions) + " billion" + (" " + _int_to_words(rest) if rest else "")


def _to_ordinal(word: str) -> str:
    """Convert a cardinal word form to its ordinal form."""
    # Check known exceptions.
    for suffix, ordinal in _ORDINAL_EXCEPTIONS.items():
        if word.endswith(suffix):
            return word[: -len(suffix)] + ordinal
    # General rule: add "th" (handles "twenty-first" etc. via recursion on last component).
    if "-" in word:
        parts = word.rsplit("-", 1)
        return parts[0] + "-" + _to_ordinal(parts[1])
    if word.endswith("t"):
        return word + "h"
    if word.endswith("e"):
        return word[:-1] + "th"
    return word + "th"


def _year_to_words(year: int) -> str:
    """Convert a 4-digit year to the natural spoken form.

    Examples: 1984 → "nineteen eighty-four", 2024 → "twenty twenty-four",
    1900 → "nineteen hundred", 2000 → "two thousand".
    """
    if year == 2000:
        return "two thousand"
    if year < 1000 or year > 2099:
        return _int_to_words(year)  # fallback

    century, remainder = divmod(year, 100)
    century_words = _int_to_words(century)

    if remainder == 0:
        return century_words + " hundred"
    if remainder < 10:
        return century_words + " oh " + _int_to_words(remainder)
    return century_words + " " + _int_to_words(remainder)


def _number_to_words(value: str, to: str = "cardinal") -> str:
    """Convert a numeric string to English words using num2words if available.

    Parameters
    ----------
    value:
        Numeric string, e.g. ``"42"``, ``"3.14"``.
    to:
        Conversion type: ``"cardinal"``, ``"ordinal"``, ``"year"``.

    Returns
    -------
    str
        English word form.
    """
    if _NUM2WORDS_AVAILABLE:
        try:
            n2w_to = "year" if to == "year" else to
            return _num2words(value, to=n2w_to, lang="en")  # type: ignore[call-arg]
        except Exception:
            pass  # Fall through to built-in

    # Built-in fallback.
    try:
        int_val = int(value)
    except ValueError:
        try:
            # Handle floats by splitting on decimal.
            parts = value.split(".")
            whole = _int_to_words(abs(int(parts[0])))
            prefix = "negative " if int(parts[0]) < 0 else ""
            frac = " point " + " ".join(_ONES[int(d)] if int(d) < 20 else _int_to_words(int(d))
                                        for d in parts[1]) if len(parts) > 1 else ""
            return prefix + whole + frac
        except Exception:
            return value  # give up

    if to == "year":
        return _year_to_words(int_val)
    if to == "ordinal":
        return _to_ordinal(_int_to_words(abs(int_val)))
    return _int_to_words(int_val)


# ---------------------------------------------------------------------------
# Number expansion patterns
# ---------------------------------------------------------------------------

# Ordinals: 1st, 2nd, 3rd, 4th, 23rd, etc.
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)

# Currency: $5, $5.99, $1,234.56 — dollars and cents.
_CURRENCY_DOLLAR_RE = re.compile(r"\$([0-9,]+)(?:\.(\d{1,2}))?")

# Percentages: 15%, 3.5%
_PERCENT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*%")

# 4-digit years in context (preceded/followed by typical year markers).
_YEAR_CONTEXT_RE = re.compile(
    r"(?<!\d)"
    r"((?:1[0-9]|20)\d{2})"  # 1000–2099
    r"(?!\d)"
    r"(?=\s*(?:AD|BC|CE|BCE|years?|century|decade)?)",
    re.IGNORECASE,
)

# Plain integers and decimals (applied last, after more specific patterns).
_NUMBER_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\b")


def _expand_ordinals(text: str) -> str:
    def _replace(m: re.Match[str]) -> str:
        return _number_to_words(m.group(1), to="ordinal")
    return _ORDINAL_RE.sub(_replace, text)


def _expand_currency(text: str) -> str:
    def _replace(m: re.Match[str]) -> str:
        whole_str = m.group(1).replace(",", "")
        cents_str = m.group(2)
        dollars = _number_to_words(whole_str, to="cardinal")
        dollar_word = "dollar" if whole_str == "1" else "dollars"
        if cents_str:
            cents = _number_to_words(cents_str.lstrip("0") or "0", to="cardinal")
            cent_word = "cent" if cents_str.lstrip("0") == "1" else "cents"
            return f"{dollars} {dollar_word} and {cents} {cent_word}"
        return f"{dollars} {dollar_word}"
    return _CURRENCY_DOLLAR_RE.sub(_replace, text)


def _expand_percentages(text: str) -> str:
    def _replace(m: re.Match[str]) -> str:
        return _number_to_words(m.group(1), to="cardinal") + " percent"
    return _PERCENT_RE.sub(_replace, text)


def _expand_years(text: str) -> str:
    """Expand 4-digit years that are plausibly calendar years."""
    def _replace(m: re.Match[str]) -> str:
        return _number_to_words(m.group(1), to="year")
    return _YEAR_CONTEXT_RE.sub(_replace, text)


def _expand_plain_numbers(text: str) -> str:
    def _replace(m: re.Match[str]) -> str:
        raw = m.group(1).replace(",", "")
        return _number_to_words(raw, to="cardinal")
    return _NUMBER_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Zero-width and invisible character removal
# ---------------------------------------------------------------------------

_ZERO_WIDTH_RE = re.compile(
    r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]"  # ZWS, ZWNJ, ZWJ, WJ, BOM, SHY
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_text(
    text: str,
    expand_numbers: bool = True,
    expand_abbreviations: bool = True,
) -> str:
    """Normalise *text* for TTS synthesis.

    Transformations are applied in this order:

    1. Unicode NFC normalisation.
    2. Zero-width character removal.
    3. Smart quote → straight quote conversion.
    4. Em-dash / en-dash → spaced em-dash.
    5. Ellipsis normalisation.
    6. Abbreviation expansion (if *expand_abbreviations* is True).
    7. Number expansion (if *expand_numbers* is True).
    8. Collapse multiple spaces (while preserving paragraph breaks).

    Parameters
    ----------
    text:
        Raw input text.
    expand_numbers:
        When *True*, convert numeric tokens to their English word form.
    expand_abbreviations:
        When *True*, expand common abbreviations to full words.

    Returns
    -------
    str
        Normalised text suitable for TTS input.
    """
    # 1. NFC normalisation.
    text = unicodedata.normalize("NFC", text)

    # 2. Zero-width characters.
    text = _ZERO_WIDTH_RE.sub("", text)

    # 3. Smart / curly quotes → straight quotes.
    #    Single quotes: U+2018 ' and U+2019 '
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    #    Double quotes: U+201C " and U+201D "
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    #    Low-9 quotation marks (German-style): „ "
    text = text.replace("\u201e", '"').replace("\u201f", '"')
    #    Prime / double-prime (occasionally misused as quotes)
    text = text.replace("\u2032", "'").replace("\u2033", '"')

    # 4. Dash normalisation.
    #    Em-dash (—) and en-dash (–) → spaced em-dash.
    text = re.sub(r"\s*[\u2013\u2014]\s*", " \u2014 ", text)
    #    Double hyphen sometimes used as em-dash.
    text = re.sub(r"\s*--\s*", " \u2014 ", text)

    # 5. Ellipsis normalisation.
    #    Unicode ellipsis character (…)
    text = text.replace("\u2026", "...")
    #    Four or more dots → three.
    text = re.sub(r"\.{4,}", "...", text)
    #    Spaced dots: ". . ." → "..."
    text = re.sub(r"\.\s\.\s\.", "...", text)

    # 6. Abbreviation expansion.
    if expand_abbreviations:
        for pattern, replacement in _ABBREVIATIONS:
            text = pattern.sub(replacement, text)

    # 7. Number expansion.
    #    Work paragraph-by-paragraph to avoid cross-boundary side-effects.
    if expand_numbers:
        paragraphs = text.split("\n\n")
        expanded: list[str] = []
        for para in paragraphs:
            para = _expand_ordinals(para)
            para = _expand_currency(para)
            para = _expand_percentages(para)
            para = _expand_years(para)
            para = _expand_plain_numbers(para)
            expanded.append(para)
        text = "\n\n".join(expanded)

    # 8. Whitespace cleanup (preserve paragraph breaks).
    paragraphs = text.split("\n\n")
    cleaned: list[str] = []
    for para in paragraphs:
        # Collapse intra-paragraph whitespace (spaces, tabs, single newlines).
        para = re.sub(r"[ \t]+", " ", para)
        para = re.sub(r"\n", " ", para)
        para = para.strip()
        if para:
            cleaned.append(para)
    text = "\n\n".join(cleaned)

    return text
