"""Dialogue detector — classifies sentences as prose, dialogue, thought, or mixed.

Works on text that has already passed through
:mod:`~audiobook_forge.processing.text_normalizer` (i.e. smart quotes have
been converted to straight ASCII quotes).

Approach
--------
* A regex-based pass identifies double-quoted spans (dialogue), single-quoted
  spans that look like internal thought (used with caution to avoid false
  positives from possessives), and attribution verbs.
* Each :class:`~audiobook_forge.processing.sentence_segmenter.Sentence` is
  classified as ``"prose"``, ``"dialogue"``, ``"thought"``, or ``"mixed"``.
* Speaker is set to ``"character"`` for sentences that contain dialogue and
  ``"narrator"`` otherwise.

The module is intentionally lightweight: speaker *identity* (character names)
is left for a future LLM-based pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .sentence_segmenter import Sentence

__all__ = ["DialogueAnnotation", "detect_dialogue"]

# ---------------------------------------------------------------------------
# Data-class
# ---------------------------------------------------------------------------


@dataclass
class DialogueAnnotation:
    """Dialogue annotation for a single sentence."""

    sentence_index: int
    """Matches :attr:`~audiobook_forge.processing.sentence_segmenter.Sentence.index`."""

    narration_mode: str
    """One of ``"prose"``, ``"dialogue"``, ``"thought"``, or ``"mixed"``."""

    speaker: str
    """``"narrator"`` for prose/thought; ``"character"`` for dialogue."""

    contains_dialogue: bool
    """*True* when the sentence has at least one quoted speech span."""

    attribution_verb: str = ""
    """The attribution verb found (e.g. ``"said"``), or empty string."""


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# ---- Dialogue (double-quoted speech) ----------------------------------------

# Matches a straight-double-quoted span.  We use a non-greedy match and allow
# the content to contain escaped quotes only via the normal no-newline rule.
_DOUBLE_QUOTE_RE = re.compile(
    r'"[^"\n]{1,500}"'       # "..." — straight double quotes
    r'|'
    r'\u201c[^\u201d\n]{1,500}\u201d',  # "..." — curly double quotes (belt-and-braces)
)

# ---- Thought (single-quoted or italics-style) --------------------------------

# Single-quoted spans are trickier because of possessives/contractions.
# We only flag a single-quoted span as "thought" when it is:
#  * at least 4 characters long,
#  * starts/ends with a word character (not a punctuation apostrophe),
#  * surrounded by non-word-characters or at sentence start/end.
_SINGLE_QUOTE_RE = re.compile(
    r"(?<!\w)'(?!\s)([^'\n]{3,200})(?<!\s)'"
    r"(?!\w)"
)

# Italics markers that survive normalisation (rare, but some plain-text sources
# use _underscores_ for emphasis).
_ITALICS_RE = re.compile(r"_[^_\n]{1,200}_")

# ---- Attribution verbs -------------------------------------------------------

# Comprehensive list of speech-attribution verbs.
_ATTRIBUTION_VERBS: tuple[str, ...] = (
    "said", "says", "say",
    "asked", "asks", "ask",
    "replied", "replies", "reply",
    "answered", "answers", "answer",
    "called", "calls", "call",
    "cried", "cries", "cry",
    "shouted", "shouts", "shout",
    "yelled", "yells", "yell",
    "whispered", "whispers", "whisper",
    "murmured", "murmurs", "murmur",
    "muttered", "mutters", "mutter",
    "growled", "growls", "growl",
    "snapped", "snaps", "snap",
    "hissed", "hisses", "hiss",
    "breathed", "breathes", "breathe",
    "laughed", "laughs", "laugh",
    "sobbed", "sobs", "sob",
    "sighed", "sighs", "sigh",
    "groaned", "groans", "groan",
    "screamed", "screams", "scream",
    "declared", "declares", "declare",
    "announced", "announces", "announce",
    "exclaimed", "exclaims", "exclaim",
    "demanded", "demands", "demand",
    "ordered", "orders", "order",
    "pleaded", "pleads", "plead",
    "begged", "begs", "beg",
    "suggested", "suggests", "suggest",
    "admitted", "admits", "admit",
    "agreed", "agrees", "agree",
    "argued", "argues", "argue",
    "insisted", "insists", "insist",
    "interrupted", "interrupts", "interrupt",
    "continued", "continues", "continue",
    "added", "adds", "add",
    "noted", "notes", "note",
    "observed", "observes", "observe",
    "thought", "thinks", "think",
    "wondered", "wonders", "wonder",
    "realized", "realises", "realize",
    "remembered", "remembers", "remember",
    "told", "tells", "tell",
    "warned", "warns", "warn",
    "promised", "promises", "promise",
    "lied", "lies", "lie",
    "teased", "teases", "tease",
    "confessed", "confesses", "confess",
    "repeated", "repeats", "repeat",
    "echoed", "echoes", "echo",
    "grumbled", "grumbles", "grumble",
    "whimpered", "whimpers", "whimper",
    "stuttered", "stutters", "stutter",
)

# Build a single compiled pattern for attribution verbs.
_ATTRIBUTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(v) for v in _ATTRIBUTION_VERBS) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def _find_attribution_verb(text: str) -> str:
    """Return the first attribution verb found in *text*, or ``""``."""
    m = _ATTRIBUTION_RE.search(text)
    return m.group(1).lower() if m else ""


def _classify_sentence(text: str) -> tuple[str, bool, str]:
    """Classify a single sentence.

    Parameters
    ----------
    text:
        Sentence text (post-normalisation).

    Returns
    -------
    tuple[str, bool, str]
        ``(narration_mode, contains_dialogue, attribution_verb)``
    """
    dialogue_spans = _DOUBLE_QUOTE_RE.findall(text)
    thought_spans = _SINGLE_QUOTE_RE.findall(text) + _ITALICS_RE.findall(text)

    has_dialogue = bool(dialogue_spans)
    has_thought = bool(thought_spans)

    # Determine how much of the sentence is covered by dialogue/thought.
    # Build a boolean mask to detect "mixed" vs pure modes.
    total_chars = len(text)
    dialogue_chars = sum(len(s) for s in dialogue_spans)
    thought_chars = sum(len(s) for s in thought_spans)
    non_quoted_chars = total_chars - dialogue_chars - thought_chars

    attribution_verb = _find_attribution_verb(text)

    if has_dialogue and has_thought:
        mode = "mixed"
    elif has_dialogue:
        # If there is substantial non-dialogue text, classify as mixed.
        if non_quoted_chars > max(20, total_chars * 0.4):
            mode = "mixed"
        else:
            mode = "dialogue"
    elif has_thought:
        mode = "thought"
    else:
        mode = "prose"

    return mode, has_dialogue, attribution_verb


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_dialogue(sentences: list[Sentence]) -> list[DialogueAnnotation]:
    """Annotate each sentence with dialogue / narration metadata.

    Parameters
    ----------
    sentences:
        Output of
        :func:`~audiobook_forge.processing.sentence_segmenter.segment_sentences`.

    Returns
    -------
    list[DialogueAnnotation]
        One annotation per input sentence, in the same order.
    """
    annotations: list[DialogueAnnotation] = []

    for sentence in sentences:
        mode, contains_dialogue, attribution_verb = _classify_sentence(sentence.text)

        # Speaker heuristic: prose and thought are narrated; dialogue is spoken
        # by a character.
        if mode in ("dialogue", "mixed") and contains_dialogue:
            speaker = "character"
        else:
            speaker = "narrator"

        annotations.append(
            DialogueAnnotation(
                sentence_index=sentence.index,
                narration_mode=mode,
                speaker=speaker,
                contains_dialogue=contains_dialogue,
                attribution_verb=attribution_verb,
            )
        )

    return annotations
