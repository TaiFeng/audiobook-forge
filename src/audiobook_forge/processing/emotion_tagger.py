"""Emotion tagger — annotates sentences with emotion and intensity metadata.

Two modes:
  * ``rules`` — fast, deterministic heuristics based on dialogue annotations,
    attribution verbs, and punctuation/casing signals.
  * ``llm``   — applies rules first, then uses an OpenAI-compatible API to
    refine ambiguous sentences, with anti-melodrama guardrails.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import TYPE_CHECKING

from audiobook_forge.config import AudioConfig, EmotionConfig
from audiobook_forge.tts.base import AnnotatedSentence

if TYPE_CHECKING:
    from audiobook_forge.processing.sentence_segmenter import Sentence
    from audiobook_forge.processing.dialogue_detector import DialogueAnnotation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_EMOTIONS: frozenset[str] = frozenset(
    {"neutral", "calm", "curious", "tense", "excited", "sad", "whispered", "angry"}
)

# Attribution verb → (emotion, intensity)
_VERB_MAP: dict[str, tuple[str, float]] = {
    "whispered": ("whispered", 0.5),
    "murmured":  ("whispered", 0.5),
    "breathed":  ("whispered", 0.5),
    "shouted":   ("excited",   0.6),
    "screamed":  ("excited",   0.6),
    "yelled":    ("excited",   0.6),
    "bellowed":  ("excited",   0.6),
    "sobbed":    ("sad",       0.5),
    "wept":      ("sad",       0.5),
    "cried":     ("sad",       0.5),
    "snapped":   ("tense",     0.5),
    "growled":   ("tense",     0.5),
    "snarled":   ("tense",     0.5),
    "laughed":   ("excited",   0.4),
    "chuckled":  ("excited",   0.4),
    "giggled":   ("excited",   0.4),
    "asked":     ("curious",   0.3),
    "wondered":  ("curious",   0.3),
    "inquired":  ("curious",   0.3),
}

# Vocabulary triggers for LLM escalation (sentence is "emotionally charged")
_STRONG_EMOTION_WORDS: frozenset[str] = frozenset(
    {
        "terrified", "terror", "petrified",
        "ecstatic", "euphoric", "elated",
        "heartbroken", "devastated", "shattered",
        "furious", "enraged", "livid",
        "anguish", "despair", "hopeless",
        "triumphant", "exhilarating", "overwhelming",
        "horrified", "panic", "dread",
        "grief", "mourning", "bereavement",
        "betrayed", "humiliated", "ashamed",
    }
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_caps_words(text: str) -> int:
    """Return the number of ALL-CAPS words (length ≥ 2) in *text*."""
    return sum(1 for tok in re.findall(r"\b[A-Z]{2,}\b", text))


def _apply_rules(
    sentence: "Sentence",
    annotation: "DialogueAnnotation | None",
    config: EmotionConfig,
    audio_config: AudioConfig,
) -> AnnotatedSentence:
    """Apply deterministic emotion rules to a single sentence."""
    text = sentence.text.strip()
    emotion: str = config.default_emotion
    intensity: float = 0.3
    narration_mode: str = "prose"
    speaker: str = "narrator"

    if annotation is not None:
        narration_mode = annotation.narration_mode or "prose"
        speaker = annotation.speaker or "narrator"

        if narration_mode == "dialogue":
            verb = (annotation.attribution_verb or "").lower().strip()
            if verb and verb not in {"said", ""}:
                mapped = _VERB_MAP.get(verb)
                if mapped:
                    emotion, intensity = mapped
        elif narration_mode == "thought":
            emotion = "calm"
            intensity = 0.3

    # Punctuation-based rules
    stripped = text.rstrip()

    if stripped.endswith("!"):
        intensity = min(intensity + 0.15, config.max_intensity)

    if stripped.endswith("?") and narration_mode == "dialogue" and emotion == config.default_emotion:
        emotion = "curious"

    if stripped.endswith("..."):
        emotion = "calm"
        intensity = max(intensity - 0.1, 0.1)

    if _count_caps_words(text) >= 2:
        intensity = min(intensity + 0.1, config.max_intensity)

    # Hard cap
    intensity = min(intensity, config.max_intensity)

    # Pause
    pause_after_ms = (
        audio_config.paragraph_pause_ms
        if sentence.is_paragraph_end
        else audio_config.sentence_pause_ms
    )

    return AnnotatedSentence(
        text=text,
        speaker=speaker,
        narration_mode=narration_mode,
        emotion=emotion,
        intensity=round(intensity, 4),
        pause_after_ms=pause_after_ms,
        is_paragraph_end=sentence.is_paragraph_end,
    )


def _is_ambiguous(
    annotated: AnnotatedSentence,
    original_annotation: "DialogueAnnotation | None",
    config: EmotionConfig,
) -> bool:
    """Return True if the sentence should be sent to the LLM for refinement."""
    text = annotated.text.lower()

    # Still neutral but contains dialogue
    if annotated.emotion == "neutral" and original_annotation is not None:
        if original_annotation.contains_dialogue:
            return True

    # Long sentence
    if len(annotated.text) > config.min_llm_sentence_length:
        # Check for strong emotional vocabulary
        words = set(re.findall(r"\b\w+\b", text))
        if words & _STRONG_EMOTION_WORDS:
            return True

    return False


def _call_llm_api(
    sentences: list[AnnotatedSentence],
    indices: list[int],
    config: EmotionConfig,
) -> dict[int, tuple[str, float]]:
    """
    Send a batch of sentences to an OpenAI-compatible chat-completion API
    and return a mapping of original index → (emotion, intensity).

    Returns an empty dict on any failure.
    """
    if not sentences:
        return {}

    items_payload = [
        {"index": idx, "text": s.text}
        for idx, s in zip(indices, sentences)
    ]

    system_prompt = (
        "You are an audiobook narrator emotion tagger. "
        "For each sentence, classify the most fitting emotion for narration. "
        f"Valid emotions: {', '.join(sorted(VALID_EMOTIONS))}. "
        "Intensity must be between 0.1 and "
        f"{config.max_intensity}. "
        "Respond ONLY with a JSON array — no markdown, no prose. "
        "Format:\n"
        '[{"index": <int>, "emotion": "<emotion>", "intensity": <float>, "reasoning": "<brief>"}]'
    )

    user_content = (
        "Tag these sentences:\n"
        + json.dumps(items_payload, ensure_ascii=False)
    )

    request_body = json.dumps(
        {
            "model": config.llm.model,
            "temperature": config.llm.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
        }
    ).encode("utf-8")

    api_url = config.llm.api_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if config.llm.api_key:
        headers["Authorization"] = f"Bearer {config.llm.api_key}"

    try:
        req = urllib.request.Request(api_url, data=request_body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))

        content = response_data["choices"][0]["message"]["content"]
        # Strip markdown fences if present
        content = re.sub(r"```(?:json)?", "", content).strip().strip("`").strip()
        results = json.loads(content)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM emotion API call failed: %s", exc)
        return {}

    output: dict[int, tuple[str, float]] = {}
    for item in results:
        try:
            idx = int(item["index"])
            emo = str(item.get("emotion", "neutral")).lower()
            inten = float(item.get("intensity", 0.3))
            if emo not in VALID_EMOTIONS:
                emo = "neutral"
            inten = max(0.1, min(inten, config.max_intensity))
            output[idx] = (emo, inten)
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Bad LLM result item %r: %s", item, exc)

    return output


def _apply_guardrails(
    annotated: list[AnnotatedSentence],
    config: EmotionConfig,
) -> list[AnnotatedSentence]:
    """
    Enforce chapter-level guardrails:

    1. No more than 30 % of sentences should be non-neutral.
    2. Consecutive intensity escalation must not exceed 0.3.
    3. No intensity may exceed max_intensity.
    """
    max_intensity = config.max_intensity

    # Rule 3: hard cap
    for s in annotated:
        s.intensity = min(s.intensity, max_intensity)

    # Rule 2: consecutive escalation cap
    for i in range(1, len(annotated)):
        prev = annotated[i - 1]
        cur  = annotated[i]
        if cur.intensity - prev.intensity > 0.3:
            cur.intensity = round(prev.intensity + 0.3, 4)

    # Rule 1: anti-melodrama (30 % cap on non-neutral)
    non_neutral_indices = [
        i for i, s in enumerate(annotated) if s.emotion != "neutral"
    ]
    allowed = max(int(len(annotated) * 0.30), 1)
    if len(non_neutral_indices) > allowed:
        # Sort by intensity ascending and reset the surplus
        sorted_by_intensity = sorted(non_neutral_indices, key=lambda i: annotated[i].intensity)
        excess = len(non_neutral_indices) - allowed
        for idx in sorted_by_intensity[:excess]:
            annotated[idx].emotion = "neutral"

    return annotated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tag_emotions(
    sentences: list["Sentence"],
    dialogue_annotations: list["DialogueAnnotation"],
    config: EmotionConfig,
    audio_config: AudioConfig,
) -> list[AnnotatedSentence]:
    """
    Annotate *sentences* with emotion and intensity metadata.

    Args:
        sentences:             Segmented sentences from the text.
        dialogue_annotations:  Dialogue detection results (one per sentence).
        config:                Emotion-tagging configuration.
        audio_config:          Audio configuration (pause durations, etc.).

    Returns:
        A list of :class:`AnnotatedSentence` ready for TTS synthesis.
    """
    if not sentences:
        return []

    # Build a fast lookup: sentence_index → annotation
    annotation_map: dict[int, "DialogueAnnotation"] = {
        ann.sentence_index: ann for ann in dialogue_annotations
    }

    # -----------------------------------------------------------------------
    # Step 1 — apply rules to every sentence
    # -----------------------------------------------------------------------
    annotated: list[AnnotatedSentence] = []
    for sentence in sentences:
        ann = annotation_map.get(sentence.index)
        result = _apply_rules(sentence, ann, config, audio_config)
        annotated.append(result)

    # -----------------------------------------------------------------------
    # Step 2 — optional LLM refinement
    # -----------------------------------------------------------------------
    if config.mode == "llm":
        # Identify ambiguous sentences (work in batches)
        ambiguous_indices: list[int] = []
        for i, (sentence, ann_sentence) in enumerate(
            zip(annotated, sentences)
        ):
            original_ann = annotation_map.get(ann_sentence.index)
            if _is_ambiguous(sentence, original_ann, config):
                ambiguous_indices.append(i)

        batch_size = config.llm.batch_size or 20
        for batch_start in range(0, len(ambiguous_indices), batch_size):
            batch_idxs = ambiguous_indices[batch_start : batch_start + batch_size]
            batch_sentences = [annotated[i] for i in batch_idxs]

            logger.debug(
                "Sending %d sentences to LLM emotion API (indices %s–%s)",
                len(batch_sentences),
                batch_idxs[0],
                batch_idxs[-1],
            )

            llm_results = _call_llm_api(batch_sentences, batch_idxs, config)

            # Merge: only override if LLM provides a non-neutral label
            for orig_idx, (emo, inten) in llm_results.items():
                if emo != "neutral":
                    annotated[orig_idx].emotion = emo
                    annotated[orig_idx].intensity = round(
                        min(inten, config.max_intensity), 4
                    )

    # -----------------------------------------------------------------------
    # Step 3 — guardrails
    # -----------------------------------------------------------------------
    annotated = _apply_guardrails(annotated, config)

    return annotated
