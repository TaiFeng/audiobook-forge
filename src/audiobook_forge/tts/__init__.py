"""TTS engine registry."""

from __future__ import annotations

from .base import BaseTTSEngine, AnnotatedSentence, TTSResult


def get_engine(engine_name: str, config) -> BaseTTSEngine:
    """Factory: return the configured TTS engine."""
    if engine_name == "kokoro":
        from .kokoro_engine import KokoroTTSEngine
        return KokoroTTSEngine(config.tts.kokoro)
    elif engine_name == "fish_audio":
        from .fish_audio_engine import FishAudioTTSEngine
        return FishAudioTTSEngine(config.tts.fish_audio)
    elif engine_name == "openai_compat":
        from .openai_compat_engine import OpenAICompatTTSEngine
        return OpenAICompatTTSEngine(config.tts.openai_compat)
    else:
        raise ValueError(f"Unknown TTS engine: {engine_name!r}. Options: kokoro, fish_audio, openai_compat")


__all__ = ["BaseTTSEngine", "AnnotatedSentence", "TTSResult", "get_engine"]
