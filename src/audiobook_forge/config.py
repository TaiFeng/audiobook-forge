"""Configuration loader — merges YAML config with .env and CLI overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class TTSKokoroConfig:
    voice: str = "af_heart"
    speed: float = 1.0
    lang_code: str = "a"


@dataclass
class TTSFishAudioConfig:
    api_url: str = "http://localhost:8080"
    reference_audio: str = ""
    reference_text: str = ""
    compile: bool = True


@dataclass
class TTSOpenAICompatConfig:
    api_url: str = "http://localhost:1234/v1"
    api_key: str = ""
    model: str = "tts-1"
    voice: str = "alloy"


@dataclass
class TTSConfig:
    engine: str = "kokoro"
    kokoro: TTSKokoroConfig = field(default_factory=TTSKokoroConfig)
    fish_audio: TTSFishAudioConfig = field(default_factory=TTSFishAudioConfig)
    openai_compat: TTSOpenAICompatConfig = field(default_factory=TTSOpenAICompatConfig)


@dataclass
class ProcessingConfig:
    max_chunk_chars: int = 400
    segmenter: str = "regex"
    normalize_numbers: bool = True
    expand_abbreviations: bool = True


@dataclass
class EmotionLLMConfig:
    api_url: str = "http://localhost:1234/v1"
    api_key: str = ""
    model: str = "local-model"
    temperature: float = 0.3
    batch_size: int = 20


@dataclass
class EmotionConfig:
    enabled: bool = False
    mode: str = "rules"
    default_emotion: str = "neutral"
    max_intensity: float = 0.7
    min_llm_sentence_length: int = 20
    llm: EmotionLLMConfig = field(default_factory=EmotionLLMConfig)


@dataclass
class AudioConfig:
    sample_rate: int = 24000
    sentence_pause_ms: int = 250
    paragraph_pause_ms: int = 500
    normalize_loudness: bool = True
    loudness_target_lufs: float = -19.0
    trim_silence: bool = True


@dataclass
class M4BConfig:
    bitrate: int = 64
    sample_rate: int = 22050
    channels: int = 1
    cover_image: str = ""


@dataclass
class ResumeConfig:
    enabled: bool = True
    checkpoint_file: str = "./output/.checkpoint.json"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str = "./output/forge.log"


@dataclass
class ProjectConfig:
    name: str = "My Audiobook"
    author: str = ""
    narrator: str = ""
    output_dir: str = "./output"
    temp_dir: str = "./output/.tmp"


@dataclass
class ForgeConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    input_file: str = ""
    tts: TTSConfig = field(default_factory=TTSConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    m4b: M4BConfig = field(default_factory=M4BConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge_dict(target: dict, source: dict) -> dict:
    """Deep-merge source into target, preferring source values."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _merge_dict(target[key], value)
        else:
            target[key] = value
    return target


def _dict_to_dataclass(dc_class: type, data: dict[str, Any]) -> Any:
    """Recursively convert a dict to a nested dataclass."""
    if not isinstance(data, dict):
        return data
    field_types = {f.name: f.type for f in dc_class.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key in field_types:
            ft = field_types[key]
            # Resolve string annotations
            if isinstance(ft, str):
                ft = eval(ft, globals(), {k: v for k, v in vars().items()})
            if hasattr(ft, "__dataclass_fields__") and isinstance(value, dict):
                kwargs[key] = _dict_to_dataclass(ft, value)
            else:
                kwargs[key] = value
    return dc_class(**kwargs)


def load_config(
    config_path: str | Path = "config.yaml",
    overrides: dict[str, Any] | None = None,
) -> ForgeConfig:
    """Load config from YAML, overlay .env, and apply CLI overrides."""
    config_path = Path(config_path)

    # Load .env if present
    env_path = config_path.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Load YAML
    raw: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

    # Apply CLI overrides
    if overrides:
        _merge_dict(raw, overrides)

    # Inject env vars into API keys
    tts_section = raw.get("tts", {})
    oc = tts_section.get("openai_compat", {})
    if not oc.get("api_key"):
        oc["api_key"] = os.getenv("OPENAI_API_KEY", "")
    tts_section["openai_compat"] = oc

    emo = raw.get("emotion", {})
    llm = emo.get("llm", {})
    if not llm.get("api_key"):
        llm["api_key"] = os.getenv("EMOTION_LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    emo["llm"] = llm
    raw["emotion"] = emo

    # Flatten input
    input_file = raw.get("input", {}).get("file", "") if isinstance(raw.get("input"), dict) else ""

    # Build config
    cfg = ForgeConfig(
        project=_dict_to_dataclass(ProjectConfig, raw.get("project", {})),
        input_file=input_file,
        tts=_dict_to_dataclass(TTSConfig, tts_section),
        processing=_dict_to_dataclass(ProcessingConfig, raw.get("processing", {})),
        emotion=_dict_to_dataclass(EmotionConfig, raw.get("emotion", {})),
        audio=_dict_to_dataclass(AudioConfig, raw.get("audio", {})),
        m4b=_dict_to_dataclass(M4BConfig, raw.get("m4b", {})),
        resume=_dict_to_dataclass(ResumeConfig, raw.get("resume", {})),
        logging=_dict_to_dataclass(LoggingConfig, raw.get("logging", {})),
    )

    return cfg
