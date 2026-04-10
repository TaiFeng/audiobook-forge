# Audiobook Forge

Local-first AI audiobook generation pipeline. Converts DRM-free EPUB or TXT
novels into chapterized M4B audiobooks using GPU-accelerated TTS on your own
hardware.

---

## Features

- **EPUB and TXT input** with automatic chapter detection
- **Three swappable TTS engines**: Kokoro TTS (default), Fish Audio S2, OpenAI-compatible API
- **Chapter markers and metadata** in the final M4B
- **Resumable processing** — interrupted runs continue from the last completed chunk
- **Conservative emotion system** — dialogue detection, attribution-based mood tagging, optional LLM refinement
- **Audio post-processing** — loudness normalization (LUFS), silence trimming, resampling
- **Config-driven** — swap models, voices, and providers via YAML

---

## Quick Start

### 1. System prerequisites

```bash
# Ubuntu / WSL2
sudo apt update && sudo apt install -y ffmpeg python3.12 python3.12-venv

# Verify
ffmpeg -version
python3.12 --version
```

### 2. Create environment and install

```bash
git clone <repo-url> audiobook-forge
cd audiobook-forge

python3.12 -m venv .venv
source .venv/bin/activate       # Linux / WSL
# .venv\Scripts\activate.bat    # Windows (native)

# Core dependencies
pip install -r requirements.txt

# Install Kokoro TTS (recommended default engine)
pip install kokoro>=0.9 soundfile

# Optional: OpenAI client (for openai_compat engine or LLM emotion tagging)
pip install openai>=1.0

# Optional: NLTK for improved sentence segmentation
pip install nltk
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env if using cloud API keys

# Edit config.yaml — at minimum set:
#   input.file: /path/to/your-book.epub
#   project.name: "Your Book Title"
#   tts.engine: kokoro
#   tts.kokoro.voice: af_heart
```

### 4. Run

```bash
# Full pipeline
python -m audiobook_forge forge --input /path/to/book.epub

# With emotion tagging enabled
python -m audiobook_forge forge --input book.epub --emotion

# Override engine
python -m audiobook_forge forge --input book.txt --engine kokoro --voice af_bella

# Check progress of an interrupted run
python -m audiobook_forge status

# Reset checkpoint to start over
python -m audiobook_forge reset --confirm
```

### 5. Output

```
output/
├── chapter_0000.wav      # Individual chapter audio files
├── chapter_0001.wav
├── ...
├── Your Book Title.m4b   # Final audiobook with chapter markers
└── forge.log             # Processing log
```

---

## TTS Engine Guide

### Kokoro TTS (Recommended Default)

| Property         | Value                                          |
| ---------------- | ---------------------------------------------- |
| VRAM             | < 2 GB                                         |
| Speed            | ~100x real-time on GPU                         |
| License          | Apache 2.0 (fully permissive)                  |
| Voice cloning    | No (fixed voice library)                       |
| Emotion control  | Indirect (speed adjustment per emotion)        |
| Long-form stable | Excellent                                      |

Best for: Fast, reliable audiobook generation. A 10-hour book processes in
under 30 minutes on an RTX 4070 Ti.

```bash
pip install kokoro>=0.9 soundfile
```

Available voices: `af_heart`, `af_bella`, `af_nicole`, `af_sarah`, `af_sky`,
`am_adam`, `am_michael` (see [Kokoro docs](https://github.com/hexgrad/kokoro)).

### Fish Audio S2

| Property         | Value                                          |
| ---------------- | ---------------------------------------------- |
| VRAM             | ~17 GB (BF16); ~9-10 GB (fp16/quantized)      |
| Speed            | ~7x real-time on RTX 4090                      |
| License          | Apache 2.0 (code) / Research License (weights) |
| Voice cloning    | Yes (10-30s reference audio)                   |
| Emotion control  | Excellent (15,000+ inline tags)                |
| Long-form stable | Very good (RL-tuned for consistency)           |

Best for: Maximum quality and emotion control. Requires 24 GB VRAM for full
model, or fp16/quantized for 12 GB cards.

**VRAM warning for RTX 4070 Ti (12 GB):** The full S2 Pro model needs ~17 GB.
Try `--half` mode or community fp8 quantizations. Fish Speech v1.5 (~2 GB) is a
lower-quality fallback.

```bash
# Start the Fish Audio server separately:
cd fish-speech
python -m tools.api_server --listen 0.0.0.0:8080 --half --compile

# Then in config.yaml:
# tts.engine: fish_audio
# tts.fish_audio.api_url: http://localhost:8080
# tts.fish_audio.reference_audio: /path/to/narrator_voice.wav
```

### OpenAI-Compatible API

Works with LM Studio, vLLM, or any OpenAI-compatible TTS endpoint.

```yaml
tts:
  engine: openai_compat
  openai_compat:
    api_url: http://localhost:1234/v1
    model: tts-1
    voice: alloy
```

---

## Emotion System

The emotion pipeline applies conservative prosody hints to improve narration
without over-dramatizing.

### Rules Mode (default when `--emotion` is passed)

1. All sentences default to `emotion: neutral`, `intensity: 0.3`
2. Dialogue is detected via quotation marks
3. Attribution verbs set emotion:
   - "whispered" / "murmured" → `whispered`
   - "shouted" / "screamed" → `excited`
   - "sobbed" / "cried" → `sad`
   - "snapped" / "growled" → `tense`
4. Punctuation signals adjust intensity: `!` (+0.15), `?` in dialogue → `curious`
5. Guardrails: max 30% non-neutral sentences per chapter, intensity capped at 0.7

### LLM Mode (optional refinement)

Set `emotion.mode: llm` in config. Ambiguous sentences are batched to an
OpenAI-compatible endpoint for structured emotion labels.

```yaml
emotion:
  enabled: true
  mode: llm
  llm:
    api_url: http://localhost:1234/v1
    model: local-model
    batch_size: 20
```

### Per-Sentence Schema

```json
{
  "sentence": "\"I can't believe it!\" she cried.",
  "speaker": "character",
  "narration_mode": "dialogue",
  "emotion": "excited",
  "intensity": 0.5,
  "pause_ms": 250
}
```

Valid emotions: `neutral`, `calm`, `curious`, `tense`, `excited`, `sad`,
`whispered`, `angry`

---

## Architecture

```
Input (.epub/.txt)
       │
       ▼
┌─────────────┐
│  Ingestion   │  epub_reader / txt_reader
│  + Chapters  │  chapter detection, cover extraction
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Text Proc   │  normalize → segment → detect dialogue
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Emotion     │  rules-based → optional LLM refinement
│  Tagger      │  → guardrails → AnnotatedSentence[]
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  TTS Engine  │  Kokoro / Fish Audio / OpenAI-compat
│  (GPU)       │  chunk-by-chunk synthesis
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Post-Proc   │  trim silence → loudnorm → resample
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  M4B Assembly│  concat → AAC encode → chapters → cover
└──────┬──────┘
       │
       ▼
   Output .m4b
```

### Project Structure

```
audiobook-forge/
├── config.yaml                 # Main configuration
├── .env.example                # API keys template
├── requirements.txt
├── README.md
├── samples/
│   └── alice_chapter1.txt      # Test sample
├── tests/
│   └── test_pipeline.py        # End-to-end test
└── src/audiobook_forge/
    ├── __init__.py
    ├── __main__.py             # python -m audiobook_forge
    ├── cli.py                  # CLI (forge, status, reset)
    ├── config.py               # YAML + .env config loader
    ├── checkpoint.py           # Resumable state manager
    ├── pipeline.py             # Main orchestrator
    ├── ingestion/
    │   ├── epub_reader.py      # EPUB → BookData
    │   ├── txt_reader.py       # TXT → BookData
    │   └── reader.py           # Format dispatcher
    ├── processing/
    │   ├── text_normalizer.py  # Unicode, numbers, abbreviations
    │   ├── sentence_segmenter.py  # Split + chunk
    │   ├── dialogue_detector.py   # Quotes, attribution verbs
    │   └── emotion_tagger.py      # Rules + optional LLM
    ├── tts/
    │   ├── base.py             # BaseTTSEngine interface
    │   ├── kokoro_engine.py    # Kokoro TTS
    │   ├── fish_audio_engine.py   # Fish Audio S2 HTTP
    │   └── openai_compat_engine.py  # OpenAI-compatible
    └── audio/
        ├── postprocessor.py    # ffmpeg loudnorm, trim, resample
        └── m4b_assembler.py    # M4B with chapters + cover
```

---

## Performance Expectations (RTX 4070 Ti, 12 GB)

| Engine           | VRAM    | Speed          | 10-hour book ETA |
| ---------------- | ------- | -------------- | ---------------- |
| Kokoro TTS       | < 2 GB  | ~100x realtime | ~6 minutes       |
| Fish Audio S2*   | 9-17 GB | ~3-7x realtime | ~1.5-3.5 hours   |
| OpenAI-compat    | varies  | API-dependent  | varies           |

*Fish Audio S2 at fp16 may fit in 12 GB; full BF16 requires 24 GB.

### Bottlenecks

1. **TTS inference** — dominates total time for all engines
2. **ffmpeg post-processing** — ~5-10% of total time (CPU-bound)
3. **M4B assembly** — fast, < 1 minute for most books
4. **Text processing** — negligible (< 1 second for full novels)

---

## Resuming Interrupted Runs

The checkpoint system tracks every completed chunk and chapter. If the process
is interrupted (Ctrl+C, crash, shutdown), re-run the same command and it picks
up exactly where it left off:

```bash
# First run (interrupted after chapter 5)
python -m audiobook_forge forge --input book.epub
# ^C

# Resume from chapter 6
python -m audiobook_forge forge --input book.epub

# Check progress
python -m audiobook_forge status

# Force restart
python -m audiobook_forge reset --confirm
```

The checkpoint is stored at `output/.checkpoint.json` by default. It includes a
SHA-256 hash of the input file — if the book changes, the checkpoint resets
automatically.

---

## Risks and Next Improvements

### Known Risks

| Risk                       | Mitigation                                    |
| -------------------------- | --------------------------------------------- |
| Fish Audio S2 VRAM on 12GB | Use fp16/quantized, or fall back to Kokoro     |
| Kokoro voice variety       | Limited to built-in voices; no cloning         |
| Emotion tag over-labeling  | Guardrails cap at 30% non-neutral, 0.7 max    |
| ffmpeg not installed       | Clear error messages with install instructions |
| Long chapter memory drift  | Chunk-based synthesis with consistent voice ref|

### Planned Improvements

1. **Web UI** — Gradio-based local interface for browsing output and adjusting settings
2. **Multi-speaker support** — assign different voices to different characters
3. **Streaming synthesis** — real-time preview while generating
4. **Quality validation** — automatic WER check on generated audio
5. **Batch processing** — process multiple books from a directory
6. **Fine-tuned voices** — StyleTTS2 integration for custom narrator voices
7. **Parallel chapter processing** — generate multiple chapters simultaneously

---

## License

The pipeline code is provided for personal use. TTS model licenses vary:

| Component       | License                        | Commercial Use |
| --------------- | ------------------------------ | -------------- |
| This pipeline   | MIT                            | Yes            |
| Kokoro TTS      | Apache 2.0                     | Yes            |
| Fish Audio S2   | Apache 2.0 (code) / Research (weights) | Weights: contact Fish Audio |
| XTTS v2         | CPML                           | Requires license |
