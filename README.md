# Meeting Transcription & Summarization Service 🎙️

> Dual-language transcription (Hindi + English) with AI-powered meeting summaries using Whisper and Claude Opus 4.1

Transform your meeting recordings into structured summaries with topics, key decisions, action items, and open points - all in minutes.

## ✨ Key Features

- **🌐 Dual-Language Transcription** - Transcribes in both Hindi and English for comprehensive coverage
- **🤖 AI-Powered Summaries** - Claude Opus 4.1 analyzes both transcripts for accurate summaries
- **📋 Structured Output** - Organized sections: overall summary, key decisions, topics, action items, and open points
- **🔧 Type-Safe** - Pydantic models for programmatic access
- **📝 Multi-Format** - Saves raw transcripts and formatted summaries
- **🚀 Production Ready** - Built with Python 3.11+, Poetry, and enterprise-grade tools

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Poetry
- Databricks token for Claude Opus 4.1 access
- ffmpeg (for audio processing)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd transcribe-meetings

# Install dependencies
poetry install

# Configure environment
cp .env.example .env
# Edit .env and add your DATABRICKS_TOKEN

# Install ffmpeg (macOS)
brew install ffmpeg
```

### Basic Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Process audio files
python -m src.cli -i path/to/meeting.mp3

# Or process entire directory
python -m src.cli -i path/to/meetings/

# Or process all files in in/ folder
python -m src.cli
```

## 📊 Output Structure

For each audio file, you get:

```
out/
├── meeting_transcript_hindi.txt    # Hindi transcription
├── meeting_transcript_english.txt  # English transcription
└── meeting.txt                     # Structured summary
```

**Summary Format:**

```
================================================================================
OVERALL SUMMARY
================================================================================
High-level overview of the meeting...

================================================================================
KEY DECISIONS
================================================================================
1. Major decision with context
2. Another key decision

================================================================================
SUMMARY BY TOPICS
================================================================================
**Topic 1**
• Key point
• Another point

**Topic 2**
• Discussion details

================================================================================
ACTION ITEMS
================================================================================
1. Person: Task description (Deadline: Date)
2. Person: Another task

================================================================================
OPEN POINTS
================================================================================
1. Unresolved issue requiring follow-up
2. Pending question or decision
```

## 🔧 Configuration

Edit `.env` file:

```env
# Required
DATABRICKS_TOKEN=your_token_here

# Optional
WHISPER_MODEL=large-v3
LOG_LEVEL=INFO
```

## 📖 Programmatic Usage

```python
from src.config import Config
from src.summarizer import Summarizer

config = Config.from_env()
summarizer = Summarizer(
    model=config.databricks_model,
    api_key=config.databricks_token,
    base_url=config.databricks_base_url
)

# Get structured summary
summary = summarizer.summarize(transcript)

# Access structured fields
print(f"Decisions: {len(summary.key_decisions)}")
print(f"Actions: {len(summary.action_items)}")
print(f"Open Points: {len(summary.open_points)}")

# Iterate sections
for decision in summary.key_decisions:
    print(f"✅ {decision}")

for item in summary.action_items:
    print(f"📋 {item}")
```

## 🏗️ Architecture

```
┌─────────────┐
│ Audio File  │
└──────┬──────┘
       ↓
┌─────────────────────────┐
│ Whisper large-v3        │
│ (Sequential Processing) │
└─────────┬───────────────┘
          ↓
    ┌─────────┴─────────┐
    ↓                   ↓
[Hindi Text]      [English Text]
    └──────┬────────────┘
           ↓
    ┌──────────────┐
    │ Claude Opus  │
    │ 4.1 Analysis │
    └──────┬───────┘
           ↓
  [Structured Summary]
```

## 📁 Project Structure

```
transcribe-meetings/
├── src/
│   ├── config.py          # Configuration with Pydantic
│   ├── transcriber.py     # Whisper integration
│   ├── summarizer.py      # Claude/DSPy integration
│   ├── pipeline.py        # Orchestration logic
│   └── cli.py             # Command-line interface
├── tests/                 # Test suite
├── in/                    # Input audio files
├── out/                   # Generated summaries
└── logs/                  # Application logs
```

## ⚡ Performance

**Transcription Speed (Dual-Language, CPU):**
- Short meetings (< 30 min): ~4-10 minutes
- Medium meetings (30-60 min): ~10-20 minutes
- Long meetings (1-2 hours): ~20-40 minutes

**Memory Requirements:**
- Whisper large-v3: ~10GB RAM
- Recommended: 16GB+ RAM (32GB+ optimal)

## 🛠️ Technology Stack

- **Transcription**: OpenAI Whisper (large-v3)
- **Summarization**: Claude Opus 4.1
- **Framework**: DSPy for LLM orchestration
- **Validation**: Pydantic v2
- **Logging**: Loguru
- **CLI**: Click
- **Dependencies**: Poetry

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📝 License

MIT

## 🙏 Acknowledgments

Built with:
- [Whisper](https://github.com/openai/whisper) by OpenAI
- [DSPy](https://github.com/stanfordnlp/dspy) by Stanford NLP
- [Claude Opus 4.1](https://www.anthropic.com/claude) by Anthropic

---

**Made with ❤️ for better meeting documentation**
