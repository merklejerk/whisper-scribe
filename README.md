# WhisperScribe

<img src="static/banner.svg" width="300px" alt="whisper scribe banner" style="display: block; margin: 2em auto">


*WhisperScribe* is a self-hosted Discord bot for live, multi-user voice transcription and automated session summarization, designed for tabletop RPGs. It leverages advanced speech-to-text models (Whisper via PyTorch and Transformers) to transcribe Discord voice channels locally, using voice activity detection (Silero) for accurate speech segmentation. Transcripts are logged in structured NDJSON files, and users can generate comprehensive session wrap-ups, including full transcripts and AI-powered summaries (requires cloud OpenAI key), on demand via a chat command.

## Requirements

- Python 3.13+
- Poetry for dependency management
- OpenAI API key (for session summaries)

## Installation

### 1. Clone the repo
```bash
git clone git@github.com:merklejerk/whisper-scribe.git
cd whisper-scribe
```
### 2. Install dependencies
```bash
poetry install
```
If you want to use GPU acceleration on an AMD card, you will need to manually install the rocm version of pytorch libs:
```bash
poetry run pip install -I 'torch==2.7' 'torchaudio==2.7' --index-url https://download.pytorch.org/whl/rocm6.3
```
### 3. Configure environment
Copy `.env.example` to `.env` and populate it with your secrets:
```bash
cp .env.example .env
```
Copy `config.example.toml` to `config.toml` and populate it with your configuration:
```
cp .config.example.toml config.toml
```

If you're planning on running this CPU-only, I recommend setting your whisper `model` to `openai/whisper-small.en` for near real-time transcription. If you've got a compatible GPU, `openai/whisper-large-v3-turbo` works fantastic.

## Usage

Run the bot with Poetry:
```bash
poetry run python -m src.cli VOICE_CHANNEL_ID SESSION_NAME
```

- Voice channel ID is the numeric ID of the channel.
- The bot will instantly join the voice channel you provide and start transcribing to `logs/{SESSION_NAME}.ndjson`.
- At any time you can say `!wrapup` in the voice channel's text channel to have the bot produce wrapup files (transcript + session summary), which get posted to chat and also saved to `wrapup/`.
- You can also say `!log` to just generate the transcript.

## How It Works

- `py-cord` provides Discord API and voice channel access for real-time audio capture and bot stuff.
- `silero-vad` for voice activity detection and for segmenting user speech.
- `transformers` + Whisper for local speech-to-text transcription.
- `openai` (GPT) for generating session summaries.
- Lots of random ugly hacks to get all the libraries to work stabley together.

## TODO

- Use a small local LLM (Gemma 1B or 4B?) to clean up transcription artifacts and mistakes.
- Docker container.