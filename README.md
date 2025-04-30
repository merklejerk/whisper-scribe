# WhisperScribe

WhisperScribe is a self-hosted Discord bot for real-time, multi-user voice transcription and automated session summarization, designed for tabletop RPGs. It leverages advanced speech-to-text models (Whisper via PyTorch and Transformers) to transcribe Discord voice channels locally, using voice activity detection (Silero) for accurate speech segmentation. Transcripts are logged in structured NDJSON files, and users can generate comprehensive session wrap-ups, including full transcripts and AI-powered summaries (requires cloud OpenAI key), on demand via a chat command.

## Requirements

- Python 3.13+
- Poetry for dependency management 
- OpenAI API key (for session summaries).

## Installation

### 1. Clone the repo
```bash
git clone <repo_url>
cd whisper-scribe
```
### 2. Install dependencies
```bash
poetry install
```
If you want to use RoCM (AMD) for pytorch, you will need to manually install the rocm version of torch libs:
```bash
poetry run pip install -I torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```
### 3. Configure environment
Copy `.env.example` to `.env` and populate it with your secrets:
```bash
cp .env.example .env
```
Copy `config.toml.example` to `config.toml` and populate it with your configuration:
```
cp .env.example config.toml
```

## Usage

Run the bot with Poetry:
```bash
poetry run python -m src.cli VOICE_CHANNEL_ID SESSION_NAME
```

The bot will instantly join the voice channel you provide and start transcribing to `logs/{SESSION_NAME}`.
At any time you can say `!wrapup` in the voice channel's text channel to have the bot produce wrapup files (transcript + session summary).

## How It Works

- **Discord Integration:** `py-cord` provides Discord API and voice channel access for real-time audio capture and command handling.
- **Audio Capture & VAD:** `silero-vad` and `webrtcvad` enable voice activity detection and silence detection for segmenting user speech.
- **Transcription:** `torch`, `torchaudio`, and `transformers` power the Whisper model for local speech-to-text transcription.
- **Wrapups:** `openai` library is used to generate session summaries via GPT models.

## License

MIT License. See [LICENSE](LICENSE) for details.
