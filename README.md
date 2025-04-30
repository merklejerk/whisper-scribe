# Discord STT Bot

A lightweight Discord bot that joins a voice channel, records speech, transcribes it locally using OpenAI Whisper, and logs both voice transcriptions and chat messages. Still WIP.

## Features

- Join a Discord voice channel and record speech in real time.
- Say `!join` anywhere in the server to join the summoner's current voice channel.
- CLI option to auto‑join on startup (`-j` / `--join`).
- Silence detection: processes audio when users stop speaking.
- Local transcription with Whisper (configurable model size).
- Logs voice transcriptions and text messages to console and persistent session logs.
- Guild/Server whitelist support.

## TODO

- Still needs small tweaks to VAD.
- AI-based transcription summary.

## Requirements

- Python 3.8+  
- Poetry for dependency management  
- FFmpeg installed and available in `PATH` (for audio handling)

## Installation

### 1. Clone the repo
```bash
git clone <repo_url>
cd discord-stt
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
At any time you can say `!wrapup` in the voice channel's text channel to have the bot produce wrapup files.

## How It Works

- **Discord Integration:** [py-cord] provides Discord API and voice channel access for real-time audio capture and command handling.
- **Audio Capture & VAD:** [silero-vad] and [webrtcvad] enable voice activity detection and silence detection for segmenting user speech.
- **Transcription:** [torch], [torchaudio], and [transformers] power the Whisper model for local speech-to-text transcription.
- **Wrapups:** [openai] library is used to generate session summaries via GPT models.

## License

MIT License. See [LICENSE](LICENSE) for details.
