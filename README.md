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
poetry run python bot.py [--join CHANNEL_ID]
```

- `--join CHANNEL_ID` (`-j`) — Automatically join the specified voice channel on startup without needing `!join`.
- Once running, use `!join` in any text channel to have the bot join your current voice channel and start recording.

## Configuration

All settings are loaded from environment variables via `python-dotenv` in `config.py`. Key options:

- `DISCORD_TOKEN` — Your bot token.
- `ALLOWED_GUILDS` — (Optional) Restrict text commands to specific servers.
- `WHISPER_MODEL` — Whisper model identifier for transcription.
- `SILENCE_THRESHOLD_SECONDS` — Seconds of silence before sending audio to Whisper.

## How It Works

- **bot.py**: Defines `STTBot` (subclass of `discord.ext.commands.Bot`) with commands, event handlers, and reconnection logic.
- **sink.py**: Implements `SilenceSink` to buffer audio per user, detect silence, and send raw audio chunks for processing.
- **audio_processor.py**: Loads Whisper model, accepts raw audio jobs, and enqueues transcription results.
- **log_manager.py**: Receives transcription and chat logs, writes them to console and session log files.
- **config.py**: Loads and validates environment configuration.

## Development & Contributing

- Ensure FFmpeg is installed.
- Run tests (if available) and lint before submitting PRs.
- Contributions, issues, and enhancements are welcome.

## License

MIT License. See [LICENSE](LICENSE) for details.
