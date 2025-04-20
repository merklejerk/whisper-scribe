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

- Better VAD (voice activity detection).
- Hallucinates with noise (related to above).
- AI-based transcription summary.

## Requirements

- Python 3.8+  
- Poetry for dependency management  
- FFmpeg installed and available in `PATH` (for audio handling)

## Installation

1. Clone or download this repository:
   ```bash
   git clone <repo_url>
   cd discord-stt
   ```
2. Install dependencies via Poetry:
   ```bash
   poetry install
   ```
3. Create a `.env` file in the project root with the following variables:
   ```ini
   DISCORD_TOKEN="YOUR_DISCORD_BOT_TOKEN"
   # Optional: Comma‑separated list of guild IDs to restrict commands
   ALLOWED_GUILDS="123456789012345678,987654321098765432"
   # Optional: Whisper model name (e.g., "openai/whisper-small", default: "openai/whisper-large-v3")
   WHISPER_MODEL="openai/whisper-base"
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
