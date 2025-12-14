# WhisperScribe

<img src="static/banner.svg" width="300px" alt="WhisperScribe banner" style="display: block; margin: 2em auto">

Self-hosted Discord bot for live, multi-user voice transcription and automated session wrap-ups tailored for tabletop RPGs.

WhisperScribe runs as two cooperating services:

- TypeScript Discord Gateway (Node): joins a voice channel, decodes Opus, performs per-user VAD/segmentation, logs events, serves in-Discord commands, and generates wrap-ups via Gemini.
- Python ASR Service: runs Whisper (transformers) locally and exposes a WebSocket API for on-device speech-to-text.

Both services write session artifacts under `data/<SESSION_NAME>/`.

## Features

- Live, multi-user transcription from Discord voice with per-user segmentation
- Local Whisper STT via transformers (CPU or GPU)
- Session logging to JSONL and formatted logs-on-demand (`!log`)
- One-command session wrap-up (`!wrapup`) via Gemini structured output, saved as Markdown
- Optional upload of wrap-up + logs to a private GitHub Gist
- Configurable user aliases and phrase normalization to clean up STT artifacts
- Profiles to override prompts, vocabulary, and permissions per campaign

## Architecture

Discord ↔ Node (discord.js) ⇄ WebSocket ⇄ Python (Whisper)

- Node captures per-user audio from Discord, performs VAD/segmentation, and sends finalized mono@16k PCM chunks to Python.
- Python normalizes/enhances audio, runs Whisper, and streams back transcriptions.
- Node appends to `data/<session>/log.jsonl` and generates wrap-ups on demand.

## Requirements

- Node.js 20+ and npm
- Python 3.12+ and uv (recommended)
- For GPU acceleration: a supported PyTorch build for your platform/driver

## Environment

Create a `.env` file by copying the example and setting the required secrets:

```bash
cp .env.example .env
```

Important variables:

- `DISCORD_TOKEN` — required for the bot to connect to Discord.
- `GEMINI_API_KEY` — required to run `!wrapup` and wrapup CLI commands that use Gemini.
- `GITHUB_TOKEN` — optional; used when `--gist` is specified to upload wrapups.

Secrets are intentionally loaded from environment variables — do not store them in `config.toml`.

## Quickstart (Docker)

1) Prepare configuration files:

```bash
cp .env.example .env
cp config.example.toml config.toml
```

2) Run everything in one container (ASR + bot):

```bash
VOICE_CHANNEL_ID=1234567890 docker compose up --build all
```

## Install & Run Locally

1) Prepare configuration files:

```bash
cp .env.example .env
cp config.example.toml config.toml
```

2) Python (ASR) dependencies and build:

```bash
cd py
uv venv --system-site-packages
uv sync
```

3) Node (bot) dependencies and build:

```bash
cd ../js
npm ci
npm run build
```

4) Start the Python ASR server:

```bash
cd ../py
uv run start
```

5) Start the Node Discord bot (in another terminal):

```bash
cd js
npm run start -- bot <VOICE_CHANNEL_ID> --ai-service-url ws://localhost:8771
```

## Running with Docker Compose

The repository includes a single `docker-compose.yml` that supports three modes via the `MODE` env var: `asr`, `bot`, and `all`.

- `asr` runs the Python ASR service only
- `bot` runs the Node bot only
- `all` runs both ASR and bot in one container (convenient for simple setups)

### Sample commands

- Run everything in one container:

```bash
VOICE_CHANNEL_ID=1234567890 docker compose up all
```

- Run ASR separately, then bot:

```bash
docker compose up asr
ASR_HOST=asr ASR_PORT=8771 VOICE_CHANNEL_ID=1234567890 docker compose up bot
```

If running split services, ensure the bot can reach the ASR service by setting `ASR_HOST` to the ASR service name (e.g., `asr`) and `ASR_PORT` to `8771`.

### Docker Compose environment variables

Docker Compose reads variables from your shell and from `.env` (the compose file loads it via `env_file: ./.env`).

Secrets:

- `DISCORD_TOKEN`: required for `bot`/`all` containers
- `GEMINI_API_KEY`: required to generate wrapups
- `GITHUB_TOKEN`: optional; used for gist uploads

Bot runtime variables:

- `VOICE_CHANNEL_ID`: voice channel to join (required for `bot` or `all` if auto-start desired)
- `SESSION`: adds `--session-name <SESSION>` to the bot CLI
- `PROFILE`: adds `--profile <PROFILE>` to the bot CLI
- `GIST`: when truthy (`1|true|yes|on`) the container will pass `--gist` to the bot CLI
- `PREV_SESSION`: adds `--prev-session <PREV_SESSION>` to the bot CLI

ASR runtime variables:

- `ASR_HOST`: bind host for ASR service (default `0.0.0.0`)
- `ASR_PORT`: bind port for ASR (default `8771`)

Pass-through arguments:

- `BOT_ARGS`: extra args appended to `node js/dist/index.js bot ...` inside the container
- `ASR_ARGS`: extra args appended to the Python ASR server invocation

Build/runtime knobs:

- `BACKEND`: Docker build backend (`cpu|cuda|rocm`) — default `cpu`
- `DEVICE`: Python device selector (`auto|cpu|cuda|rocm`) — default `auto`
- `UID`/`GID`: container user/group IDs, defaults to `1000`
- `DEBUG`: if set, debug logging is enabled
- `HF_HOME`: path on the host to mount for the Hugging Face cache (recommended `$HOME/.cache/huggingface`)

## In-Discord commands

- `!log` — replies with a formatted text log attachment for the current session
- `!wrapup` — generates a structured wrap-up via Gemini and returns `wrapup.md`

Permissions: if `discord.allowed_commanders` (or a profile override) is non-empty, only those IDs may run the commands.

## CLI helpers (Node)

You can operate on recorded sessions from `js/` without connecting to Discord:

```bash
# Print a formatted log
npm run start -- log <SESSION_NAME>

# Generate/refresh a wrapup (uses cached unless --new)
npm run start -- wrapup <SESSION_NAME> --profile example --new --gist
```

## Data layout

Artifacts are stored under `data/<SESSION_NAME>/`:

- `log.jsonl` — JSON lines: `{ userId, displayName, startTs, endTs, origin, text }`
- `wrapup.md` — Markdown recap

## Config reference (`config.toml`)

See `config.example.toml` for full options. Highlights:

- `[discord]`
  - `allowed_commanders`: list of user IDs allowed to run bot commands
- `[net]`
  - `ai_service_url`: WebSocket URL the Node bot uses to reach the Python ASR service
- `[whisper]` / `[whisper]` (Python)
  - `model`, `logprob_threshold`, `no_speech_threshold`, `prompt`
- `[voice]` (Node segmenter)
  - `vad_db_threshold`, `silence_gap_ms`, `vad_frame_ms`, `max_segment_ms`, `min_segment_ms`
- `[wrapup]` (Node wrap-up generator)
  - `model`: Gemini model id (e.g., `gemini-2.5-flash`)
  - `tips`, `vocabulary`, `prompt`, `temperature`, `max_output_tokens`
- `[userid_map]` / `[phrase_map]`
  - map user IDs → aliases; normalize common mis-hearings in logs before wrap-up
- `[profile.<name>]`
  - per-campaign overrides for prompts, tips, vocabulary, and allowed commanders

## License

See `LICENSE`.
