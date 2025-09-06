# WhisperScribe

<img src="static/banner.svg" width="300px" alt="WhisperScribe banner" style="display: block; margin: 2em auto">

Self-hosted Discord bot for live, multi-user voice transcription and automated session wrap-ups tailored for tabletop RPGs. It now runs as two cooperating services:

- TypeScript Discord Gateway (Node): joins your voice channel, performs per-user VAD/segmentation, logs text/voice events, serves chat commands, and generates wrap-ups with Gemini.
- Python ASR Service: runs Whisper (Hugging Face transformers) locally and exposes a WebSocket API for fast, on-device speech-to-text.

Both services share a simple JSON message protocol over WebSocket and write session artifacts under `data/<SESSION_NAME>/`.

## Features

- Live, multi-user transcription from Discord voice with robust per-user segmentation
- Local Whisper STT via transformers (CPU or GPU)
- Session logging to JSONL and formatted logs-on-demand (`!log`)
- One-command session wrap-up (`!wrapup`) with Gemini structured output, saved as Markdown
- Optional upload of wrap-up + logs to a private GitHub Gist
- Configurable user aliases and phrase normalization to clean up STT artifacts
- Profiles to override prompts, vocabulary, and permissions per campaign

## Architecture

Discord ↔ Node (discord.js) ⇄ WebSocket ⇄ Python (Whisper)

- Node captures per-user audio from Discord, performs VAD/segmentation, and sends finalized mono@16k PCM chunks to Python.
- Python normalizes/enhances audio, runs Whisper, and streams back transcriptions.
- Node appends to `data/<session>/log.jsonl`, tracks context, and generates wrap-ups with Gemini on demand.

## Requirements

- Node.js 20+ and npm
- Python 3.12+ and uv (recommended)
- For GPU acceleration: a supported PyTorch build for your platform/driver. Otherwise, CPU works with smaller models.
- Linux/macOS/Windows are fine; Discord voice capture requires Opus decoding (bundled via `@discordjs/opus`).

## environment

You can run/develop in your host OS or use the provided Dev Containers (recommended):

- Dev Containers: ready-to-use images for `cpu`, `rocm`, and `cuda` backends.
	1) Copy `.devcontainer/.env.example` to `.devcontainer/.env`.
	2) Set `BACKEND=cpu`, `rocm`, or `cuda` (CUDA image is currently untested).
	3) Open the repo in VS Code and “Reopen in Container”. System packages for PyTorch (and friends like Triton) will be preinstalled for the chosen backend.

- Note: the Python service expects certain heavy deps (e.g., torch, triton, etc.) to be available on the `PYTHONPATH`. The devcontainers take care of this for you.

Environment variables (copy `.env.example` to `.env`):

- DISCORD_TOKEN: required for the bot
- GEMINI_API_KEY: required for wrap-up generation (Node)
- GITHUB_TOKEN: optional, enables gist uploads


## Install & Configure

1) Clone

```bash
git clone git@github.com:merklejerk/whisper-scribe.git
cd whisper-scribe
```

2) Config files

```bash
cp .env.example .env
cp config.example.toml config.toml
```

Edit `config.toml` as needed (see “Config reference”).

3) Python ASR service deps (virtual env with system site packages)

```bash
cd py
uv venv --system-site-packages
uv sync
```

4) Node gateway deps and build

```bash
cd ../js
npm install && npm run build
```

## Run

You run the Python ASR service, then the Node bot. The default WebSocket is `ws://localhost:8771` (configurable).

1) Start Python ASR

```bash
cd py
uv run start
```

This loads the Whisper model defined in `config.toml` and listens for audio jobs. To influence device selection, set `DEVICE` (e.g., `cuda`, `mps`, `cpu`) in your environment; default is `auto`.

2) Start the Node Discord bot

```bash
cd ../js
npm run start -- bot <VOICE_CHANNEL_ID> \
	--ai-service-url ws://localhost:8771 \
	--session-name "08.30.25" \
	--profile example \
	--prev-session "07.19.25" \
	--gist \
	--allowed-commanders 123456789012345678
```

Tips:

- Get the voice channel ID from Discord (User Settings → Advanced → Developer Mode; right-click channel → Copy ID).
- When running without `--session-name`, a UUID is used.
- If `--gist` is set and `GITHUB_TOKEN` is present, `wrapup.md` and `log.jsonl` are uploaded to a private gist on `!wrapup`.

### In-Discord commands

- `!log` — replies with a formatted text log attachment for the current session
- `!wrapup` — generates a structured wrap-up via Gemini and replies with `wrapup.md`

Permissions: if `discord.allowed_commanders` (or profile override) is non-empty, only those IDs/tags can run commands.

### CLI helpers (Node)

From `js/` you can operate on recorded sessions without Discord:

```bash
# Print formatted log
npm run start -- log <SESSION_NAME>

# Generate/refresh wrapup (uses cached file unless --new)
npm run start -- wrapup <SESSION_NAME> --profile example --new --gist
```

## Data layout

Artifacts are stored under `data/<SESSION_NAME>/`:

- `log.jsonl` — JSON lines: { userId, displayName, startTs, endTs, origin, text }
- `wrapup.md` — Markdown recap with scenes, quotes, items, developments

## Config reference (config.toml)

See `config.example.toml` for full examples. Highlights:

- `[discord]`
	- `allowed_commanders`: restrict who can run `!log`/`!wrapup`.
- `[net]`
	- `ai_service_url`: WebSocket URL the Node bot uses (and Python binds to).
- `[whisper]`
	- `model`: e.g., `openai/whisper-small.en` for CPU, `whisper-v3-large-turbo` for strong GPUs
	- `logprob_threshold`, `no_speech_threshold`, `prompt`: base system prompt for Whisper
- `[voice]` (Node segmenter)
	- `vad_db_threshold`, `silence_gap_ms`, `vad_frame_ms`, `max_segment_ms`, `min_segment_ms`
- `[wrapup]` (Node wrap-up generator)
	- `model`: Gemini model id (e.g., `gemini-2.5-flash`)
	- `tips`, `vocabulary`, `prompt`, `temperature`, `max_output_tokens`
- `[userid_map]` / `[phrase_map]`
	- Map user IDs → aliases; normalize common mis-hearings in logs before wrap-up.
- `[profile.<name>]`
	- Per-campaign overrides: `whisper_prompt`, `wrapup_prompt`, `wrapup_tips`, `wrapup_vocabulary`, `allowed_commanders`, and nested `userid_map` / `phrase_map` merges.

Notes:

- The Node gateway appends a rolling text window to the ASR prompt (`[whisper].prompt`) to help resolve pronouns and local context.
- The Python service ignores `[wrapup]` — wrap-ups are generated entirely on the Node side.

## Model guidance

- CPU-only: prefer `openai/whisper-small.en` for near‑real‑time.
- GPU (CUDA/MPS/ROCm): prefer `whisper-v3-large-turbo` for best accuracy/performance balance.
- Set `DEVICE` to force device (e.g., `DEVICE=cuda`) when starting the Python service.

## License

See `LICENSE`.