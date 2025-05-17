# WhisperScribe

<img src="static/banner.svg" width="300px" alt="whisper scribe banner" style="display: block; margin: 2em auto">

*WhisperScribe* is a self-hosted Discord bot for live, multi-user voice transcription and automated session summarization, designed for tabletop RPGs. It leverages local speech-to-text transformer models (Whisper) to transcribe Discord voice and text channels as you play. In chat, users can command the bot to post session logs and an AI-generated session summary (requires OpenAI key).

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
- If you provide a github token and the `--gist` CLI flag, it'll upload wrapup/log files to github gist instead of directly to discord.

Unless everyone in your group has a AAA podcasting setup in a perfect environment, transcripts will probably be riddled with artifacts and misheard words. Adding your most commonly used proper nouns to the `whisper.prompt` config can help. However, I do find that modern retail LLMs are actually pretty good at teasing out a remarkably cohesive narrative from low quality transcripts, so you can probably still get a useful wrapup summary.

## Example Output

You can find a log and outline generated from a real session [HERE](https://gist.github.com/merklejerk/25c4504a51c7e67b1c7a3b5199459a49).

## How It Works

- `py-cord` provides Discord API and voice channel access for real-time audio capture and bot stuff.
- `silero-vad` for voice activity detection and for segmenting user speech.
- `transformers` + Whisper for local speech-to-text transcription.
- `openai` (GPT) for generating session summaries.
- `ollama` for [refiner](#optional-transcript-refiner).
- Lots of random ugly hacks to get all the libraries to work stabley together.

## TODO

- Swap out refiner's ollama dependency for transformers.
- Docker container.

## Advanced Features

### Optional: Transcript Refiner

If you're encountering very many transcription errors, you can experiment with theoptional refiner pipeline, which uses a local LLM (via Ollama) to clean up and correct transcription artifacts in real To enable, spin up ollama locally, install the model of choice, and configure `[refiner.model]` in your `config.toml`. The model must support structured output. Note that this can be demanding on your system and a bit finnicky.