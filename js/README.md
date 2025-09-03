# WhisperScribe Node.js Discord Gateway Service (WIP)

This service replaces the legacy Python `py-cord` layer. It connects to Discord (gateway + voice), captures audio, and streams it to the Python transcription service over a WebSocket protocol.

## High-Level Flow (Phase 1)

1. Login with `DISCORD_TOKEN`.
2. Join target voice channel (env or CLI supplied `VOICE_CHANNEL_ID`).
3. Capture user audio via `@discordjs/voice` receiver.
4. Assemble fixed-duration PCM chunks per user (e.g. 1000ms @ 48kHz stereo → mono downmix).
5. Base64 encode and send `audio.segment` messages to Python service.
6. Forward `!wrapup` / `!log` commands from the associated text channel.
7. Receive events (`transcription.segment`, `wrapup.ready`) and post messages/files back to Discord.

## Status

Scaffold only. Wire-up to Discord and Python service will follow incremental tasks.

## Env Vars

- `DISCORD_TOKEN` (required)
- `VOICE_CHANNEL_ID` (required for auto-join) or pass via CLI
- `PY_SERVICE_URL` (WebSocket URL, e.g. `ws://localhost:8765`)
- `GUILD_ID` (optional constraint)
- `SESSION_NAME` (optional override, else timestamp)
- `CHUNK_MS` (default 1000)

## Development

Install deps (after adding `package.json`):

```
npm install
```

Run (dev):

```
npx ts-node-dev src/index.ts --voice <VOICE_CHANNEL_ID>
```

## Next Steps

See root `refactor-plan.md` section 12.
