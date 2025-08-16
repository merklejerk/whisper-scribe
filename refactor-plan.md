# WhisperScribe Refactor Plan: Split Discord Integration (JS) From STT + NLP Pipeline (Python)

Goal: Replace broken py-cord voice + command handling with a Node.js (discord.js v14+) companion service while preserving and improving the existing Python audio processing, transcription, refinement, and wrapup generation pipeline.

---
## 0. Guiding Principles
- Minimize changes to stable Python transcription / refinement code paths.
- Keep the Python pipeline headless (no direct Discord dependency).
- Use a clear, versioned inter-process protocol (JSON messages over WebSocket or gRPC; start with WebSocket for speed of implementation, later optional upgrade).
- Make components independently runnable & testable.
- Enable future scaling (multiple Discord guild sessions feeding a pool of Python workers) without redesign.

---
## 1. Target Architecture
```
+--------------------+        WS / IPC        +-------------------------+
|  discord-gateway   |  JSON control/events  |  python-transcriber     |
|  (Node.js service) | <--------------------> |  (audio+NLP pipeline)   |
|                    |  audio chunk stream   |                         |
+--------------------+        (binary/JSON)   +-------------------------+
        |  ^                               ^            ^
        |  |                               |            |
 Discord API (voice, guild, text)     Model inference  Wrapup (OpenAI / local)
```

### 1.1 Node.js Service Responsibilities
- Connect/login to Discord (discord.js)
- Manage guild + voice channel joins
- Capture raw Opus/PCM voice frames per user
- Demultiplex, normalize timestamps, attach user metadata
- Stream audio segments to Python (already VAD-segmented OR raw frame stream TBD)
- Forward text channel commands (!wrapup, !log) to Python service
- Receive transcription events & post to text channel (optional realtime mode)
- Receive wrapup/log file generation results; upload to Discord (or trigger gist flow via Python and forward URL)

### 1.2 Python Service Responsibilities
- Accept session start/stop + audio segment messages
- Apply VAD (either keep existing or move to JS—initially KEEP in Python to reduce risk)
- Enhance + resample audio, run Whisper transcription
- (Optional) Run refiner (Ollama) if configured
- Maintain rolling context & log writing (current `logging.py` + `_recent_log_entries` logic)
- On wrapup request: load log entries, generate outline & transcript, return paths/URLs
- Provide progress + error events back to Node.js

### 1.3 Communication Channel
- Phase 1: Single WebSocket connection (Node.js = client, Python = server) OR reverse.
  - Choose: Python listens (simpler reload cycle). Node connects.
- Message framing: newline-delimited JSON (text) for control/events; binary frames for audio.
  - Simpler alternative: All JSON; base64 audio payload (higher overhead). Start with base64 for speed; optimize later.
- Version key in every message: `"v":1`.

### 1.4 Message Types (Initial Draft)
Control -> Python:
- `session.start { v, session_id, guild_id, voice_channel_id, start_ts }`
- `session.stop { v, session_id }`
- `audio.chunk { v, session_id, user_id, user_name, index, pcm_format:{sr,channels,sample_width}, started_ts, capture_ts, duration_ms, data_b64 }`
- `command.wrapup { v, session_id, request_id }`
- `command.log { v, session_id, request_id }`

Python -> Node:
- `ack { v, ref_type, ref_id, ok, error? }`
- `transcription.segment { v, session_id, user_id, user_name, capture_ts, text, refined? }`
- `log.rotated { v, session_id, path }`
- `wrapup.ready { v, session_id, request_id, transcript_path, outline_path?, gist_url? }`
- `error { v, code, message, details? }`
- `session.status { v, session_id, state, reason? }`

Future (Optional):
- `audio.segment` (if VAD moved to Node)
- `metrics.update`

---
## 2. Repository Restructure
```
/ (root)
  refactor-plan.md
  pyproject.toml
  package.json (new, for Node service)
  /python/ (move current src/* here gradually or alias)
     src/ (existing modules)
  /js/
     src/
        index.ts (bootstrap)
        discordClient.ts
        voiceReceiver.ts
        websocketClient.ts
        commandRouter.ts
        audioEncoder.ts (PCM assembly/base64)
     test/
  /protocol/
     schema/ (JSON Schema or TypeScript + Pydantic models)
       messages.json
       ts/ (generated TS types)
       py/ (pydantic models auto-generated)
```
Short-term: keep Python at root for minimal breakage; introduce `/js` alongside. After stable, optionally move Python code under `/python` and adjust imports / packaging.

---
## 3. Python Refactor Tasks
1. Extract Discord-specific logic out of `bot.py`:
   - Create `session_manager.py` (start_session, stop_session, push_audio_chunk, submit_command).
   - Rename `DiscordBot` → `TranscriptionService` (headless) or replace with functional interface.
2. Implement a WebSocket server (`ws_server.py`) using `websockets` or `aiohttp` (whichever already available—NEITHER is installed now). ACTION: Ask user to add dependency (e.g., `pip install websockets`).
3. Define Pydantic models for protocol in `protocol/models.py` for validation.
4. Replace direct voice capture pipeline entrypoint with `push_audio_chunk(metadata, pcm_bytes)` feeding existing VAD path:
   - Adapt `VoiceCaptureSink` logic: either reuse enhancement + VAD pieces (move into utility `vad_pipeline.py`).
5. Adjust transcription queue consumer to emit events via WS to Node instead of (or in addition to) printing.
6. Wrap existing wrapup generation functions to respond to `command.wrapup` and `command.log` messages.
7. Maintain logging: keep JSONL writing unchanged.
8. Provide graceful shutdown on `session.stop`.

---
## 4. Node.js Service Implementation Tasks
1. Initialize project (`npm init -y` or `pnpm init`). Add dependencies: `discord.js`, `ws`, `prism-media` (for Opus decoding), `@types/ws`, `typescript`, `ts-node-dev`.
2. Implement config loader (env: DISCORD_TOKEN, PY_SERVICE_URL, GUILD_LIMITS...).
3. Discord login + ready lifecycle.
4. Voice join + raw Opus packet capture.
   - Use `discord.js` voice subsystem (`@discordjs/voice`): create receiver, subscribe to users.
   - Decode to PCM 16-bit 48kHz, downsample to 24kHz/16kHz if desired now or leave for Python.
5. Chunk assembly: accumulate frames per user until size or time threshold (e.g., 1s / 48000 samples) then send `audio.chunk`.
6. Implement WS client to Python with reconnect + backpressure awareness (queue length high-water mark to drop oldest or pause capture).
7. Command listener (messageCreate) in appropriate text channel -> map `!wrapup` / `!log` -> send command messages, post acknowledgements.
8. Handle inbound events (transcription.segment → post to channel if realtime mode enabled; wrapup.ready → attach files or gist URL).
9. Add structured logging (pino or simple console) with correlation IDs (request_id).

---
## 5. Protocol & Schemas
- Define JSON Schema for each message type under `/protocol/schema`.
- Auto-generate TS types via `typescript-json-schema` or keep manually in `protocol/types.ts` initially.
- Generate Pydantic models from the same schema (or manually mirror) to ensure validation on both ends.
- Include `v` (protocol version) and `type` in every message.
- Maintain backward compatibility by additive evolution; breaking changes -> bump `v`.

---
## 6. Audio Handling Strategy
Phase 1 (simplest):
- Node sends continuous fixed-duration PCM chunks (e.g., 0.5–1.0s) per user (base64) without VAD.
- Python concatenates per-user buffers, runs existing VAD pipeline unchanged to produce speech segments → transcription.
Pros: Reuse Python’s tuned logic.
Cons: More bandwidth.

Phase 2 (optimization):
- Move VAD to Node using a JS VAD lib or wasm; send only speech segments.

---
## 7. Security & Hardening
- Mutual secret (shared token) in WS headers or first message (HMAC challenge later).
- Rate limit commands per guild.
- Validate all inbound messages; reject oversize audio (> e.g., 5 MB).
- Sanitize filenames (session_name) before file IO.

---
## 8. Deployment Considerations
- Two processes (Node + Python) orchestrated by:
  - Docker Compose (future): `services: node, python` with shared volume for logs/wrapups OR pure WS streaming no shared FS.
- For gist uploads: keep logic centralized (Python already does). Node just relays event.
- Health endpoints: Python `/healthz` (aiohttp optional) and Node `/healthz` (Express optional) later.

---
## 9. Migration Steps
1. Add plan file (this one) and commit.
2. Scaffold `/js` project (no integration yet).
3. Introduce protocol schemas & Pydantic models (Python) + dummy WS server that logs received messages.
4. Node: Implement WS client + stub sending heartbeat, session.start, then close.
5. Replace `DiscordBot.start` usage in CLI with new Node-driven model (new `python_service.py` entrypoint).
6. Remove py-cord dependency once stable (or gate old path behind flag `--legacy-discord`).
7. Comprehensive testing of end-to-end transcription accuracy vs legacy path (same audio fixture -> compare transcripts).

---
## 10. Testing Strategy
- Unit: Protocol model validation, audio chunk assembler, VAD segmentation given synthetic sine + silence.
- Integration: Replay recorded PCM frames through WS to Python, assert transcript lines produced.
- E2E: Local docker-compose bringing up Node + Python + mock Discord (or real test guild) with short audio injection.
- Regression: Store golden transcripts for sample session & diff after refactors.

---
## 11. Risk Mitigation
| Risk | Mitigation |
|------|------------|
| Latency increases due to base64 overhead | Optimize to binary frames later |
| Audio drift / misaligned timestamps | Include capture_ts per chunk; Python normalizes sequencing |
| Backpressure leads to OOM | Implement bounded queue + drop oldest or pause capture |
| Diverging schemas | Single source JSON Schema + codegen |
| Partial migration complexity | Keep legacy path behind feature flag until parity confirmed |

---
## 12. Immediate Actionable TODO (Sequence)
1. (Python) Implement minimal WS server + log incoming JSON.
2. (JS) Initialize project, connect and send `session.start`, then close.
3. Add protocol schemas & validation on both ends.
4. Stream dummy `audio.chunk` messages; Python prints receipt.
5. Pipe chunks into existing VAD/transcriber path; hardcode user metadata.
6. Emit `transcription.segment` back to Node & log.
7. Hook Node to Discord voice + real audio capture.
8. Implement command forwarding & wrapup cycle.
9. Remove legacy py-cord usage & prune code.
10. Optimize / refactor for cleanliness & docs.

---
## 13. Decommissioning Legacy Code
Files to retire/replace:
- `bot.py` (split into `ws_server.py`, `session_manager.py`, `transcription_service.py`)
- `patched_voice_client.py`, `voice_capture_sink.py` (refactor usable DSP parts into `audio/` utils)
- CLI: add new subcommands: `python-service` (run headless), maybe keep `wrapup`.

---
## 14. Open Questions
- Should we persist per-chunk raw audio for debugging? (Maybe optional flag.)
- Do we want live partial transcription events (stream token-level)? Possibly later via Whisper streaming or incremental chunk decode.
- gRPC vs WS? gRPC gives schema enforcement; can revisit after MVP.

---
## 15. Success Criteria
- Node + Python run concurrently; bot joins a voice channel, produces real-time transcripts with comparable latency/performance to legacy.
- `!wrapup` generates same artifacts as before (outline + transcript + optional gist).
- No py-cord dependency required in default run path.
- Clear developer docs for running both services.

---
(End of Plan)
