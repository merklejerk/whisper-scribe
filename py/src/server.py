from __future__ import annotations

import asyncio
import json
import base64
import signal
import websockets
from websockets.legacy.server import WebSocketServerProtocol
import contextlib
import uuid
import typing
from dataclasses import dataclass
from pydantic import ValidationError
import numpy as np
from . import messages
from .config import load_app_config
from .transcriber import AsyncWhisperTranscriber, TranscribeJob, TranscriptionResult
from .wrapup import generate_transcript, GeminiWrapupGenerator, LLMRequestError
from .audio import normalize_to_mono16k, enhance_speech, TARGET_SR

MAX_MSG_SIZE = 10 * 1024 * 1024

class WsServer:
	loop: asyncio.AbstractEventLoop
	transcriber: AsyncWhisperTranscriber
	clients: set[WebSocketServerProtocol]
	_user_prompt: dict[str, typing.Optional[str]]
	_job_meta: dict[str, JobMeta]
	_stopping: bool

	def __init__(self):
		self.loop = asyncio.get_event_loop()
		# Base config; callers may pass per-request overrides in messages
		app_cfg = load_app_config()
		whisper_cfg = AsyncWhisperTranscriber.WhisperRuntimeCfg(
			device=app_cfg.device,
			model=app_cfg.whisper.model,
			logprob_threshold=app_cfg.whisper.logprob_threshold,
			no_speech_threshold=app_cfg.whisper.no_speech_threshold,
		)
		self.transcriber = AsyncWhisperTranscriber(whisper_cfg)
		self.transcriber.set_emit_callback(self.handle_transcription_segment)
		self.clients = set()
		# Track the most recent per-user prompt provided by the client
		self._user_prompt = {}
		# Correlation metadata for in-flight transcription jobs
		self._job_meta = {}
		self._stopping = False
		# No internal segmenters; Node performs VAD and sends complete segments


	async def emit(self, msg: messages.Outgoing):
		data = msg.model_dump()
		payload = json.dumps(data)
		dead = []
		for ws in self.clients:
			try:
				await ws.send(payload)
			except Exception:
				dead.append(ws)
		for d in dead:
			self.clients.discard(d)

	async def handle_transcription_segment(self, segment: TranscriptionResult):
		"""Callback for when the transcriber finishes a segment."""
		meta = self._job_meta.pop(segment.id, None)
		if meta is None:
			# Required correlation metadata missing: fail fast (internal consistency error)
			raise KeyError(f"Missing job metadata for transcription id={segment.id}")
		msg = messages.TranscriptionMessage(
			v=1,
			type='transcription',
			id=meta.id,
			text=segment.text,
			capture_ts=float(meta.capture_ts),
			end_ts=float(meta.end_ts),
		)
		await self.emit(msg)

	async def handle_incoming(self, raw: str):
		try:
			obj = json.loads(raw)
		except json.JSONDecodeError as e:
			await self._emit_error('bad_json', f'JSON parse error: {e}')
			return

		if not isinstance(obj, dict):
			await self._emit_error('bad_request', 'payload must be a JSON object')
			return

		msg_type = obj.get('type')

		if msg_type == 'audio.segment':
			try:
				msg = messages.AudioSegmentMessage(**obj)
			except ValidationError as e:
				await self._emit_error('bad_request', f'invalid audio.segment: {e}')
				return

			# Store last-seen prompt for this user; Node may override per segment
			self._user_prompt[msg.id] = getattr(msg, 'prompt', None)

			# Decode/normalize audio to mono16k
			fmt = msg.pcm_format
			try:
				pcm16 = normalize_to_mono16k(
					base64.b64decode(msg.data_b64),
					sr=fmt.sr,
					channels=fmt.channels,
					sample_width=fmt.sample_width,
				)
			except ValueError as e:
				await self._emit_error('bad_audio_format', str(e))
				return

			# Enhance audio and submit as a single ASR job
			audio_float32 = np.frombuffer(pcm16, dtype='<i2').astype('float32') / 32768.0
			enhanced_float32 = enhance_speech(audio_float32, TARGET_SR)
			enhanced_pcm16 = (enhanced_float32 * 32767.0).astype('<i2').tobytes()

			job_id = uuid.uuid4().hex
			# Capture both start and end timestamps from Node-provided values
			self._job_meta[job_id] = JobMeta(
				id=msg.id,
				capture_ts=float(msg.started_ts),
				end_ts=float(msg.capture_ts),
			)
			prompt = self._user_prompt.get(msg.id)
			job = TranscribeJob(id=job_id, pcm16=enhanced_pcm16, prompt=prompt)
			await self.transcriber.submit(job)

		elif msg_type == 'wrapup.request':
			try:
				msg = messages.WrapupRequestMessage(**obj)
			except ValidationError as e:
				await self._emit_error('bad_request', f'invalid wrapup.request: {e}')
				return
			base_cfg = load_app_config()
			if base_cfg is None:
				await self._emit_error('server_config', 'No configuration available')
				return
			if base_cfg.gemini_api_key is None:
				await self._emit_error('missing_api_key', 'Gemini API key is not configured')
				return

			transcript = generate_transcript(
				log_entries=msg.log_entries,
				session_name=msg.session_name,
			)
			# Use Gemini-based wrapup generator; pass configured API key
			generator = GeminiWrapupGenerator(
				base_cfg.gemini_api_key,
				base_cfg.wrapup.model,
				prompt=(getattr(msg, 'wrapup_prompt', None) or base_cfg.wrapup.prompt),
				temperature=base_cfg.wrapup.temperature,
				max_output_tokens=base_cfg.wrapup.max_output_tokens,
			)
			try:
				tips = getattr(msg, 'wrapup_tips', None)
				outline = await generator.generate(transcript, tips=(list(tips) if tips is not None else list(base_cfg.wrapup.tips)))
			except LLMRequestError as e:
				# Preserve the original error code/message in the WS error reply
				code = str(e.code) if hasattr(e, 'code') else 'llm_error'
				message = e.message if hasattr(e, 'message') else str(e)
				await self._emit_error(code, message)
				return
			else:
				# Ensure outline is a string for the response model
				if outline is None:
					outline = ""
				response = messages.WrapupResponseMessage(
					v=1,
					type='wrapup.response',
					outline=outline,
					request_id=msg.request_id,
				)
				await self.emit(response)

		else:
			await self._emit_error('unknown_type', f"unknown message type: {msg_type!r}")

	async def _handler(self, ws):
		self.clients.add(ws)
		try:
			async for msg in ws:
				if isinstance(msg, str):
					await self.handle_incoming(msg)
				else:
					await self._emit_error('unsupported_frame', 'binary frames are not supported')
		finally:
			self.clients.discard(ws)
	async def _emit_error(self, code: str, message: str):
		err = messages.ErrorMessage(code=code, message=message)
		await self.emit(err)

	async def start(self):
		stop_future = self.loop.create_future()

		def _signal(sig):
			if not stop_future.done():
				print(f"Received signal {sig.name}, shutting down...")
				stop_future.set_result(True)

		for s in (signal.SIGINT, signal.SIGTERM):
			try:
				self.loop.add_signal_handler(s, lambda sig=s: _signal(sig))
			except NotImplementedError:
				pass

		cfg = load_app_config()
		host = cfg.net.host
		port = cfg.net.port

		# Do not start heavy components (model load) until socket bind succeeds.
		try:
			async with websockets.serve(self._handler, host, port, max_size=MAX_MSG_SIZE):
				print(f"WS server listening on ws://{host}:{port}")
				# Configure fatal propagation for the transcriber
				def _fatal_handler(exc: BaseException):
					if not stop_future.done():
						stop_future.set_exception(exc)
				self.transcriber.set_on_fatal(_fatal_handler)
				# Start ASR worker now that bind succeeded
				await self.transcriber.start()
				await stop_future
				# Shutdown sequence
				self._stopping = True
				# Stop transcriber
				with contextlib.suppress(Exception):
					await self.transcriber.stop()
				await asyncio.gather(*[c.close() for c in list(self.clients)], return_exceptions=True)
				print("Shutdown complete.")
		except OSError as e:
			# Bind failed (address in use or permission denied). Avoid loading model.
			print(f"Failed to bind ws://{host}:{port}: {e}")


# Typed job metadata for correlation and timestamps
@dataclass
class JobMeta:
	id: str
	capture_ts: float
	end_ts: float

def main():
	async def _main():
		server = WsServer()
		await server.start()
	asyncio.run(_main())

if __name__ == '__main__':
	main()
