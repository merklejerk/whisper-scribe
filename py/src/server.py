from __future__ import annotations

import asyncio
import json
import base64
import signal
import websockets
from websockets.legacy.server import WebSocketServerProtocol
import time
import contextlib
import uuid
from typing import Dict, Optional
from dataclasses import dataclass
from pydantic import ValidationError
import numpy as np
from . import messages
from .config import load_app_config
from .transcriber import AsyncWhisperTranscriber, TranscribeJob, TranscriptionResult
from .segmenter import PerUserSegmenter
from .wrapup import generate_transcript, GeminiWrapupGenerator, LLMRequestError
from .audio import normalize_to_mono16k, enhance_speech, TARGET_SR

class WsServer:
	def __init__(self):
		self.loop = asyncio.get_event_loop()
		# Load global config once and refine per component
		app_cfg = load_app_config()
		# Keep a reference to the loaded app config to avoid reloading
		self.app_cfg = app_cfg
		whisper_cfg = AsyncWhisperTranscriber.WhisperRuntimeCfg(
			device=app_cfg.device,
			model=app_cfg.whisper.model,
			logprob_threshold=app_cfg.whisper.logprob_threshold,
			no_speech_threshold=app_cfg.whisper.no_speech_threshold,
			prompt=app_cfg.whisper.prompt,
		)
		self.transcriber = AsyncWhisperTranscriber(whisper_cfg)
		self.clients: set[WebSocketServerProtocol] = set()
		self.segmenters: Dict[str, PerUserSegmenter] = {}
		# Correlation metadata for in-flight transcription jobs
		self._job_meta: Dict[str, JobMeta] = {}
		self._stopping = False
		self._flusher_task: asyncio.Task | None = None
		# Flusher wake-up event (nudge on new chunks)
		self._flush_nudge: asyncio.Event = asyncio.Event()

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
			user_id=meta.user_id,
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

		if msg_type == 'audio.chunk':
			try:
				msg = messages.AudioChunkMessage(**obj)
			except ValidationError as e:
				await self._emit_error('bad_request', f'invalid audio.chunk: {e}')
				return

			user_key = msg.user_id
			if user_key not in self.segmenters:
				cfg = self.app_cfg
				self.segmenters[user_key] = PerUserSegmenter(
					user_id=msg.user_id,
					silence_gap_s=cfg.voice.silence_threshold_seconds,
					vad_threshold=cfg.voice.vad_threshold,
					max_segment_s=max(1.0, float(cfg.voice.max_speech_buf_seconds) or 60.0),
				)

			segmenter = self.segmenters[user_key]
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

			# Queue chunk; flusher will process and submit
			segmenter.feed(pcm16, msg.capture_ts)
			# Nudge flusher so inactivity-based finalization runs promptly
			self._flush_nudge.set()

		elif msg_type == 'wrapup.request':
			try:
				msg = messages.WrapupRequestMessage(**obj)
			except ValidationError as e:
				await self._emit_error('bad_request', f'invalid wrapup.request: {e}')
				return
			cfg = self.app_cfg
			if cfg.gemini_api_key is None:
				await self._emit_error('missing_api_key', 'Gemini API key is not configured')
				return

			transcript = generate_transcript(msg.log_entries, msg.session_name)
			# Use Gemini-based wrapup generator; pass configured API key
			generator = GeminiWrapupGenerator(
				cfg.gemini_api_key,
				cfg.wrapup.model,
				prompt=cfg.wrapup.prompt,
				temperature=cfg.wrapup.temperature,
				max_output_tokens=cfg.wrapup.max_output_tokens,
			)
			try:
				outline = await generator.generate(transcript, tips=list(cfg.wrapup.tips))
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
			async with websockets.serve(self._handler, host, port):
				print(f"WS server listening on ws://{host}:{port}")
				# Start transcriber after successful bind
				self.transcriber.set_emit_callback(self.handle_transcription_segment)
				# If the transcriber encounters a fatal error, halt the server by setting the stop future's exception
				def _fatal_handler(exc: BaseException):
					if not stop_future.done():
						stop_future.set_exception(exc)
				self.transcriber.set_on_fatal(_fatal_handler)
				transcriber_started = False
				try:
					await self.transcriber.start()
					transcriber_started = True
					# Start periodic flusher to finalize segments on inactivity
					self._flusher_task = asyncio.create_task(self._run_flusher(stop_future), name="segmenter-flusher")
					# Fail-fast: if the flusher task crashes, propagate its exception to stop the server
					def _flusher_done(task: asyncio.Task):
						if task.cancelled():
							return
						exc = task.exception()
						if exc and not stop_future.done():
							stop_future.set_exception(exc)
					self._flusher_task.add_done_callback(_flusher_done)
					await stop_future
				finally:
					self._stopping = True
					if transcriber_started:
						await self.transcriber.stop()
					if self._flusher_task is not None:
						self._flusher_task.cancel()
						with contextlib.suppress(asyncio.CancelledError):
							await self._flusher_task
					await asyncio.gather(*[c.close() for c in list(self.clients)], return_exceptions=True)
					print("Shutdown complete.")
		except OSError as e:
			# Bind failed (address in use or permission denied). Avoid loading model.
			print(f"Failed to bind ws://{host}:{port}: {e}")

	async def _run_flusher(self, stop_future: asyncio.Future):
		"""Submit segments in one place, driven by collect_ready.

		Wakes on either a fixed interval (250 ms) or an explicit nudge when new
		audio arrives. PerUserSegmenter manages ready segments; collect_ready returns them.
		"""
		interval = 0.25
		while not stop_future.done():
			# Wait for either a nudge or the interval to elapse
			try:
				await asyncio.wait_for(self._flush_nudge.wait(), timeout=interval)
			except asyncio.TimeoutError:
				pass
			self._flush_nudge.clear()

			now = time.time()
			# Process pending audio and finalize segments; drains ready queue
			for user_id, seg in list(self.segmenters.items()):
				segments = seg.collect_ready(now)
				for s in segments:
					# Create a correlation id and store metadata for the outgoing message
					job_id = uuid.uuid4().hex
					self._job_meta[job_id] = JobMeta(user_id=user_id, capture_ts=s.start_ts, end_ts=s.end_ts)
					
					# Enhance audio before submitting
					audio_float32 = np.frombuffer(s.pcm16, dtype='<i2').astype('float32') / 32768.0
					enhanced_float32 = enhance_speech(audio_float32, TARGET_SR)
					enhanced_pcm16 = (enhanced_float32 * 32767.0).astype('<i2').tobytes()

					job = TranscribeJob(id=job_id, pcm16=enhanced_pcm16)
					await self.transcriber.submit(job)

# Typed job metadata for correlation and timestamps
@dataclass
class JobMeta:
	user_id: str
	capture_ts: float
	end_ts: float

def main():
	async def _main():
		server = WsServer()
		await server.start()
	asyncio.run(_main())

if __name__ == '__main__':
	main()
