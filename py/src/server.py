from __future__ import annotations

import asyncio
import json
import base64
import signal
import websockets
from websockets.legacy.server import WebSocketServerProtocol
from typing import Dict
from pydantic import ValidationError
from . import messages
from .config import load_app_config
from .transcriber import AsyncWhisperTranscriber, TranscribeJob, TranscriptionResult
from .segmenter import PerUserSegmenter, SpeechSegment
from .wrapup import generate_transcript, generate_outline
from .audio import normalize_to_mono16k

class WsServer:
	def __init__(self):
		self.loop = asyncio.get_event_loop()
		self.transcriber = AsyncWhisperTranscriber()
		self.clients: set[WebSocketServerProtocol] = set()
		self.segmenters: Dict[str, PerUserSegmenter] = {}
		self._stopping = False

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
		msg = messages.TranscriptionMessage(
			v=1,
			type='transcription',
			user_id=segment.user_id,
			text=segment.text,
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
				self.segmenters[user_key] = PerUserSegmenter(user_id=msg.user_id)

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

			segments: list[SpeechSegment] = segmenter.feed(pcm16, msg.capture_ts)
			for seg in segments:
				job = TranscribeJob(user_id=msg.user_id, capture_ts=seg.start_ts, pcm16=seg.pcm16)
				await self.transcriber.submit(job)

		elif msg_type == 'summarize.request':
			try:
				msg = messages.SummarizeRequestMessage(**obj)
			except ValidationError as e:
				await self._emit_error('bad_request', f'invalid summarize.request: {e}')
				return
			transcript = generate_transcript(msg.log_entries, msg.session_name)
			outline = generate_outline(msg.log_entries, msg.session_name)
			response = messages.SummarizeResponseMessage(
				v=1,
				type='summarize.response',
				transcript=transcript,
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
				transcriber_started = False
				try:
					await self.transcriber.start()
					transcriber_started = True
					await stop_future
				finally:
					self._stopping = True
					if transcriber_started:
						await self.transcriber.stop()
					await asyncio.gather(*[c.close() for c in list(self.clients)], return_exceptions=True)
					print("Shutdown complete.")
		except OSError as e:
			# Bind failed (address in use or permission denied). Avoid loading model.
			print(f"Failed to bind ws://{host}:{port}: {e}")

def main():
	async def _main():
		server = WsServer()
		await server.start()
	asyncio.run(_main())

if __name__ == '__main__':
	main()
