from __future__ import annotations

import asyncio
from typing import Optional, Callable, Awaitable, Protocol, runtime_checkable
import numpy as np
from dataclasses import dataclass
from transformers import pipeline, WhisperProcessor

from .devices import resolve_device

from . import config

@dataclass
class TranscribeJob:
	user_id: str
	capture_ts: float
	pcm16: bytes  # 16kHz mono little-endian

TARGET_SR = 16000  # expected incoming sample rate


@runtime_checkable
class TranscriberLike(Protocol):
	async def start(self) -> None: ...
	async def stop(self) -> None: ...
	async def submit(self, job: 'TranscribeJob') -> None: ...


# New lightweight result returned from the transcriber; not tied to Pydantic models
@dataclass
class TranscriptionResult:
	user_id: str
	text: str
	# capture_ts and duration omitted intentionally — reconstruction happens upstream if needed


class AsyncWhisperTranscriber:
	def __init__(self, emit_cb: Optional[Callable[[TranscriptionResult], Awaitable[None]]] = None):
		self.emit_cb = emit_cb
		self.queue: asyncio.Queue[TranscribeJob] = asyncio.Queue(maxsize=64)
		self._task: Optional[asyncio.Task] = None
		self._pipe = None
		self._processor = None
		self._prompt_ids = None
		self._config = config.load_app_config()

	def set_emit_callback(self, cb: Callable[[TranscriptionResult], Awaitable[None]]):
		self.emit_cb = cb

	async def start(self):
		if self._task:
			return
		# Lazy load model in executor to avoid blocking loop
		loop = asyncio.get_running_loop()
		def load_pipe():
			# Device selection now handled internally (env: DEVICE) unless explicitly passed
			resolved = resolve_device(self._config.device)
			dev_arg = resolved.pipeline_index
			proc = WhisperProcessor.from_pretrained(self._config.whisper.model)
			self._processor = proc
			return pipeline(
				task='automatic-speech-recognition',
				model=self._config.whisper.model,
				device=dev_arg,
			)
		self._pipe = await loop.run_in_executor(None, load_pipe)
		# Precompute prompt_ids once if prompt configured. Let failures surface.
		app_cfg = config.load_app_config()
		prompt_text = app_cfg.whisper.prompt
		if prompt_text and self._processor:
			self._prompt_ids = self._processor.get_prompt_ids(prompt_text, return_tensors="pt")

		self._task = asyncio.create_task(self._run(), name='whisper-transcriber')

	async def stop(self):
		if self._task:
			self._task.cancel()
			try:
				await self._task
			except asyncio.CancelledError:
				pass
			self._task = None

	async def submit(self, job: TranscribeJob):
		try:
			self.queue.put_nowait(job)
		except asyncio.QueueFull:
			# Drop oldest to make room
			try:
				_ = self.queue.get_nowait()
			except Exception:
				pass
			await self.queue.put(job)

	async def _run(self):
		while True:
			job = await self.queue.get()
			try:
				# Convert bytes to float32 numpy in -1..1 range
				if not job.pcm16:
					return
				audio_np = np.frombuffer(job.pcm16, dtype='<i2').astype('float32') / 32768.0
				if audio_np.size == 0:
					return

				# Build generation kwargs approximating legacy thresholds
				app_cfg = config.load_app_config()
				whisper_cfg = app_cfg.whisper
				generate_kwargs = {
					'temperature': (0.0, 0.1, 0.25, 0.5),
					'logprob_threshold': whisper_cfg.logprob_threshold,
					'no_speech_threshold': whisper_cfg.no_speech_threshold,
					'condition_on_prev_tokens': True,
					'compression_ratio_threshold': 1.35,
					'forced_decoder_ids': None,
				}
				# Attach precomputed prompt ids once (first segment) if available
				if self._prompt_ids is not None:
					generate_kwargs['prompt_ids'] = self._prompt_ids
					generate_kwargs['prompt_condition_type'] = 'first-segment'

				# Force english if model isn't .en variant like legacy
				model_name_lower = self._config.whisper.model.lower()
				if not model_name_lower.endswith('.en'):
					generate_kwargs['language'] = 'english'
					generate_kwargs['task'] = 'transcribe'

				# Run pipeline (blocking) in executor
				loop = asyncio.get_running_loop()
				def infer():
					if self._pipe is None:
						return None
					return self._pipe(
						audio_np,
						return_timestamps=False,
						generate_kwargs=generate_kwargs,
						chunk_length_s=30,
					)
				result = await loop.run_in_executor(None, infer)
				if not result:
					return
				# HF pipeline may return dict or list with segments; normalize
				if isinstance(result, list):
					# take concatenated text fields
					texts = [seg.get('text','') for seg in result if isinstance(seg, dict)]
					text = ' '.join(t.strip() for t in texts).strip()
				elif isinstance(result, dict):
					text = result.get('text','').strip()
				else:
					text = ''
				if text:
					seg = TranscriptionResult(
						user_id=job.user_id,
						text=text,
					)
					if self.emit_cb:
						await self.emit_cb(seg)
			finally:
				self.queue.task_done()
