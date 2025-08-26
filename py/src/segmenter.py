from __future__ import annotations
"""Per-user speech segmentation with optional Silero VAD.

Workflow:
- feed(raw_pcm16, capture_ts) for each incoming mono 16k PCM chunk (bytes)
- internally evaluates VAD over 32 ms frames (512 samples @ 16 kHz)
- accumulate speech frames until silence gap exceeded or max length reached
- finalize() returns list of (pcm_bytes, started_ts, last_capture_ts)
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .devices import resolve_device
from .debug import make_debug_logger
from .vad import SileroVAD, SAMPLE_RATE, InsufficientSamplesError


@dataclass
class SpeechSegment:
	pcm16: bytes
	start_ts: float
	end_ts: float

@dataclass
class _BufferState:
	started_ts: float | None = None
	last_speech_ts: float | None = None
	last_capture_ts: float | None = None
	samples: List[np.ndarray] = field(default_factory=list)

	def trim_prefix(self, n_samples: int) -> None:
		"""Drop the first n_samples across concatenated sample chunks in-place."""
		if n_samples <= 0 or not self.samples:
			return
		remaining = n_samples
		new_samples: List[np.ndarray] = []
		for chunk in self.samples:
			if remaining <= 0:
				new_samples.append(chunk)
				continue
			if chunk.shape[0] <= remaining:
				remaining -= chunk.shape[0]
				continue
			# trim within this chunk
			new_samples.append(chunk[remaining:])
			remaining = 0
		self.samples = new_samples

class PerUserSegmenter:
	def __init__(self,
				 user_id: str,
				 silence_gap_s: float = 0.8,
				 max_segment_s: float = 25.0,
				 vad_threshold: float = 0.5,
				 min_segment_s: float = 0.2,
				 energy_floor: float = 0.0005):
		self.user_id = user_id
		self.silence_gap_s = silence_gap_s
		self.max_segment_s = max_segment_s
		self.vad_threshold = vad_threshold
		self.min_segment_s = min_segment_s
		self.energy_floor = energy_floor
		# how much audio to accumulate (seconds) before running VAD
		self.vad_window_s = 0.2
		self.buf = _BufferState()
		# Device selected for logging only; SileroVAD resolves internally by default.
		self.device = resolve_device().name
		try:
			self._vad = SileroVAD.create(self.device)
		except Exception as e:
			raise RuntimeError(f"Failed to load Silero VAD: {e}")
		# Shared debug logger (noop unless APP_DEBUG=1)
		self._dbg = make_debug_logger('segmenter')

	def _total_samples(self) -> int:
		"""Return total samples currently buffered."""
		return sum(a.shape[0] for a in self.buf.samples)

	def _vad_trim_and_detect(self, arr: np.ndarray) -> tuple[bool, np.ndarray]:
		"""Trim leading silence via VAD and return (is_speech, trimmed_array).

		- Uses VAD to suggest a safe leading-silence trim.
		- Then evaluates speech probability over the last window of the trimmed buffer.
		- If the slice is too short for one frame, returns (False, trimmed).
		"""
		pcm16 = arr.astype('<i2', copy=False)
		try:
			# Unified call: get drop suggestion and tail window probability
			drop, max_p = self._vad.analyze(pcm16, SAMPLE_RATE, self.vad_window_s, self.vad_threshold)
		except InsufficientSamplesError:
			# Not enough audio to form a full frame in the tail window; keep buffering
			return (False, pcm16)
		except Exception:
			# If validation fails for any other reason, avoid trimming and decide no speech
			return (False, pcm16)
		trimmed = pcm16[drop:] if drop > 0 else pcm16
		return (max_p >= self.vad_threshold, trimmed)

	def feed(self, pcm16: bytes, capture_ts: float) -> List[SpeechSegment]:
		finalized: List[SpeechSegment] = []
		# call _dbg unconditionally; it's a noop when disabled
		self._dbg(f"feed chunk bytes={len(pcm16)} capture_ts={capture_ts:.3f}")
		# Append incoming chunk as a single array. We'll run VAD once we have
		# at least `vad_window_s` worth of audio buffered.
		arr = np.frombuffer(pcm16, dtype='<i2')

		# If there's a discontinuity between the incoming chunk and the
		# buffered audio, flush the current buffer first. Compare the
		# incoming chunk's start timestamp to the buffer's end timestamp so
		# we don't merge audio separated by a gap.
		if self.buf.samples and self.buf.last_capture_ts is not None:
			# incoming chunk duration in seconds
			incoming_dur = arr.size / float(SAMPLE_RATE)
			incoming_start_ts = capture_ts - incoming_dur
			# buffer end timestamp: use last_capture_ts (which is the capture_ts
			# recorded when the last chunk was appended)
			buffer_end_ts = self.buf.last_capture_ts
			gap = incoming_start_ts - buffer_end_ts
			if gap >= self.silence_gap_s:
				self._dbg(f"incoming gap start={incoming_start_ts:.3f} buffer_end={buffer_end_ts:.3f} gap={gap:.3f}s >= {self.silence_gap_s}s -> flush before append")
				# prefer last_speech_ts for end timestamp, fall back to last_capture_ts
				end_ts = self.buf.last_speech_ts if self.buf.last_speech_ts is not None else self.buf.last_capture_ts
				finalized.extend(self._flush(force=False, end_ts=end_ts))

		# If there's no audio in the incoming chunk, return finalized segments
		if arr.size == 0:
			return finalized

		# Append chunk.
		self.buf.samples.append(arr)
		self.buf.last_capture_ts = capture_ts

		# Send to VAD and flush if we have enough samples.
		total_samples = self._total_samples()
		if total_samples >= int(self.vad_window_s * SAMPLE_RATE):
			concat = np.concatenate(self.buf.samples)
			is_speech, trimmed = self._vad_trim_and_detect(concat)
			# Apply the trim to the live buffer only before we've started a segment
			if self.buf.started_ts is None and trimmed is not concat:
				drop = concat.shape[0] - trimmed.shape[0]
				if drop > 0:
					self._dbg(f"trimming leading silence samples={drop}")
					self.buf.trim_prefix(drop)
					concat = trimmed
			if is_speech:
				if self.buf.started_ts is None:
					self._dbg(f"speech start ts={capture_ts:.3f}")
					self.buf.started_ts = capture_ts
				self.buf.last_speech_ts = capture_ts
				# Max length guard
				if self.buf.started_ts and (capture_ts - self.buf.started_ts) >= self.max_segment_s:
					self._dbg(f"max_segment reached duration={capture_ts - self.buf.started_ts:.2f}s -> flush(force)")
					finalized.extend(self._flush(force=True, end_ts=capture_ts))
			else:
				gap = None
				if self.buf.last_speech_ts is not None:
					gap = capture_ts - self.buf.last_speech_ts
				if self.buf.last_speech_ts is not None and gap is not None and gap >= self.silence_gap_s:
					self._dbg(f"silence gap={gap:.2f}s >= {self.silence_gap_s}s -> flush")
					finalized.extend(self._flush(force=False, end_ts=self.buf.last_speech_ts))
		return finalized

	def flush(self, force: bool = False) -> List[SpeechSegment]:
		if self.buf.samples and self.buf.last_capture_ts:
			return self._flush(force=force, end_ts=self.buf.last_capture_ts)
		return []

	def get_buffer_details(self) -> tuple[int, float | None]:
		"""Returns (sample_count, last_capture_ts)"""
		return (len(self.buf.samples), self.buf.last_capture_ts)

	def _flush(self, force: bool, end_ts: float) -> List[SpeechSegment]:
		if not self.buf.samples or self.buf.started_ts is None:
			self._dbg("flush called with empty buffer - no segment")
			self.buf = _BufferState()
			return []
		samples_concat = np.concatenate(self.buf.samples)
		duration_s = samples_concat.shape[0] / SAMPLE_RATE
		if not force and duration_s < self.min_segment_s:
			# discard tiny blip
			self._dbg(f"discarding short segment duration={duration_s:.3f}s (< {self.min_segment_s}s)")
			self.buf = _BufferState()
			return []
		pcm16 = samples_concat.astype('<i2').tobytes()
		segment = SpeechSegment(pcm16=pcm16, start_ts=self.buf.started_ts, end_ts=end_ts)
		self._dbg(f"emitting segment duration={duration_s:.3f}s bytes={len(pcm16)} start={segment.start_ts:.3f} end={segment.end_ts:.3f} force={force}")
		self.buf = _BufferState()
		return [segment]
