from __future__ import annotations
"""Per-user speech segmentation using Silero VAD.

- Input: mono 16 kHz PCM16 chunks with capture_ts.
- VAD: Silero 32 ms frames; evaluated once we buffer at least `vad_window_s`.
- Emit: when a silence gap or max length is reached; segments shorter than
	`min_segment_s` are always dropped.
- Periodic processing supported via `collect_ready(now_ts)` (no new audio needed).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Deque, Tuple
from collections import deque
import numpy as np

from .devices import resolve_device
from .debug import make_debug_logger
from .vad import SileroVAD, SAMPLE_RATE, FRAME_MS


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
	# Pending chunks to be integrated by collect_ready: (arr, capture_ts)
	pending: Deque[Tuple[np.ndarray, float]] = field(default_factory=deque)

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
				 silence_gap_s: float = 1.5,
				 max_segment_s: float = 60.0,
				 vad_threshold: float = 0.5,
				 min_segment_s: float = 0.2):
		self.user_id = user_id
		self.silence_gap_s = silence_gap_s
		self.max_segment_s = max_segment_s
		self.vad_threshold = vad_threshold
		self.min_segment_s = min_segment_s
		# how much audio to accumulate (seconds) before running VAD
		self.vad_window_s = 0.2
		# Enforce window covers at least one VAD frame (32 ms)
		frame_s = FRAME_MS / 1000.0
		if self.vad_window_s < frame_s:
			raise ValueError(f"vad_window_s ({self.vad_window_s}) must be >= one VAD frame ({frame_s}s)")
		self.buf = _BufferState()
		# Device selected for logging only; SileroVAD resolves internally by default.
		self.device = resolve_device().name
		self._vad = SileroVAD.create(self.device)
		# Enforce policy: segments must be at least one VAD window long
		if self.min_segment_s < self.vad_window_s:
			raise ValueError(f"min_segment_s ({self.min_segment_s}) must be >= vad_window_s ({self.vad_window_s})")
		# Shared debug logger (noop unless APP_DEBUG=1)
		self._dbg = make_debug_logger('segmenter')
		# Internal ready-queue of completed segments (drained by collect_ready)
		self._ready: Deque[SpeechSegment] = deque()

	def _total_samples(self) -> int:
		"""Return total samples currently buffered."""
		return sum(a.shape[0] for a in self.buf.samples)

	def _vad_trim_and_detect(self, arr: np.ndarray) -> tuple[bool, np.ndarray]:
		"""Trim leading silence via VAD and return (is_speech, trimmed_array).

		- Uses VAD to suggest a safe leading-silence trim.
		- Then evaluates speech probability over the last window of the trimmed buffer.
		- Precondition: caller only invokes this after buffering at least
		  `vad_window_s` of audio, so a full 32 ms frame is always available.
		"""
		pcm16 = arr.astype('<i2', copy=False)
		drop, max_p = self._vad.analyze(pcm16, self.vad_window_s, self.vad_threshold)
		trimmed = pcm16[drop:] if drop > 0 else pcm16
		return (max_p >= self.vad_threshold, trimmed)

	def feed(self, pcm16: bytes, capture_ts: float) -> None:
		"""Queue a chunk; processing is deferred to collect_ready.

	We only decode bytes to int16 and enqueue (arr, capture_ts). All gap
	handling, VAD decisions, and finalization happen in collect_ready.
		"""
		self._dbg(f"feed chunk bytes={len(pcm16)} capture_ts={capture_ts:.3f}")
		arr = np.frombuffer(pcm16, dtype='<i2')
		if arr.size > 0:
			self.buf.pending.append((arr, capture_ts))
		return None

	def get_buffer_details(self) -> tuple[int, float | None]:
		"""Returns (chunk_count, last_capture_ts)."""
		return (len(self.buf.samples), self.buf.last_capture_ts)

	# --- Internal helpers to keep collect_ready focused and readable ---
	def _flush_and_queue(self, end_ts: float) -> None:
		"""Flush current buffer and queue resulting segment(s) if any."""
		_flushed = self._flush(end_ts=end_ts)
		if _flushed:
			self._ready.extend(_flushed)

	def _check_discontinuity_and_maybe_flush(self, arr: np.ndarray, capture_ts: float) -> None:
		"""If the incoming chunk starts after a long gap, flush the current segment first."""
		if self.buf.samples and self.buf.last_capture_ts is not None:
			incoming_dur = arr.size / float(SAMPLE_RATE)
			incoming_start_ts = capture_ts - incoming_dur
			buffer_end_ts = self.buf.last_capture_ts
			gap = incoming_start_ts - buffer_end_ts
			if gap >= self.silence_gap_s:
				self._dbg(
					f"incoming gap start={incoming_start_ts:.3f} buffer_end={buffer_end_ts:.3f} "
					f"gap={gap:.3f}s >= {self.silence_gap_s}s -> flush before append"
				)
				end_ts = self.buf.last_speech_ts if self.buf.last_speech_ts is not None else self.buf.last_capture_ts
				self._flush_and_queue(end_ts=end_ts)

	def _process_vad_for_buffer(self, capture_ts: float) -> None:
		"""Run VAD for the current buffer and perform segmentation decisions."""
		total_samples = self._total_samples()
		if total_samples < int(self.vad_window_s * SAMPLE_RATE):
			return
		concat = np.concatenate(self.buf.samples)
		is_speech, trimmed = self._vad_trim_and_detect(concat)
		# Trim leading silence only before a segment starts
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
			# Enforce max segment length
			if self.buf.started_ts and (capture_ts - self.buf.started_ts) >= self.max_segment_s:
				self._dbg(f"max_segment reached duration={capture_ts - self.buf.started_ts:.2f}s -> flush")
				self._flush_and_queue(end_ts=capture_ts)
		else:
			gap = None
			if self.buf.last_speech_ts is not None:
				gap = capture_ts - self.buf.last_speech_ts
			if self.buf.last_speech_ts is not None and gap is not None and gap >= self.silence_gap_s:
				self._dbg(f"silence gap={gap:.2f}s >= {self.silence_gap_s}s -> flush")
				self._flush_and_queue(end_ts=self.buf.last_speech_ts)

	def _time_based_finalization(self, now_ts: float) -> None:
		"""Finalize segments based on time with no new chunks arriving."""
		if self.buf.samples and self.buf.started_ts is not None and self.buf.last_speech_ts is not None:
			gap = now_ts - self.buf.last_speech_ts
			if gap >= self.silence_gap_s:
				self._dbg(f"silence gap elapsed gap={gap:.2f}s >= {self.silence_gap_s}s -> flush")
				self._flush_and_queue(end_ts=self.buf.last_speech_ts)

		# Enforce max segment length even without new chunks
		if self.buf.samples and self.buf.started_ts is not None and (now_ts - self.buf.started_ts) >= self.max_segment_s:
			end_ts = self.buf.last_capture_ts if self.buf.last_capture_ts is not None else self.buf.started_ts
			self._dbg(f"max_segment elapsed duration={now_ts - self.buf.started_ts:.2f}s -> flush")
			self._flush_and_queue(end_ts=end_ts)

	def _drain_ready(self) -> List[SpeechSegment]:
		"""Drain internal ready queue into a list."""
		ready: List[SpeechSegment] = []
		while self._ready:
			ready.append(self._ready.popleft())
		return ready

	def collect_ready(self, now_ts: float) -> List[SpeechSegment]:
		"""Process pending chunks, apply time-based finalization, and return segments.

		- If `silence_gap_s` has elapsed since the last detected speech, flush
		  using `last_speech_ts` as the end timestamp.
		- If `max_segment_s` is exceeded without new chunks, flush using
		  `last_capture_ts` (or `started_ts` as fallback) as the end timestamp.
		- Otherwise, returns an empty list.

		Intended to be called periodically by the orchestrator.
		"""
		# 1) Integrate any pending chunks, handling gaps and VAD decisions
		while self.buf.pending:
			arr, capture_ts = self.buf.pending.popleft()
			self._check_discontinuity_and_maybe_flush(arr, capture_ts)
			# Append the chunk
			self.buf.samples.append(arr)
			self.buf.last_capture_ts = capture_ts
			# VAD and segmentation updates
			self._process_vad_for_buffer(capture_ts)

		# 2) Time-based finalization with no new chunks
		self._time_based_finalization(now_ts)

		# 3) Drain and return any ready segments
		return self._drain_ready()

	def _flush(self, end_ts: float) -> List[SpeechSegment]:
		if not self.buf.samples or self.buf.started_ts is None:
			self._dbg("flush called with empty buffer - no segment")
			self.buf = _BufferState()
			return []
		samples_concat = np.concatenate(self.buf.samples)
		duration_s = samples_concat.shape[0] / SAMPLE_RATE
		if duration_s < self.min_segment_s:
			# discard tiny blip
			self._dbg(f"discarding short segment duration={duration_s:.3f}s (< {self.min_segment_s}s)")
			self.buf = _BufferState()
			return []
		pcm16 = samples_concat.astype('<i2').tobytes()
		segment = SpeechSegment(pcm16=pcm16, start_ts=self.buf.started_ts, end_ts=end_ts)
		self._dbg(f"emitting segment duration={duration_s:.3f}s bytes={len(pcm16)} start={segment.start_ts:.3f} end={segment.end_ts:.3f}")
		self.buf = _BufferState()
		return [segment]
