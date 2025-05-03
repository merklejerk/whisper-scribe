import discord
import numpy as np
from collections import defaultdict
import datetime
import asyncio
from typing import Optional, DefaultDict, Tuple
from typing import Optional, DefaultDict, Tuple
import asyncio
from silero_vad import load_silero_vad, get_speech_timestamps
from dataclasses import dataclass, field
import threading

from .config import SILENCE_THRESHOLD_SECONDS, VAD_THRESHOLD, MAX_SPEECH_BUF_SECONDS
from .waveform_utils import pcm16_to_norm_waveform, has_sound

# Module-level audio constants
SOURCE_SR = 48000  # Discord's PCM sample rate
TARGET_SR = 16000  # Required sample rate for VAD and model
BYTES_PER_SAMPLE = 4 # float32

# VAD constants
MIN_DURATION_FOR_VAD_SECONDS = 0.750
MIN_LENGTH_FOR_VAD = int(TARGET_SR * MIN_DURATION_FOR_VAD_SECONDS)

# raw-buffer constants for custom VAD pre-buffering
RAW_BUFFER_DURATION = 1.0  # seconds
RAW_BUFFER_MAX_LENGTH = int(TARGET_SR * RAW_BUFFER_DURATION)

PRUNE_THRESHOLD_SECONDS = 5.

if RAW_BUFFER_DURATION < MIN_DURATION_FOR_VAD_SECONDS:
    raise ValueError(f"RAW_BUFFER_DURATION ({RAW_BUFFER_DURATION}) must be greater than MIN_DURATION_FOR_VAD_SECONDS ({MIN_DURATION_FOR_VAD_SECONDS})")

@dataclass
class VoiceMetadata:
    """Metadata for audio processing jobs."""
    user_id: int
    user_name: str
    channel_id: int
    capture_time: datetime.datetime

    def __repr__(self) -> str:
        """String representation of VoiceMetadata."""
        return f"VoiceMetadata(user_name={self.user_name}, capture_time={self.capture_time.isoformat()}, ...)"

class AudioBuffer(object):
    array: np.ndarray
    pos: int

    def __init__(self):
        self.array = np.zeros((0,), dtype=np.float32)
        self.pos = 0
    
    def get_contents_slice(self) -> np.ndarray:
        """Get the contents of the buffer."""
        return self.array[:self.pos]
    
    def write(self, data: np.ndarray) -> None:
        """Write data to the buffer, expanding it if necessary."""
        end = self.pos + len(data)
        if end > len(self.array):
            new_size = max(len(self.array) * 2, end)
            self.expand(new_size - len(self.array))
        self.array[self.pos:end] = data
        self.pos += len(data)
    
    def expand(self, n: int) -> None:
        """Expand the size of the buffer."""
        if n > 0:
            self.array.resize((len(self.array) + n,))
    
    def shift(self, n: int) -> int:
        """Shift the buffer contents to the left by n items."""
        if n > 0:
            self.array[:self.pos - n] = self.array[n:self.pos]
            self.pos -= n
        return self.pos
    
@dataclass
class UserState:
    speech_buf: AudioBuffer
    raw_buf: AudioBuffer
    first_spoke: Optional[datetime.datetime] = None
    last_spoke: Optional[datetime.datetime] = None
    last_noise: Optional[datetime.datetime] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

class VoiceCaptureSink(discord.sinks.Sink):
    """A sink that detects segments of voice activity and submits it to a queue."""
    voice_client: Optional[discord.VoiceClient]
    capture_queue: asyncio.Queue[Tuple[VoiceMetadata, bytes]]  # Queue for buffering audio to bot
    user_states: DefaultDict[int, UserState]
    cached_user_names: dict[int, str]
    _heartbeat_task: Optional[asyncio.Task]
    _get_speech_timestamps: callable # Silero VAD utility function

    def __init__(self, capture_queue: asyncio.Queue[Tuple[VoiceMetadata, bytes]]):
        super().__init__()
        self.voice_client = None
        self.capture_queue = capture_queue
        # per-user audio state
        self.user_states: DefaultDict[int, UserState] = defaultdict(
            lambda: UserState(
                speech_buf=AudioBuffer(),
                raw_buf=AudioBuffer(),
            )
        )
        self._cached_user_names = {}
        vad_model = load_silero_vad()
        self._get_speech_timestamps = lambda audio: \
            get_speech_timestamps(audio, model=vad_model, threshold=VAD_THRESHOLD, min_speech_duration_ms=250)
        self._heartbeat_task = None

    def _has_voice(self, norm_audio: np.ndarray) -> bool:
        """Check if normalized float32 audio contains speech using Silero VAD"""
        if len(norm_audio) < MIN_LENGTH_FOR_VAD:
            return False
        speech_ts = self._get_speech_timestamps(norm_audio)
        return len(speech_ts) > 0

    # Gets called on another thread.
    def write(self, data: bytes, user_id: Optional[int]) -> None:
        """Buffer raw audio until voice detected, then commit to speech buffer."""
        if user_id is None:
            return
        audio_np = pcm16_to_norm_waveform(data, SOURCE_SR, TARGET_SR)
        state = self.user_states[user_id]
        # synchronize buffer access
        with state.lock:
            speech_buf = state.speech_buf
            raw_buf = state.raw_buf
            state.last_noise = datetime.datetime.now(datetime.timezone.utc)

            # accumulate raw pre-speech buffer and perform VAD check.
            raw_buf.write(audio_np)
            is_speaking = self._has_voice(raw_buf.get_contents_slice())
            was_speaking = state.last_spoke is not None

            if was_speaking:
                # If we have already detected speech, we keep appending to the speech buffer regardless of VAD.
                speech_buf.write(audio_np)
            elif is_speaking:
                # If we haven't detected speech yet and VAD is triggered, push the raw buffer contents to the speech buffer
                # and clear the raw buffer.
                speech_buf.write(raw_buf.get_contents_slice())
                state.first_spoke = state.last_noise

            if is_speaking:
                raw_buf.pos = 0
                state.last_spoke = state.last_noise
            else:
                # No speech detected. Keep the contents of the raw buffer for future VAD checks.
                # But if the buffer is too large, we need to shift it, discarding the oldest data.
                raw_buf.shift(raw_buf.pos - RAW_BUFFER_MAX_LENGTH)

    async def _get_user_name(self, user_id: int) -> str:
        """Get the username for a given user ID, caching it if necessary."""
        if user_id not in self._cached_user_names:
            user = self.voice_client.client.get_user(user_id)
            if user:
                self._cached_user_names[user_id] = user.name
            else:
                user = await self.voice_client.client.fetch_user(user_id)
                if user:
                    self._cached_user_names[user_id] = user.name
                else:
                    self._cached_user_names[user_id] = f"user_{user_id}"
        return self._cached_user_names[user_id]
    
    async def _heartbeat_loop(self) -> None:
        """Internal loop that looks for silence breaks in captured audio and submits it to the queue."""
        try:
            while True:
                channel_id = self.voice_client.channel.id
                for user_id, state in list(self.user_states.items()):
                    with state.lock:
                        now = datetime.datetime.now(datetime.timezone.utc)
                        speech_size = state.speech_buf.pos
                        if speech_size > 0 and state.last_spoke:
                            detected_silence = (now - state.last_spoke).total_seconds() > SILENCE_THRESHOLD_SECONDS
                            buf_seconds = speech_size / TARGET_SR
                            is_max_buf = MAX_SPEECH_BUF_SECONDS and buf_seconds > MAX_SPEECH_BUF_SECONDS
                            if detected_silence or is_max_buf:
                                audio_np: bytes = state.speech_buf.get_contents_slice().copy()
                                user_name = await self._get_user_name(user_id)
                                # Filter samples that are too quiet.
                                if has_sound(audio_np, threshold=0.05, total_sound_ms=500):
                                    metadata = VoiceMetadata(
                                        user_id=user_id,
                                        user_name=user_name,
                                        channel_id=channel_id,
                                        capture_time=state.first_spoke,
                                    )
                                    await self.capture_queue.put((metadata, audio_np))
                                state.last_spoke = None
                                state.first_spoke = None
                                state.speech_buf.pos = 0
                        if state.last_noise and (now - state.last_noise).total_seconds() > PRUNE_THRESHOLD_SECONDS:
                            self.user_states.pop(user_id, None)
                await asyncio.sleep(0.1)
        finally:
            print("VoiceCaptureSink: heartbeat loop finished.")

    def is_started(self) -> bool:
        """Check if the sink is started."""
        return self._heartbeat_task is not None
    
    async def start(self, vc: discord.VoiceClient):
        """Starts the background heartbeat task."""
        # Then start the silence detection loop
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.voice_client = vc
        print("VoiceCaptureSink started.")
    
    async def stop(self):
        """Stops the heartbeat task and clears buffers."""
        # Stop the heartbeat loop (prevents new jobs being submitted by it)
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass # Expected.
            self._heartbeat_task = None
        
        # Clear buffers.
        self.user_states.clear()
        print("VoiceCaptureSink stopped.")