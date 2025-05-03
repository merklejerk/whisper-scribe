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
import traceback
import math

from .config import SILENCE_THRESHOLD_SECONDS, VAD_THRESHOLD, MAX_SPEECH_BUF_SECONDS
from .waveform_utils import (
    pcm16_to_norm_waveform,
    has_sound,
    bandpass_filter,
    pre_emphasis,
    rms_normalize,
    get_duration,
    resample,
)

# Module-level audio constants
SOURCE_SR = 48000  # Discord's PCM sample rate
TARGET_SR = 16000  # Required sample rate for VAD and transcriber
BUF_SR = 24000  # Buffer sample rate
BYTES_PER_SAMPLE = 4 # float32

# VAD constants
MIN_DURATION_FOR_VAD_SECONDS = 0.5

# raw-buffer constants for custom VAD pre-buffering
RAW_BUFFER_DURATION = 5.0  # seconds

PRUNE_THRESHOLD_SECONDS = 120.

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
    
    def get_content_slice(self) -> np.ndarray:
        """Get the contents of the buffer."""
        return self.array[:self.pos]
    
    def write(self, data: np.ndarray) -> None:
        """Write data to the buffer, expanding it if necessary."""
        end = self.pos + len(data)
        if end > len(self.array):
            new_size = max(len(self.array) * 2, end)
            self.expand(new_size - len(self.array))
        self.array[self.pos:end] = data
        self.pos = end
    
    def expand(self, n: int) -> None:
        """Expand the size of the buffer."""
        if n > 0:
            self.array.resize((len(self.array) + n,), refcheck=False)
    
    def shift(self, n: int) -> int:
        """Shift the buffer contents to the left by n items."""
        n = min(n, self.pos)
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
    _process_task: Optional[asyncio.Task]
    _get_speech_timestamps: callable # Silero VAD utility function
    _states_lock: threading.Lock
    _process_event: asyncio.Event
    _loop: asyncio.AbstractEventLoop

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
        self._process_task = None
        self._states_lock = threading.Lock()
        self._process_event = asyncio.Event()

    def _get_voice_time(self, norm_audio: np.ndarray, sr: int) -> float | None:
        """Check if normalized float32 audio contains speech using Silero VAD"""
        if get_duration(norm_audio, sr) < MIN_DURATION_FOR_VAD_SECONDS:
            return False
        if sr != TARGET_SR:
            norm_audio = resample(norm_audio, sr, TARGET_SR)
        speech_ts = self._get_speech_timestamps(norm_audio)
        if len(speech_ts) == 0:
            return None
        return speech_ts[0]['start'] / TARGET_SR

    # Gets called on another thread.
    def write(self, data: bytes, user_id: Optional[int]) -> None:
        """Buffer raw audio until voice detected, then commit to speech buffer."""
        if user_id is None or len(data) == 0:
            return
        audio_np = pcm16_to_norm_waveform(data, SOURCE_SR, BUF_SR)
        with self._states_lock:
            state = self.user_states[user_id]
        # synchronize buffer access
        with state.lock:
            speech_buf = state.speech_buf
            raw_buf = state.raw_buf
            state.last_noise = datetime.datetime.now(datetime.timezone.utc)

            # accumulate raw pre-speech buffer and perform VAD check.
            raw_buf.write(audio_np)
            voice_time = self._get_voice_time(raw_buf.get_content_slice(), BUF_SR)
            is_speaking = voice_time is not None
            was_speaking = state.last_spoke is not None

            if was_speaking:
                # If we have already detected speech, we keep appending to the speech buffer regardless of VAD.
                speech_buf.write(audio_np)
            elif is_speaking:
                # If we haven't detected speech yet and VAD is triggered, push the raw buffer contents from
                # the VAD point onwards into the speech buffer.
                speech_buf.write(raw_buf.get_content_slice()[int(voice_time * BUF_SR):])
                state.first_spoke = state.last_noise

            if is_speaking:
                raw_buf.pos = 0
                state.last_spoke = state.last_noise
            else:
                # No speech detected. Keep the contents of the raw buffer for future VAD checks.
                # But if the buffer is too large, we need to shift it, discarding the oldest data.
                raw_buf.shift(raw_buf.pos - int(math.ceil(RAW_BUFFER_DURATION * BUF_SR)))
        self._loop.call_soon_threadsafe(lambda: self._process_event.set())
    
    async def _process_loop(self) -> None:
        """Internal loop that looks for silence breaks in captured audio and submits it to the queue."""
        try:
            while True:
                await self._process_event.wait()
                self._process_event.clear()
                with self._states_lock:
                    state_items = list(self.user_states.items())
                await asyncio.gather(
                    *[self._process_speech_state(user_id, state) for user_id, state in state_items]
                )
                with self._states_lock:
                    now = datetime.datetime.now(datetime.timezone.utc)
                    for user_id, state in (x for x in state_items if x[1].last_noise):
                        silent_time = (now - state.last_noise).total_seconds()
                        if silent_time > PRUNE_THRESHOLD_SECONDS:
                            print(f"Pruning user {user_id} buf after {silent_time:.2f}s of silence.")
                            self.user_states.pop(user_id, None)
                    if self.user_states:
                        self._loop.call_later(0.1, lambda: self._process_event.set())
                
        except Exception as e:
            traceback.print_exc()
            raise

    async def _process_speech_state(self, user_id: int, state: UserState) -> None:
        """ Process the speech state for a given user."""
        with state.lock:
            now = datetime.datetime.now(datetime.timezone.utc)
            speech = state.speech_buf.get_content_slice()
            if len(speech) == 0 or not state.last_spoke:
                return

            detected_silence = (now - state.last_spoke).total_seconds() > SILENCE_THRESHOLD_SECONDS + MIN_DURATION_FOR_VAD_SECONDS
            is_max_buf = MAX_SPEECH_BUF_SECONDS and get_duration(speech, BUF_SR) > MAX_SPEECH_BUF_SECONDS

            if not detected_silence and not is_max_buf:
                return
            
            speech = state.speech_buf.get_content_slice()
            capture_time = state.first_spoke
            state.last_spoke = None
            state.first_spoke = None
            state.speech_buf.pos = 0
            
            if not has_sound(speech, threshold=0.05, total_sound_ms=500):
                return
            speech = speech.copy()
        
        metadata = VoiceMetadata(
            user_id=user_id,
            user_name=await self._get_user_name(user_id),
            channel_id=self.voice_client.channel.id,
            capture_time=capture_time,
        )
        await self.capture_queue.put((
            metadata,
            resample(enhance_speech(speech, BUF_SR), BUF_SR, TARGET_SR),
        ))    

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

    def is_started(self) -> bool:
        """Check if the sink is started."""
        return self._process_task is not None
    
    async def start(self, vc: discord.VoiceClient):
        """Starts the background process task."""
        # Then start the silence detection loop
        if self._process_task is None:
            self._process_task = asyncio.create_task(self._process_loop())
        self.voice_client = vc
        self._loop = asyncio.get_event_loop()
        print("VoiceCaptureSink started.")

    
    async def stop(self):
        """Stops the process task and clears buffers."""
        # Stop the process task (prevents new jobs being submitted by it)
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass # Expected.
            self._process_task = None
        
        # Clear buffers.
        self.user_states.clear()
        print("VoiceCaptureSink stopped.")


def enhance_speech(audio: np.ndarray, sr: int) -> np.ndarray:
    return rms_normalize(
        bandpass_filter(
            pre_emphasis(audio, 0.8),
            sr,
            order=3
        ),
    ).clip(-1.0, 1.0)