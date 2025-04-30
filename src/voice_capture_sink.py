import discord
import numpy as np
import io
from collections import defaultdict
import datetime
import asyncio
from typing import Optional, DefaultDict, Tuple
from scipy.signal import resample_poly
from typing import Optional, DefaultDict, Tuple
import asyncio
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from dataclasses import dataclass, field
import threading

from .config import SILENCE_THRESHOLD_SECONDS, VAD_THRESHOLD, MAX_SPEECH_BUF_BYTES

# Module-level audio constants
SOURCE_SR = 48000  # Discord's PCM sample rate
TARGET_SR = 16000  # Required sample rate for VAD and model

# raw-buffer constants for custom VAD pre-buffering
RAW_BUFFER_DURATION = 0.5  # seconds
RAW_BUFFER_MAX_BYTES = int(TARGET_SR * RAW_BUFFER_DURATION * 2)

PRUNE_THRESHOLD_SECONDS = 5.

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

@dataclass
class UserState:
    speech_buf: io.BytesIO
    vad_buf: io.BytesIO
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
                speech_buf=io.BytesIO(),
                vad_buf=io.BytesIO(),
            )
        )
        self._cached_user_names = {}
        vad_model = load_silero_vad()
        self._get_speech_timestamps = lambda audio: \
            get_speech_timestamps(audio, model=vad_model, threshold=VAD_THRESHOLD, min_speech_duration_ms=200)
        self._heartbeat_task = None

    def _has_voice(self, audio_16khz: bytes) -> bool:
        """Check if 16kHz mono int16 audio contains speech using Silero VAD"""
        # convert raw PCM int16 to normalized float32 waveform
        samples = np.frombuffer(audio_16khz, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(samples).squeeze(0)
        # get speech timestamps from Silero VAD
        speech_ts = self._get_speech_timestamps(waveform)

        return len(speech_ts) > 0

    def _convert_audio(self, audio: bytes) -> bytes:
        """Convert raw 48kHz stereo PCM int16 audio to 16kHz mono PCM int16"""
        samples = np.frombuffer(audio, dtype=np.int16).reshape(-1, 2)
        mono = np.mean(samples.astype(np.float32), axis=1)
        resampled = resample_poly(mono, TARGET_SR, SOURCE_SR)
        pcm16 = np.clip(resampled, -32768, 32767).astype(np.int16)
        return pcm16.tobytes()

    # Gets called on another thread.
    def write(self, data: bytes, user_id: Optional[int]) -> None:
        """Buffer raw audio until voice detected, then commit to speech buffer."""
        if user_id is None:
            return
        audio16 = self._convert_audio(data)
        state = self.user_states[user_id]
        # synchronize buffer access
        with state.lock:
            speech_buf = state.speech_buf
            vad_buf = state.vad_buf
            state.last_noise = datetime.datetime.now(datetime.timezone.utc)

            # accumulate raw pre-speech buffer
            vad_buf.seek(0, io.SEEK_END)
            vad_buf.write(audio16)
            # trim if exceeds max size
            if vad_buf.tell() > RAW_BUFFER_MAX_BYTES:
                vad_buf.seek(0)
                buf_data = vad_buf.getvalue()[-RAW_BUFFER_MAX_BYTES:]
                vad_buf.truncate(0)
                vad_buf.write(buf_data)

            # VAD check on buffered audio
            raw_bytes = vad_buf.getvalue()
            is_speaking = self._has_voice(raw_bytes)

            if is_speaking:
                # clear raw buffer when speech starts
                vad_buf.seek(0)
                vad_buf.truncate(0)

            if speech_buf.tell() > 0:
                # Always append the new clip to the speech buffer if it has data because it implies that
                # we detected speech previously and are still in the same speaking session.
                speech_buf.write(audio16)
                if is_speaking:
                    state.last_spoke = datetime.datetime.now(datetime.timezone.utc)
            elif is_speaking:
                # First time speech detected, write pre-buffered audio to speech buffer.
                speech_buf.write(raw_bytes)
                state.last_spoke = datetime.datetime.now(datetime.timezone.utc)
                state.first_spoke = state.last_spoke

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
                        speech_size = state.speech_buf.tell()
                        if speech_size > 0 and state.last_spoke:
                            detected_silence = (now - state.last_spoke).total_seconds() > SILENCE_THRESHOLD_SECONDS
                            is_max_size = MAX_SPEECH_BUF_BYTES and speech_size > MAX_SPEECH_BUF_BYTES
                            if detected_silence or is_max_size:
                                raw_audio: bytes = state.speech_buf.getvalue()
                                user_name = await self._get_user_name(user_id)
                                if detected_silence:
                                    print(f"VoiceCaptureSink: Detected silence for user {user_name}. Submitting job...")
                                else:
                                    print(f"VoiceCaptureSink: Buffer size exceeded for user {user_name}. Submitting job...")
                                metadata = VoiceMetadata(
                                    user_id=user_id,
                                    user_name=user_name,
                                    channel_id=channel_id,
                                    capture_time=state.first_spoke,
                                )
                                await self.capture_queue.put((metadata, raw_audio))
                                state.last_spoke = None
                                state.first_spoke = None
                                state.speech_buf.seek(0)
                                state.speech_buf.truncate(0)
                        if state.last_noise and (now - state.last_noise).total_seconds() > PRUNE_THRESHOLD_SECONDS:
                            self.user_states.pop(user_id, None)
                await asyncio.sleep(0.33)
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