import discord
import numpy as np
import io
from collections import defaultdict
import datetime
import asyncio
from typing import Optional, Dict, Any, List, DefaultDict, Tuple
from scipy.signal import resample_poly
from config import SILENCE_THRESHOLD_SECONDS
from typing import Optional, DefaultDict, Tuple
import asyncio
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from dataclasses import dataclass, field
import threading

# Module-level audio constants
SOURCE_SR = 48000  # Discord's PCM sample rate
TARGET_SR = 16000  # Required sample rate for VAD and model

# raw-buffer constants for custom VAD pre-buffering
RAW_BUFFER_DURATION = 0.5  # seconds
RAW_BUFFER_MAX_BYTES = int(TARGET_SR * RAW_BUFFER_DURATION * 2)

PRUNE_THRESHOLD_SECONDS = 5.


# Define the specific type of metadata we will use
BotMetadata = Tuple[int, Optional[int], datetime.datetime]  # (user_id, channel_id, capture_time)

@dataclass
class UserState:
    speech_buf: io.BytesIO
    vad_buf: io.BytesIO
    last_spoke: Optional[datetime.datetime]
    last_noise: Optional[datetime.datetime]
    lock: threading.Lock = field(default_factory=threading.Lock)

class SilenceSink(discord.sinks.Sink):
    """A sink that buffers audio and submits it to an AudioProcessor when a user stops speaking."""
    # Type hints
    voice_client: discord.VoiceClient
    capture_queue: asyncio.Queue[Tuple[BotMetadata, bytes]]  # Queue for buffering audio to bot
    user_states: DefaultDict[int, UserState]
    silence_threshold: float
    _cleanup_task: Optional[asyncio.Task]
    _get_speech_timestamps: callable # Silero VAD utility function

    # Updated __init__ signature
    def __init__(self, vc: discord.VoiceClient, capture_queue: asyncio.Queue[Tuple[BotMetadata, bytes]]):
        super().__init__()
        self.voice_client = vc
        self.capture_queue = capture_queue
        # per-user audio state
        self.user_states: DefaultDict[int, UserState] = defaultdict(
            lambda: UserState(io.BytesIO(), io.BytesIO(), None, None)
        )
        # Load Silero VAD model and utilities from silero_vad library
        vad_model = load_silero_vad()
        self._get_speech_timestamps = lambda audio: get_speech_timestamps(audio, model=vad_model, threshold=0.55, min_speech_duration_ms=250)
        self._cleanup_task = None

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
                    # print(f"SilenceSink: User {user_id} is still speaking.")
            elif is_speaking:
                # First time speech detected, write pre-buffered audio to speech buffer.
                speech_buf.write(raw_bytes)
                state.last_spoke = datetime.datetime.now(datetime.timezone.utc)
                print(f"SilenceSink: User {user_id} speech detected in pre-buffer.")

    async def _cleanup_loop(self) -> None:
        """Internal loop that detects silence and submits jobs to the AudioProcessor."""
        try:
            while True:
                # Derive text channel ID from voice client
                channel_id = self.voice_client.channel.id

                for user_id, state in list(self.user_states.items()):
                    # synchronize buffer access
                    with state.lock:
                        now = datetime.datetime.now(datetime.timezone.utc)
                        if state.last_spoke and (now - state.last_spoke).total_seconds() > SILENCE_THRESHOLD_SECONDS:
                            if state.speech_buf.tell() > 0:
                                raw_audio: bytes = state.speech_buf.getvalue()
                                print(f"SilenceSink: Detected silence for user {user_id}. Submitting job...")
                                metadata: BotMetadata = (user_id, channel_id, state.last_spoke)
                                await self.capture_queue.put((metadata, raw_audio))
                                state.last_spoke = None
                                state.speech_buf.seek(0)
                                state.speech_buf.truncate(0)
                                print(f"SilenceSink: Buffer for user {user_id} submitted and cleared.")
                        if state.last_noise and (now - state.last_noise).total_seconds() > PRUNE_THRESHOLD_SECONDS:
                            self.user_states.pop(user_id, None)
                            print(f"SilenceSink: Cleaned up entries for silent user {user_id}")
                await asyncio.sleep(0.33)
        except asyncio.CancelledError:
            print("SilenceSink: Cleanup task cancelled.")
            # No need to process remaining audio here, stop_cleanup handles it.
        except Exception as e:
             import traceback
             print(f"SilenceSink: Error in cleanup loop: {e} {traceback.format_exc()}")
        finally:
            print("SilenceSink: Cleanup loop finished.")


    async def start_cleanup(self) -> None:
        """Starts the background cleanup task and the audio processor thread."""
        # Then start the silence detection loop
        if self._cleanup_task is None or self._cleanup_task.done():
            print("SilenceSink: Starting silence detection task.")
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        else:
            print("SilenceSink: Cleanup task already running.")

    async def stop_cleanup(self) -> None:
        """Stops the cleanup task and the audio processor gracefully."""
        # 1. Stop the cleanup loop (prevents new jobs being submitted by it)
        if self._cleanup_task and not self._cleanup_task.done():
            print("SilenceSink: Stopping silence detection task...")
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                print("SilenceSink: Cleanup task successfully cancelled.") # Expected
            self._cleanup_task = None
        else:
            print("SilenceSink: Cleanup task not running or already stopped.")

        # 2. Submit any remaining audio from buffers
        print("SilenceSink: Submitting any remaining audio buffers before shutdown...")
        remaining_jobs = 0
        # Derive text channel ID from voice client for final submission
        channel_id = self.voice_client.channel.id
        for user_id, state in list(self.user_states.items()):
            # synchronize final buffer submission
            with state.lock:
                if state.speech_buf.getbuffer().nbytes > 0:
                    print(f"SilenceSink: Submitting remaining audio for user {user_id}...")
                    audio_data = state.speech_buf.getvalue()
                    ts = state.last_spoke
                    metadata: BotMetadata = (user_id, channel_id, ts)
                    await self.capture_queue.put((metadata, audio_data))
                    remaining_jobs += 1
                    state.speech_buf.seek(0)
                    state.speech_buf.truncate(0)
        self.user_states.clear()
        print(f"SilenceSink: Finished submitting {remaining_jobs} remaining audio jobs.")

        # 3. Nothing further to do here; bot will stop audio_processor

        print("SilenceSink cleanup complete.")