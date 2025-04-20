import discord
import numpy as np
import io
from collections import defaultdict
import datetime
import asyncio
# Updated imports
from typing import Optional, Dict, Any, List, DefaultDict, Tuple
from scipy.signal import resample_poly

from config import SILENCE_THRESHOLD_SECONDS
# Import generic Metadata type
from typing import Optional, DefaultDict, Tuple
import asyncio

# Define the specific type of metadata we will use
BotMetadata = Tuple[int, Optional[int], datetime.datetime]  # (user_id, channel_id, capture_time)

class SilenceSink(discord.sinks.Sink):
    """A sink that buffers audio and submits it to an AudioProcessor when a user stops speaking."""
    # Type hints
    voice_client: discord.VoiceClient
    capture_queue: asyncio.Queue[Tuple[BotMetadata, bytes]]  # Queue for buffering audio to bot
    client: discord.Client # Keep client for potential future use
    user_audio_buffers: DefaultDict[int, io.BytesIO]
    user_last_spoke: DefaultDict[int, datetime.datetime]
    silence_threshold: float
    _cleanup_task: Optional[asyncio.Task]
    # Removed whisper_pipeline, session_log, log_lock, _target_sr, _discord_sr

    # Updated __init__ signature
    def __init__(self, vc: discord.VoiceClient, capture_queue: asyncio.Queue[Tuple[BotMetadata, bytes]], client: discord.Client):
        super().__init__()
        self.voice_client = vc
        self.capture_queue = capture_queue
        self.client = client
        self.user_audio_buffers = defaultdict(io.BytesIO)
        self.user_last_spoke = defaultdict(lambda: datetime.datetime.now(datetime.timezone.utc))
        self.silence_threshold = SILENCE_THRESHOLD_SECONDS
        self._cleanup_task = None

    # write method remains the same
    def write(self, data: bytes, user_id: Optional[int])-> None:
        """Called by py-cord when audio data is received for a user."""
        if user_id is None:
            return

        buffer = self.user_audio_buffers[user_id]
        buffer.write(data)
        self.user_last_spoke[user_id] = datetime.datetime.now(datetime.timezone.utc)

    async def _cleanup_loop(self) -> None:
        """Internal loop that detects silence and submits jobs to the AudioProcessor."""
        try:
            while True:
                now: datetime.datetime = datetime.datetime.now(datetime.timezone.utc)
                users_processed = []
                # Derive text channel ID from voice client
                channel_id = self.voice_client.channel.id

                for user_id, last_spoke_time in list(self.user_last_spoke.items()):
                    if (now - last_spoke_time).total_seconds() > self.silence_threshold:
                        buffer: Optional[io.BytesIO] = self.user_audio_buffers.get(user_id)
                        if buffer and buffer.getbuffer().nbytes > 0:
                            print(f"SilenceSink: Detected silence for user {user_id}. Submitting job...")
                            audio_data: bytes = buffer.getvalue() # Get a copy

                            # Create the metadata tuple with the specific type, using last spoke time
                            metadata: BotMetadata = (user_id, channel_id, last_spoke_time)

                            # Enqueue job into capture queue for bot to submit
                            await self.capture_queue.put((metadata, audio_data))

                            # Clear the original buffer immediately after submitting
                            buffer.seek(0)
                            buffer.truncate(0)
                            print(f"SilenceSink: Buffer for user {user_id} submitted and cleared.")

                        # Mark user for cleanup check
                        users_processed.append(user_id)

                # Clean up entries for users who met the silence threshold
                for user_id in users_processed:
                     current_buffer: Optional[io.BytesIO] = self.user_audio_buffers.get(user_id)
                     if current_buffer is None or current_buffer.getbuffer().nbytes == 0:
                         self.user_audio_buffers.pop(user_id, None)
                         self.user_last_spoke.pop(user_id, None)
                         print(f"SilenceSink: Cleaned up entries for silent user {user_id}")

                await asyncio.sleep(0.5)
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
        # Nothing to start here; bot handles audio_processor

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
        for user_id, buffer in list(self.user_audio_buffers.items()):
             if buffer.getbuffer().nbytes > 0:
                 print(f"SilenceSink: Submitting remaining audio for user {user_id}...")
                 audio_data = buffer.getvalue()
                 # Use the last spoke timestamp or now if missing
                 ts = self.user_last_spoke.get(user_id, datetime.datetime.now(datetime.timezone.utc))
                 metadata: BotMetadata = (user_id, channel_id, ts)
                 # Enqueue remaining job
                 await self.capture_queue.put((metadata, audio_data))
                 remaining_jobs += 1
                 buffer.seek(0)
                 buffer.truncate(0)
        self.user_audio_buffers.clear()
        self.user_last_spoke.clear()
        print(f"SilenceSink: Finished submitting {remaining_jobs} remaining audio jobs.")

        # 3. Nothing further to do here; bot will stop audio_processor

        print("SilenceSink cleanup complete.")

    # Removed process_audio and process_remaining_audio methods
