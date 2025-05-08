import discord
from discord.ext.commands import Bot,Command
import asyncio
import datetime
from typing import Optional, Any
import asyncio
import re
import os
import traceback
from collections import deque

import src.config as config
from .wrapup import create_wrapup_from_log_entries
from .logging import write_log_entry, load_log, LogEntry
from .transcriber import Transcriber, TranscriptionResult
from .voice_capture_sink import VoiceCaptureSink, VoiceMetadata
from .patched_voice_client import PatchedVoiceClient
from .refiner import TranscriptRefiner

def get_session_log_path(session_name: str) -> str:
    """Returns the path to the session log file."""
    return f"logs/{session_name}.jsonl"

MAX_JOIN_TIMEOUT_SECONDS = 10.0
RETRY_CONNECTION_SECONDS = 5.0

class DiscordBot(object):
    session_name: str
    voice_channel_id: int
    _client: Bot
    _vc: Optional[discord.VoiceClient]
    _sink: VoiceCaptureSink
    _transcription_queue: asyncio.Queue[TranscriptionResult]
    _capture_queue: asyncio.Queue
    _transcriber: Transcriber[VoiceMetadata]
    _refiner: TranscriptRefiner
    _transcription_consumer_task: Optional[asyncio.Task]
    _capture_consumer_task: Optional[asyncio.Task]
    _recent_log_entries: deque[LogEntry]

    def __init__(
            self,
            voice_channel_id: int,
            *,
            session_name: Optional[str] = None,
            device: str = "cpu",
            log_metadata: bool = False,
        ):
        # Setup Discord client and internal state
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        self._client = Bot(intents=intents, command_prefix="!")
        self.voice_channel_id = voice_channel_id
        # Use a timestamp for session id if not provided.
        self.session_name = session_name or datetime.datetime.now().strftime("%d-%m-%Y_%H%M")

        self._vc = None
        # Create the output queue, specifying the generic type
        self._transcription_queue = asyncio.Queue()
        # Create the capture queue for raw audio samples
        self._capture_queue = asyncio.Queue()
        self._sink = VoiceCaptureSink(capture_queue=self._capture_queue)
        self._transcriber: Transcriber[VoiceMetadata] = Transcriber(
            output_queue=self._transcription_queue,
            model_name=config.WHISPER_MODEL,
            device=device
        )
        self._refiner = TranscriptRefiner()
        # Task for consuming transcriptions
        self._transcription_consumer_task = None
        # Task for consuming capture queue and submitting jobs
        self._capture_consumer_task = None
        # Initialize recent log entries buffer
        self._recent_log_entries = deque(maxlen=config.REFINER_CONTEXT_LOG_LINES)
        self._log_metadata = log_metadata

        # Regiser commands.
        self._client.add_command(Command(self._wrapup_command, name='wrapup'))
        self._client.add_command(Command(self._log_command, name='log'))  # Add log command
        # Register listeners.
        self._client.add_listener(self._on_message, 'on_message')
        self._client.add_listener(self._on_ready, 'on_ready')
        # Add the global check
        self._client.add_check(self._command_check)
        
    async def bootstrap_recent_log_entries(self):
        """Load the last N log entries from the session log file into _recent_log_entries."""
        log_path = get_session_log_path(self.session_name)
        entries = await load_log(log_path, allow_missing=True)
        # Only keep the last N entries
        for entry in entries[-config.REFINER_CONTEXT_LOG_LINES:]:
            self._recent_log_entries.append(entry)
        if len(self._recent_log_entries) != 0:
            print(f"Bootstrapped _recent_log_entries with {len(self._recent_log_entries)} entries from log.")

    async def start(self, api_key: str, *, on_ready: Optional[callable] = None):
        await self.bootstrap_recent_log_entries()
        self._transcription_consumer_task = asyncio.create_task(
            self._process_transcription_queue(),
            name="transcription_consumer_task"
        )
        self._capture_consumer_task = asyncio.create_task(
            self._process_capture_queue(),
            name="capture_consumer_task"
        )
        if on_ready:
            async def wrapped_on_ready():
                await on_ready(self)
            self._client.add_listener(wrapped_on_ready, 'on_ready')
        await self._client.start(api_key)

    def _command_check(self, ctx: Any) -> bool:
        # Command must be invoked in the voice channel.
        if ctx.channel.id != self.voice_channel_id:
            return False
        # Must be invoked by an allowed user.
        if config.ALLOWED_COMMANDERS:
            if not ctx.author or ctx.author.id not in config.ALLOWED_COMMANDERS:
                return False
        return True

    async def _connect_and_start_recording(self, channel_id: int):
        """Connects to the voice channel, creates sink, and starts recording.

        Returns True on success, False on failure.
        Handles updating self._vc and self._sink.
        """
        # Disconnect any existing voice client.
        channel = await self._client.fetch_channel(channel_id)
        if not isinstance(channel, discord.VoiceChannel):
            raise ValueError(f"Error: Channel ID {channel_id} is not a voice channel or not found.")

        if self._vc:
            if self._vc.is_connected():
                await self._vc.disconnect(force=True)
            print("Forcibly disconnected old voice client.")
            self._vc = None

        try:
            self._transcriber.start()
            print(f"Attempting to join voice channel: {channel.name}...")
            vc = self._vc = await channel.connect(
                # reconnect=True,
                cls=PatchedVoiceClient,
                timeout=MAX_JOIN_TIMEOUT_SECONDS,
            )
            print(f"Successfully joined voice channel: {channel.name} (ID: {channel.id})")

            await self._sink.start(vc)

            inst = self
            async def _finished_callback(_: VoiceCaptureSink, channel: discord.VoiceChannel) -> None:
                """Called by py-cord when recording stops."""
                print(f"Warning: Recording stopped unexpectedly in channel {channel.name} (ID: {channel.id})!")
                num_retries = 8
                for attempt in range(num_retries):
                    try:
                        await inst._connect_and_start_recording(channel.id)
                        break
                    except Exception as e:
                        traceback.print_exc()
                        if attempt < num_retries - 1:
                            print(f"Trying again in {RETRY_CONNECTION_SECONDS} seconds ({attempt + 2}/{num_retries})...")
                            await asyncio.sleep(RETRY_CONNECTION_SECONDS)
                        else:
                            raise e
            
            vc.start_recording(self._sink, _finished_callback, vc.channel)
        except:
            if self._vc and self._vc.is_connected():
                await self._vc.disconnect(force=True)
            self._vc = None
            raise

    # Called when the bot successfully connects and is ready.
    async def _on_ready(self) -> None:
        """Called when the bot successfully connects and is ready."""
        print(f'Logged in as {self._client.user.name} (ID: {self._client.user.id})')
        # Automatically connect to the specified voice channel.
        await self.join_channel(self.voice_channel_id)

    async def write_log_entry(self, entry: LogEntry):
        print(f"> {entry.user_name}: {entry.content}")
        await write_log_entry(
            log_path=get_session_log_path(self.session_name),
            entry=entry,
        )

    async def _process_transcription_queue(self):
        """Consumes transcription results, refines them, and logs them."""
        print("Transcription consumer task started.")
        while True:
            result = await self._transcription_queue.get()
            user_id = result.metadata.user_id
            user_name = result.metadata.user_name
            capture_time = result.metadata.capture_time
            content = result.transcription
            try:
                if not re.search(r"[a-z0-9]+", content):
                    continue
                # Refine the transcription
                context_log = list(self._recent_log_entries)
                entry = LogEntry(
                    user_id=user_id,
                    user_name=user_name,
                    timestamp=capture_time,
                    content=content,
                    medium="voice",
                    metadata={
                        "raw_content": content,
                        "mean_logprob": result.mean_logprob,
                        "std_logprob": result.std_logprob,
                        "n_tokens": result.n_tokens,
                    } if self._log_metadata else None,
                )
                refined_content = await self._refiner.refine(entry, context_log)
                if not refined_content:
                    print(f"Rejected transcript for {entry.user_name}:\n\t\"{entry.content}\"")
                    continue
                if refined_content != content:
                    print(f"Refined transcript for {entry.user_name}:\n\t--- \"{entry.content}\"\n\t+++ \"{refined_content}\"")
                entry.content = refined_content
                self._recent_log_entries.append(entry)

                # Log the refined transcription
                await self.write_log_entry(entry)
            except Exception as e:
                print(f"Error processing transcription: {e}")
                traceback.print_exc()
            finally:
                self._transcription_queue.task_done()

    async def _process_capture_queue(self):
        """Consumes raw audio from capture queue and submits to Transcriber."""
        print("Capture consumer task started.")
        while True:
            metadata, audio_data = await self._capture_queue.get()
            try:
                self._transcriber.submit_job(metadata, audio_data)
            finally:
                self._capture_queue.task_done()

    # Event handler: Called when a message is sent.
    async def _on_message(self, message: discord.Message) -> None:
        """Called when a message is sent."""
        if message.author == self._client.user: # Use self.user
            return
        if message.author.bot:
            return
        if message.content.startswith("!"):
            # Ignore commands
            return
        # Only log messages in the text channel with the same ID as the last voice channel
        if message.channel.id == self.voice_channel_id:
            # Delegate formatting and logging
            await self.write_log_entry(
                LogEntry(
                    medium="text",
                    user_id=message.author.id,
                    user_name=message.author.name,
                    content=message.content,
                    timestamp=datetime.datetime.now(datetime.timezone.utc),
                )
            )

    async def shutdown(self):
        """Cleanup logic to run when the bot is shutting down."""
        print("Bot is shutting down...")

        if self._vc and self._vc.is_connected():
            print("Disconnecting from voice channel...")
            # Sink cleanup already handled above
            await self._vc.disconnect(force=True)
            print("Disconnected.")
        
        if not self._client.is_closed():
             await self._client.close()

        # Stop sink and transcriber
        if self._sink and self._sink.is_started():
            await self._sink.stop()
        if self._transcriber.is_started():
            self._transcriber.stop()

        # Cancel the transcription consumer task first
        if self._transcription_consumer_task and not self._transcription_consumer_task.done():
            print("Stopping transcription consumer task...")
            self._transcription_consumer_task.cancel()
            try:
                await self._transcription_consumer_task
            except asyncio.CancelledError:
                print("Transcription consumer task successfully cancelled.")
            self._transcription_consumer_task = None

        # Cancel the capture consumer task
        if self._capture_consumer_task and not self._capture_consumer_task.done():
            print("Stopping capture consumer task...")
            self._capture_consumer_task.cancel()
            try:
                await self._capture_consumer_task
            except asyncio.CancelledError:
                print("Capture consumer task successfully cancelled.")
            self._capture_consumer_task = None

        print("Shutdown complete.")

    async def join_channel(self, channel_id: int):
        if self._client.is_closed():
            raise RuntimeError("Not started.")
        await self._connect_and_start_recording(channel_id)

    async def _wrapup_command(self, ctx: Any):
        """Process the current session log and generate a D&D 5e outline."""
        await ctx.send("🖨️ Generating wrapup...")
        print(f"Generating wrapup for session {self.session_name}...")
        
        log_path = get_session_log_path(self.session_name)
        if not os.path.exists(log_path):
            # Nothing to do.
            ctx.send(f"No session log to wrap up.")
            return
        
        try:
            wrapup_files = await create_wrapup_from_log_entries(
                await load_log(log_path),
                self.session_name,
            )
            print("Wrapup generation complete.")

            files_to_upload = [
                discord.File(wrapup_files.chatlog_path),
            ]
            if wrapup_files.outline_path:
                files_to_upload.append(discord.File(wrapup_files.outline_path))

            print(f"Uploading {len(files_to_upload)} files to channel {ctx.channel.id}...")
            await ctx.send("🎁", files=files_to_upload)

        except Exception as e:
            await ctx.send(f"Failed to generate wrapup.")
            raise

    async def _log_command(self, ctx: Any):
        """Generate and post only the wrapup chatlog for the current session."""
        await ctx.send("🖨️ Generating chatlog...")
        print(f"Generating chatlog for session {self.session_name}...")
        log_path = get_session_log_path(self.session_name)
        if not os.path.exists(log_path):
            await ctx.send(f"No session log to generate chatlog from.")
            return
        wrapup_files = await create_wrapup_from_log_entries(
            await load_log(log_path),
            self.session_name,
            outline=False  # Only generate chatlog, not outline
        )
        print("Chatlog generation complete.")
        files_to_upload = [discord.File(wrapup_files.chatlog_path)]
        print(f"Uploading chatlog to channel {ctx.channel.id}...")
        await ctx.send("📝", files=files_to_upload)
