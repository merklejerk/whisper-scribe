import discord
from discord.ext.commands import Bot,Command
import asyncio
import datetime
from typing import Optional, Any
import asyncio
import re
import os # Added import

import src.config as config
from .wrapup import create_wrapup_from_log_entries, WrapupFiles # Added WrapupFiles import
from .logging import add_entry as add_log_entry, load_log
from .transcriber import Transcriber
from .voice_capture_sink import VoiceCaptureSink, VoiceMetadata
from .patched_voice_client import PatchedVoiceClient

def get_session_log_path(session_name: str) -> str:
    """Returns the path to the session log file."""
    return f"logs/{session_name}.ndjson"

MAX_JOIN_TIMEOUT = 10.0  # seconds

class DiscordBot(object):
    session_name: str
    voice_channel_id: int
    _client: Bot
    _vc: Optional[discord.VoiceClient]
    _sink: VoiceCaptureSink
    _transcription_queue: asyncio.Queue
    _capture_queue: asyncio.Queue
    _transcriber: Transcriber[VoiceMetadata]
    _transcription_consumer_task: Optional[asyncio.Task]
    _capture_consumer_task: Optional[asyncio.Task]

    def __init__(
            self,
            voice_channel_id: int,
            *,
            session_name: Optional[str] = None,
            device: str = "cpu",
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
        # Task for consuming transcriptions
        self._transcription_consumer_task = None
        # Task for consuming capture queue and submitting jobs
        self._capture_consumer_task = None

        # Regiser commands.
        self._client.add_command(Command(self._wrapup_command, name='wrapup'))
        # Register listeners.
        self._client.add_listener(self._on_message, 'on_message')
        self._client.add_listener(self._on_ready, 'on_ready')
        # Add the global check
        self._client.add_check(self._command_check)
        
    async def start(self, api_key: str, *, on_ready: Optional[callable] = None):
        self._transcription_consumer_task = asyncio.create_task(self._process_transcription_queue())
        self._capture_consumer_task = asyncio.create_task(self._process_capture_queue())
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

    async def _connect_and_start_recording(self, voice_channel: discord.VoiceChannel):
        """Connects to the voice channel, creates sink, and starts recording.

        Returns True on success, False on failure.
        Handles updating self._vc and self._sink.
        """
        # Disconnect any existing voice client.
        if self._vc:
            if self._vc.is_connected():
                await self._vc.disconnect(force=True)
            print("Forcibly disconnected old voice client.")
            self._vc = None

        try:
            print(f"Starting transcriber...")
            self._transcriber.start()
            print(f"Attempting to join voice channel: {voice_channel.name}...")
            vc = self._vc = await voice_channel.connect(
                reconnect=True,
                cls=PatchedVoiceClient,
                timeout=MAX_JOIN_TIMEOUT,
            )
            print(f"Successfully joined voice channel: {vc.channel.name} (ID: {self.voice_channel_id})")

            await self._sink.start(vc)
            vc.start_recording(self._sink, self._finished_callback, vc.channel)
        except:
            if self._vc and self._vc.is_connected():
                await self._vc.disconnect(force=True)
            self._vc = None
            raise

    async def _finished_callback(self, channel: discord.VoiceChannel) -> None:
        """Called by py-cord when recording stops."""
        print(f"Warning: Recording stopped unexpectedly in channel {channel.name} (ID: {channel.id})!")
        await self._connect_and_start_recording(channel)

    # Called when the bot successfully connects and is ready.
    async def _on_ready(self) -> None:
        """Called when the bot successfully connects and is ready."""
        print(f'Logged in as {self._client.user.name} (ID: {self._client.user.id})')
        # Automatically connect to the specified voice channel.
        await self.join_channel(self.voice_channel_id)

    async def _add_log_entry(
            self,
            *,
            medium: str,
            user_id: int,
            user_name: str,
            content: str,
            timestamp: datetime.datetime,
        ) -> None:
        print(f"> {user_name if user_name else user_id}: {content}")
        await add_log_entry(
            log_path=get_session_log_path(self.session_name),
            content=content,
            medium=medium,
            user_id=user_id,
            user_name=user_name,
            timestamp=timestamp,
        )

    async def _process_transcription_queue(self):
        """Consumes transcription results from the queue and logs them."""
        print("Transcription consumer task started.")
        while True:
            metadata, transcription = await self._transcription_queue.get()
            try:
                if re.search(r"[a-z0-9]+", transcription):
                    await self._add_log_entry(
                        medium="voice",
                        user_id=metadata.user_id,
                        user_name=metadata.user_name,
                        timestamp=metadata.capture_time,
                        content=transcription,
                    )
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
            await self._add_log_entry(
                medium="text",
                user_id=message.author.id,
                user_name=message.author.name,
                content=message.content,
                timestamp=datetime.datetime.now(datetime.timezone.utc)
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

    async def join_channel(self, channel_id: int) -> bool:
        if self._client.is_closed():
            raise RuntimeError("Not started.")
        
        """Helper method to join a voice channel by its ID."""
        channel = self._client.get_channel(channel_id)
        if isinstance(channel, discord.VoiceChannel):
            return await self._connect_and_start_recording(channel)
        print(f"Error: Channel ID {channel_id} is not a voice channel or not found.")
        return False

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
