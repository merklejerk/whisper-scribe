import discord
from discord.ext.commands import Bot,Command
import asyncio
import datetime
import random
from typing import Optional, Any
import asyncio
import traceback
import re

import src.config as config
from .processing import process_entries
from .logging import add_entry as add_log_entry, load_log
from .audio_processor import AudioProcessor
from .sink import SilenceSink, BotMetadata
from .patched_voice_client import PatchedVoiceClient

def get_session_log_path(session_name: str) -> str:
    """Returns the path to the session log file."""
    return f"logs/{session_name}.ndjson"

# Reconnect settings
MAX_RECONNECT_ATTEMPTS = 5
INITIAL_RECONNECT_DELAY = 2.0

class DiscordBot(object):
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
        self._current_vc: Optional[discord.VoiceClient] = None
        self._current_sink: Optional[SilenceSink] = None
        self.voice_channel_id: int = voice_channel_id
        # Use a timestamp for session id if not provided.
        self.session_name = session_name or datetime.datetime.now().strftime("%d-%m-%Y_%H%M")

        # Create the output queue, specifying the generic type
        self._transcription_queue: asyncio.Queue = asyncio.Queue()
        # Create the capture queue for raw audio samples
        self._capture_queue: asyncio.Queue = asyncio.Queue()

        # Instantiate AudioProcessor, specifying the generic type
        self._audio_processor: AudioProcessor[BotMetadata] = AudioProcessor(
            output_queue=self._transcription_queue,
            model_name=config.WHISPER_MODEL,
            device=device
        )

        # Task for consuming transcriptions
        self._transcription_consumer_task: Optional[asyncio.Task] = None
        # Task for consuming capture queue and submitting jobs
        self._capture_consumer_task: Optional[asyncio.Task] = None

        # Regiser commands.
        self._client.add_command(Command(self._wrapup_command, name='wrapup'))

        # Register listeners.
        self._client.add_listener(self._on_message, 'on_message')
        self._client.add_listener(self._on_ready, 'on_ready')

        # Add the global check
        self._client.add_check(self._command_check)
        
    async def start(self, api_key: str, *, on_ready: Optional[callable] = None):
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
                print('baaa')
                return False
        return True

    async def _connect_and_start_recording(self, voice_channel: discord.VoiceChannel):
        """Connects to the voice channel, creates sink, and starts recording.

        Returns True on success, False on failure.
        Handles updating self._current_vc and self._current_sink.
        Updates self.current_text_channel if a text channel matching the voice channel ID is found.
        Stores the voice channel ID for potential reconnects.
        """
        try:
            print(f"Attempting to join voice channel: {voice_channel.name}...")
            if self._current_vc and self._current_vc.is_connected():
                print("Warning: _connect_and_start_recording called while already connected. Disconnecting first.")
                await self._current_vc.disconnect(force=True)
                self._current_vc = None
                if self._current_sink:
                    await self._current_sink.stop_cleanup()
                    self._current_sink = None

            vc: discord.VoiceClient = await asyncio.wait_for(
                voice_channel.connect(reconnect=True, cls=PatchedVoiceClient),
                timeout=30.0
            )
            self._current_vc = vc
            self.voice_channel_id = voice_channel.id # Store the voice channel ID
            print(f"Successfully joined voice channel: {vc.channel.name} (ID: {self.voice_channel_id})")

            # Instantiate SilenceSink, passing the AudioProcessor instance
            sink = SilenceSink(
                vc=vc,
                capture_queue=self._capture_queue, # Pass capture queue
            )
            self._current_sink = sink

            vc.start_recording(sink, self._finished_callback, vc.channel)
            print("Started recording...")
            # Start audio processor thread
            self._audio_processor.start()
            # Start capture consumer task
            if self._capture_consumer_task is None or self._capture_consumer_task.done():
                print("Starting capture consumer task...")
                self._capture_consumer_task = asyncio.create_task(self._process_capture_queue())
            # Start sink cleanup loop
            await sink.start_cleanup()
            return True

        except asyncio.TimeoutError:
            print(f"Error: Timed out while trying to connect to voice channel {voice_channel.name}.")
            if self._current_vc and self._current_vc.is_connected():
                await self._current_vc.disconnect(force=True)
            self._current_vc = None
            self._current_sink = None
            return False
        except discord.errors.ClientException as e:
            print(f"Error connecting to voice channel: {e}")
            self._current_vc = None
            self._current_sink = None
            return False
        except Exception as e:
            if self._current_vc and self._current_vc.is_connected():
                await self._current_vc.disconnect(force=True)
            self._current_vc = None
            self._current_sink = None
            raise 

    # TODO: Clean up this ugly AI slop.
    async def attempt_reconnect(self, guild: discord.Guild):
        """Attempts to reconnect to the last connected voice channel after a disconnect."""
        print("Attempting to reconnect...")

        # --- Cleanup existing state --- #
        if self._current_sink:
            await self._current_sink.stop_cleanup()
            self._current_sink = None
        if self._current_vc and self._current_vc.is_connected():
            try:
                await self._current_vc.disconnect(force=True)
                print("Forcibly disconnected old voice client.")
            except Exception as e:
                print(f"Error disconnecting old voice client: {e}")
            self._current_vc = None
        # --- End Cleanup --- #

        if self.voice_channel_id is None:
            print("Reconnect failed: No last voice channel ID stored.")
            return

        # --- Get Voice Channel Object --- #
        target_voice_channel = guild.get_channel(self.voice_channel_id)

        if not isinstance(target_voice_channel, discord.VoiceChannel):
            print(f"Reconnect failed: Last voice channel {self.voice_channel_id} not found or not a voice channel in guild {guild.name}.")
            return
        # --- End Get Voice Channel Object --- #

        # Text channel retrieval logic removed here - will be handled inside _connect_and_start_recording
        print(f"Attempting reconnect to Voice: '{target_voice_channel.name}' (ID: {target_voice_channel.id})")

        # Set the text channel to None initially for reconnect.
        # _connect_and_start_recording will try to find the matching text channel.

        delay = INITIAL_RECONNECT_DELAY
        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            print(f"Reconnect attempt {attempt + 1}/{MAX_RECONNECT_ATTEMPTS}: Trying connection logic...")
            try:
                # Pass the retrieved voice channel object
                # _connect_and_start_recording will now handle setting self.current_text_channel
                success = await self._connect_and_start_recording(target_voice_channel)
                if success:
                    print("Successfully reconnected and restarted recording.")
                    # Send confirmation to the text channel matching the voice channel ID
                    text_channel = guild.get_channel(self.voice_channel_id)
                    if isinstance(text_channel, discord.TextChannel):
                        await text_channel.send(f"Successfully reconnected to voice channel '{target_voice_channel.name}'.")
                    else:
                        print("Reconnect successful, but no matching text channel found for confirmation message.")
                    return # Exit reconnect loop on success
                else:
                    print(f"Reconnect attempt {attempt + 1} failed during connection/startup phase.")

            except Exception as e:
                print(f"Unexpected error during reconnect attempt {attempt + 1} (calling connect_and_start): {e} {traceback.format_exc()}")

            # Wait before next attempt
            jitter = random.uniform(0.5, 1.5)
            wait_time = delay * jitter
            print(f"Waiting {wait_time:.2f} seconds before next reconnect attempt...")
            await asyncio.sleep(wait_time)
            delay = min(delay * 2, 60) # Exponential backoff with cap

        print(f"Failed to reconnect after {MAX_RECONNECT_ATTEMPTS} attempts.")

    async def _finished_callback(self, sink: SilenceSink, channel: discord.VoiceChannel, *args: Any) -> None:
        """Called by discord.py when recording unexpectedly stops."""
        print(f"Warning: Recording stopped unexpectedly in channel {channel.name} (ID: {channel.id}).")

        # Store the channel ID where recording stopped, in case it's different from voice_channel_id
        stopped_channel_id = channel.id

        old_sink = sink # Local reference
        await old_sink.stop_cleanup()

        is_connected_to_channel = False
        if self._current_vc and self._current_vc.is_connected() and self._current_vc.channel.id == stopped_channel_id:
            is_connected_to_channel = True

        if is_connected_to_channel:
            print("Bot seems to still be in the correct channel, but recording stopped. Not attempting automatic reconnect.")
            # Consider if you want to try restarting recording here without full disconnect/reconnect
            return

        print("Assuming disconnection from target channel, scheduling reconnection attempt...")
        # Ensure we have a guild object to pass to attempt_reconnect
        if channel and channel.guild:
             # TODO: Add a lock/flag to prevent concurrent reconnect attempts
             asyncio.create_task(self.attempt_reconnect(channel.guild))
        else:
             print("Error: Cannot attempt reconnect, guild information unavailable from channel.")

    # Called when the bot successfully connects and is ready.
    async def _on_ready(self) -> None:
        """Called when the bot successfully connects and is ready."""
        print(f'Logged in as {self._client.user.name} (ID: {self._client.user.id})')
        print('------')
        
        # Start the transcription consumer task
        if self._transcription_consumer_task is None or self._transcription_consumer_task.done():
            print("Starting transcription consumer task...")
            self._transcription_consumer_task = asyncio.create_task(self._process_transcription_queue())
        # Start the capture consumer task if not already
        if self._capture_consumer_task is None or self._capture_consumer_task.done():
            print("Starting capture consumer task...")
            self._capture_consumer_task = asyncio.create_task(self._process_capture_queue())

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
        print(f"{user_name if user_name else user_id}: {content}")
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
            except:
                raise
            finally:
                self._transcription_queue.task_done()

    async def _process_capture_queue(self):
        """Consumes raw audio from capture queue and submits to AudioProcessor."""
        print("Capture consumer task started.")
        while True:
            metadata, audio_data = await self._capture_queue.get()
            try:
                self._audio_processor.submit_job(metadata, audio_data)
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
                user_name=message.author.user_name,
                content=message.content,
                timestamp=datetime.datetime.now(datetime.timezone.utc)
            )

    async def close_and_cleanup(self) -> None:
        """Cleanup logic to run when the bot is shutting down."""
        print("Bot is shutting down...")

        # Cancel the transcription consumer task first
        if self._transcription_consumer_task and not self._transcription_consumer_task.done():
            print("Stopping transcription consumer task...")
            self._transcription_consumer_task.cancel()
            try:
                await self._transcription_consumer_task
            except asyncio.CancelledError:
                print("Transcription consumer task successfully cancelled.") # Expected
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
        # Stop sink and processor
        if self._current_sink:
            await self._current_sink.stop_cleanup()
        else:
            # Ensure processor is stopped even if sink wasn't active
            self._audio_processor.stop()

        if self._current_vc and self._current_vc.is_connected():
            print("Disconnecting from voice channel...")
            # Sink cleanup already handled above
            await self._current_vc.disconnect(force=True)
            print("Disconnected.")

        # Close the bot itself if not already closed
        if not self._client.is_closed():
             await self._client.close() # Use self.close()
        print("Discord client closed.")
        print("Shutdown complete.")

    async def join_channel(self, channel_id: int) -> bool:
        """Helper method to join a voice channel by its ID."""
        channel = self._client.get_channel(channel_id)
        if isinstance(channel, discord.VoiceChannel):
            return await self._connect_and_start_recording(channel)
        print(f"Error: Channel ID {channel_id} is not a voice channel or not found.")
        return False

    async def _wrapup_command(self, ctx: Any):
        """Process the current session log and generate a D&D 5e outline."""
        await ctx.send("Processing session, please wait...")
        await process_entries(
            await load_log(get_session_log_path(self.session_name)),
            self.session_name,
        )
