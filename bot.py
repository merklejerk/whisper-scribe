import discord
from discord.ext import commands
import asyncio
import datetime
import sys
import random
from typing import Optional, Any, Tuple
from sink import SilenceSink, BotMetadata
from log_manager import LogManager
from audio_processor import AudioProcessor, TranscriptionResult

# Import configuration and validation
import config
if not config.validate_config():
    sys.exit("Exiting due to missing configuration.")

import argparse  # For CLI argument parsing

# Global check function for allowed guilds
async def is_in_allowed_guild(ctx: commands.Context) -> bool:
    """Checks if the command is invoked in an allowed guild."""
    if config.ALLOWED_GUILD_IDS is None:
        return True # Allow all guilds if not configured
    if ctx.guild is None:
        # Optionally send a message or log that commands can't be used in DMs
        # await ctx.send("Commands can only be used in allowed servers.")
        print(f"Command '{ctx.command}' blocked in DM by user {ctx.author.id}")
        return False # Cannot run commands in DMs if whitelist is active
    is_allowed = ctx.guild.id in config.ALLOWED_GUILD_IDS
    if not is_allowed:
        print(f"Command '{ctx.command}' blocked in guild {ctx.guild.id} by user {ctx.author.id}")
    return is_allowed

# Reconnect settings
MAX_RECONNECT_ATTEMPTS = 5
INITIAL_RECONNECT_DELAY = 2.0


# Inherit directly from commands.Bot
class STTBot(commands.Bot):
    def __init__(self):
        # Setup Discord client and internal state
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        # Call commands.Bot constructor (removed guild_ids)
        super().__init__(
            command_prefix='!',
            intents=intents,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self.current_vc: Optional[discord.VoiceClient] = None
        self.current_sink: Optional[SilenceSink] = None
        self.last_voice_channel_id: Optional[int] = None # Store the ID of the last connected voice channel

        # Instantiate LogManager
        self.log_manager = LogManager()

        # Create the output queue, specifying the generic type
        self.transcription_queue: asyncio.Queue = asyncio.Queue()
        # Create the capture queue for raw audio samples
        self.capture_queue: asyncio.Queue = asyncio.Queue()

        # Instantiate AudioProcessor, specifying the generic type
        self.audio_processor: AudioProcessor[BotMetadata] = AudioProcessor(
            output_queue=self.transcription_queue,
            model_name=config.WHISPER_MODEL
        )

        # Task for consuming transcriptions
        self._transcription_consumer_task: Optional[asyncio.Task] = None
        # Task for consuming capture queue and submitting jobs
        self._capture_consumer_task: Optional[asyncio.Task] = None

        # Manually add the command
        self.add_command(commands.Command(self._join_command, name='join'))

        # Add the global check
        self.add_check(is_in_allowed_guild)

    # Command method (decorator removed)
    async def _join_command(self, ctx: commands.Context):
        print(f"Join command invoked by {ctx.author} in {ctx.channel}")
        # Ensure author is in a voice channel
        voice_state = ctx.author.voice
        if not (voice_state and voice_state.channel):
            await ctx.send("You need to be in a voice channel for me to join.")
            return

        # Attempt to connect and record
        success = await self._connect_and_start_recording(voice_state.channel)
        if success:
            # Send confirmation to the text channel matching the voice channel ID
            text_channel = self.get_channel(self.last_voice_channel_id)
            if isinstance(text_channel, discord.TextChannel):
                await text_channel.send(f"Joined voice channel '{voice_state.channel.name}' and started recording.")
            else:
                await ctx.channel.send(f"Joined voice channel '{voice_state.channel.name}' and started recording.")
        else:
            # Send failure message to invoking channel
            await ctx.channel.send(f"Failed to join voice channel '{voice_state.channel.name}'.")


    async def _connect_and_start_recording(self, voice_channel: discord.VoiceChannel):
        """Connects to the voice channel, creates sink, and starts recording.

        Returns True on success, False on failure.
        Handles updating self.current_vc and self.current_sink.
        Updates self.current_text_channel if a text channel matching the voice channel ID is found.
        Stores the voice channel ID for potential reconnects.
        """
        try:
            print(f"Attempting to join voice channel: {voice_channel.name}...")
            if self.current_vc and self.current_vc.is_connected():
                print("Warning: _connect_and_start_recording called while already connected. Disconnecting first.")
                await self.current_vc.disconnect(force=True)
                self.current_vc = None
                if self.current_sink:
                    await self.current_sink.stop_cleanup()
                    self.current_sink = None

            vc: discord.VoiceClient = await asyncio.wait_for(voice_channel.connect(), timeout=30.0)
            self.current_vc = vc
            self.last_voice_channel_id = voice_channel.id # Store the voice channel ID
            print(f"Successfully joined voice channel: {vc.channel.name} (ID: {self.last_voice_channel_id})")

            # No longer track text channels; rely on voice channel ID matching text channel ID

            # Instantiate SilenceSink, passing the AudioProcessor instance
            sink = SilenceSink(
                vc=vc,
                capture_queue=self.capture_queue, # Pass capture queue
            )
            self.current_sink = sink

            vc.start_recording(sink, self._finished_callback, vc.channel)
            print("Started recording...")
            # Start audio processor thread
            self.audio_processor.start()
            # Start capture consumer task
            if self._capture_consumer_task is None or self._capture_consumer_task.done():
                print("Starting capture consumer task...")
                self._capture_consumer_task = asyncio.create_task(self._process_capture_queue())
            # Start sink cleanup loop
            await sink.start_cleanup()
            return True

        except asyncio.TimeoutError:
            print(f"Error: Timed out while trying to connect to voice channel {voice_channel.name}.")
            if self.current_vc and self.current_vc.is_connected():
                await self.current_vc.disconnect(force=True)
            self.current_vc = None
            self.current_sink = None
            return False
        except discord.errors.ClientException as e:
            print(f"Error connecting to voice channel: {e}")
            self.current_vc = None
            self.current_sink = None
            return False
        except Exception as e:
            import traceback
            print(f"An unexpected error occurred during connection/recording start: {e} {traceback.format_exc()}")
            if self.current_vc and self.current_vc.is_connected():
                await self.current_vc.disconnect(force=True)
            self.current_vc = None
            self.current_sink = None
            return False

    async def attempt_reconnect(self, guild: discord.Guild):
        """Attempts to reconnect to the last connected voice channel after a disconnect."""
        print("Attempting to reconnect...")

        # --- Cleanup existing state --- #
        if self.current_sink:
            await self.current_sink.stop_cleanup()
            self.current_sink = None
        if self.current_vc and self.current_vc.is_connected():
            try:
                await self.current_vc.disconnect(force=True)
                print("Forcibly disconnected old voice client.")
            except Exception as e:
                print(f"Error disconnecting old voice client: {e}")
            self.current_vc = None
        # --- End Cleanup --- #

        # --- Check if we have a last channel to reconnect to --- #
        if self.last_voice_channel_id is None:
            print("Reconnect failed: No last voice channel ID stored.")
            return
        # --- End Check --- #

        # --- Get Voice Channel Object --- #
        target_voice_channel = guild.get_channel(self.last_voice_channel_id)

        if not isinstance(target_voice_channel, discord.VoiceChannel):
            print(f"Reconnect failed: Last voice channel {self.last_voice_channel_id} not found or not a voice channel in guild {guild.name}.")
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
                    text_channel = guild.get_channel(self.last_voice_channel_id)
                    if isinstance(text_channel, discord.TextChannel):
                        await text_channel.send(f"Successfully reconnected to voice channel '{target_voice_channel.name}'.")
                    else:
                        print("Reconnect successful, but no matching text channel found for confirmation message.")
                    return # Exit reconnect loop on success
                else:
                    print(f"Reconnect attempt {attempt + 1} failed during connection/startup phase.")

            except Exception as e:
                import traceback
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

        # Store the channel ID where recording stopped, in case it's different from last_voice_channel_id
        stopped_channel_id = channel.id

        old_sink = sink # Local reference
        await old_sink.stop_cleanup()

        is_connected_to_channel = False
        if self.current_vc and self.current_vc.is_connected() and self.current_vc.channel.id == stopped_channel_id:
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

    # Event handler: Called when the bot successfully connects and is ready.
    # The name "on_ready" is automatically recognized by commands.Bot
    async def on_ready(self) -> None:
        """Called when the bot successfully connects and is ready."""
        print(f'Logged in as {self.user.name} (ID: {self.user.id})') # Use self.user
        print(f'Discord.py version: {discord.__version__}')
        print('------')

        # Start the transcription consumer task
        if self._transcription_consumer_task is None or self._transcription_consumer_task.done():
            print("Starting transcription consumer task...")
            self._transcription_consumer_task = asyncio.create_task(self._process_transcription_queue())
        # Start the capture consumer task if not already
        if self._capture_consumer_task is None or self._capture_consumer_task.done():
            print("Starting capture consumer task...")
            self._capture_consumer_task = asyncio.create_task(self._process_capture_queue())

        print("Ready. Use '!join' in a text channel to have me join your voice channel.")

    async def _add_log_entry(
            self,
            *,
            category: str,
            user_id: int,
            user_name: Optional[str] = None,
            channel_id: int,
            content: str,
            timestamp: datetime.datetime,
        ) -> None:
        user_obj = self.get_user(user_id)
        if user_obj is None:
            user_obj = await self.fetch_user(user_id)
        if user_obj is None:
            user_obj = await self.fetch_user(user_id)
        user_name = user_obj.display_name if user_obj else None
        print(f"{user_name if user_name else user_id}: {content}")
        await self.log_manager.add_entry(
            content=content,
            channel_id=channel_id,
            user_id=user_id,
            user_name=user_name,
            timestamp=timestamp,
        )

    async def _process_transcription_queue(self):
        """Consumes transcription results from the queue and logs them."""
        print("Transcription consumer task started.")
        try:
            while True:
                (user_id, channel_id, capture_time), transcription = await self.transcription_queue.get()
                try:
                    await self._add_log_entry(
                        category="Voice",
                        user_id=user_id,
                        channel_id=channel_id,
                        content=transcription,
                        timestamp=capture_time,
                    )
                except Exception as e:
                    uid_for_log = user_id if 'user_id' in locals() else "unknown user"
                    print(f"Error logging transcription for user {uid_for_log}: {e}")
                finally:
                    self.transcription_queue.task_done()
        except asyncio.CancelledError:
            print("Transcription consumer task cancelled.")
        except Exception as e:
            import traceback
            print(f"Transcription consumer task encountered an error: {e} {traceback.format_exc()}")
        finally:
            print("Transcription consumer task finished.")

    async def _process_capture_queue(self):
        """Consumes raw audio from capture queue and submits to AudioProcessor."""
        print("Capture consumer task started.")
        try:
            while True:
                metadata, audio_data = await self.capture_queue.get()
                try:
                    self.audio_processor.submit_job(metadata, audio_data)
                except Exception as e:
                    print(f"Error submitting job to AudioProcessor: {e}")
                finally:
                    self.capture_queue.task_done()
        except asyncio.CancelledError:
            print("Capture consumer task cancelled.")
        except Exception as e:
            import traceback
            print(f"Capture consumer encountered error: {e} {traceback.format_exc()}")
        finally:
            print("Capture consumer task finished.")

    # Event handler: Called when a message is sent.
    # The name "on_message" is automatically recognized by commands.Bot
    async def on_message(self, message: discord.Message) -> None:
        """Called when a message is sent."""
        if message.author == self.user: # Use self.user
            return
        if message.author.bot:
            return

        # Process commands first - essential for commands.Bot
        await self.process_commands(message) # Use self.process_commands

        # Log non-command messages in the designated text channel
        ctx = await self.get_context(message) # Use self.get_context
        if ctx.valid: # Don't log commands themselves here
            return

        # Only log messages in the text channel with the same ID as the last voice channel
        if message.channel.id == self.last_voice_channel_id:
            # Delegate formatting and logging
            await self._add_log_entry(
                category="Text",
                user_id=message.author.id,
                channel_id=message.channel.id,
                user_name=message.author.display_name,
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
        if self.current_sink:
            await self.current_sink.stop_cleanup()
        else:
            # Ensure processor is stopped even if sink wasn't active
            self.audio_processor.stop()

        if self.current_vc and self.current_vc.is_connected():
            print("Disconnecting from voice channel...")
            # Sink cleanup already handled above
            await self.current_vc.disconnect(force=True)
            print("Disconnected.")

        # Close the bot itself if not already closed
        if not self.is_closed():
             await self.close() # Use self.close()
        print("Discord client closed.")
        print("Shutdown complete.")

    async def join_channel(self, channel_id: int) -> bool:
        """Helper method to join a voice channel by its ID."""
        channel = self.get_channel(channel_id)
        if isinstance(channel, discord.VoiceChannel):
            return await self._connect_and_start_recording(channel)
        print(f"Error: Channel ID {channel_id} is not a voice channel or not found.")
        return False

# Entrypoint: create bot instance and run
async def main() -> None:
    parser = argparse.ArgumentParser(description="STTBot")
    parser.add_argument("-j", "--join", type=int, help="Voice channel ID to auto-join on startup.")
    args = parser.parse_args()

    bot = STTBot()
    # If a channel ID is provided, add a one-shot ready listener to auto-join without storing state
    if args.join:
        channel_id = args.join
        async def on_ready_autojoin():
            # Remove this listener after first run
            bot.remove_listener(on_ready_autojoin, 'on_ready')
            await bot.join_channel(channel_id)
        bot.add_listener(on_ready_autojoin, 'on_ready')

    try:
        print("Starting bot...")
        await bot.start(config.DISCORD_TOKEN)
    except discord.errors.LoginFailure:
        print("Error: Invalid Discord token provided.")
    except discord.errors.PrivilegedIntentsRequired:
        print("Error: Privileged intents (Message Content, Voice States) are not enabled.")
        print("Please enable these intents in the Discord Developer Portal.")
    except Exception as e:
        import traceback
        print(f"An error occurred while running the bot: {e} {traceback.format_exc()}")
    finally:
        # Ensure cleanup runs even if start fails after initialization
        # Check if bot object exists and has the cleanup method before calling
        if 'bot' in locals() and hasattr(bot, 'close_and_cleanup'):
            await bot.close_and_cleanup()

# Run the main async function
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutdown requested via KeyboardInterrupt.")
        # The finally block in main should handle cleanup now.
        pass
