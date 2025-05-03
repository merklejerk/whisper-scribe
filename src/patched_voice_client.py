import discord
import nacl
import traceback
import asyncio
import threading

class PatchedVoiceClient(discord.VoiceClient):
    """A subclass of discord.VoiceClient to mitigate instability in py-cord's 2.6.1 `VoiceClient` impl."""
    _recording_callback: callable
    _recording_args: tuple[any]
    _restart_recording_future: asyncio.Future[None] | None
    _recording_state_lock: asyncio.Lock

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recording_callback = None
        self._recording_args = None
        self._restart_recording_future = None
        self._recording_state_lock = asyncio.Lock()

    def start_recording(self, *args, **kwargs):
        """ Overridden to acquire a lock before starting the recording. """
        supe = super()
        async def _logic():
            async with self._recording_state_lock:
                supe.start_recording(*args, **kwargs)
        asyncio.create_task(_logic())
    
    def stop_recording(self, *args, **kwargs):
        """ Overridden to acquire a lock before stopping the recording. """
        supe = super()
        async def _logic():
            async with self._recording_state_lock:
                supe.stop_recording(*args, **kwargs)
        asyncio.create_task(_logic())
        
    def recv_audio(self, sink, callback, *args):
        """ Overridden to gracefully handle exceptions in the recv thread. """
        self._recording_callback = callback
        self._recording_args = args
        return super().recv_audio(sink, callback, *args)

    def unpack_audio(self, data):
        """ Overrident because this fails regularly with IndexError and CryptoError. """
        if len(data) == 0:
            return
        try:
            super().unpack_audio(data)
        except IndexError as e:
            # This is recoverable. Just ignore it.
            print(f"Suppressed VoiceClient IndexError: {e}, data: {data.hex()}")
        except nacl.exceptions.CryptoError as e:
            print(f"Caught CryptoError: {e}")
            traceback.print_exc()
            # Not recoverable. Restart the recording thread.
            self._restart_recording_future = asyncio.run_coroutine_threadsafe(
                self._restart_recording(threading.current_thread()),
                self.loop,
            )
    
    async def _restart_recording(self, recv_thread: threading.Thread):
        """ Restart the recording thread. """
        async with self._recording_state_lock:
            if self.recording:
                print("Restarting recording thread...")
                sink = self.sink
                super().stop_recording()
                await asyncio.to_thread(recv_thread.join, timeout=5)
                super().start_recording(sink, self._recording_callback, *self._recording_args)

    # def _decrypt_xsalsa20_poly1305_lite(self, header, data):
    #     # 0002004600019e0c3137332e3136382e37342e31333700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009f8e