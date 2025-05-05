import discord
import nacl
import traceback
import asyncio
import threading
import errno

class PatchedVoiceClient(discord.VoiceClient):
    """A subclass of discord.VoiceClient to mitigate instability in py-cord's 2.6.1 `VoiceClient` impl."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fuse = kwargs.get("fuse", 5.0)
        # asyncio.run
        # self.loop.call_later(5.0, self.disconnect

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
            # 0002004600019e0c3137332e3136382e37342e31333700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009f8e
            print(f"Encountered CryptError {e}, data: {data.hex()}")
            traceback.print_exc()
            # Not recoverable.
            self.stop_recording()