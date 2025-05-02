import discord
import nacl
import traceback
import asyncio
import threading

class PatchedVoiceClient(discord.VoiceClient):
    """A subclass of discord.VoiceClient to mitigate instability in py-cord's 2.6.1 `VoiceClient` impl."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def recv_audio(self, sink, callback, *args):
        """ Overridden to gracefully handle exceptions in the recv thread. """
        try:
            return super().recv_audio(sink, callback, *args)
        except:
            self.sink.cleanup()
            asyncio.run_coroutine_threadsafe(callback(*args), self.loop).result()
            raise

    def unpack_audio(self, data):
        """ Overrident because this fails regularly with IndexError and CryptoError. """
        if len(data) == 0:
            return
        try:
            super().unpack_audio(data)
        except IndexError as e:
            # This is recoverable. Just ignore it.
            print(f"Suppressed VoiceClient IndexError: {e}, data: {data.hex()}")

    def _decrypt_xsalsa20_poly1305_lite(self, header, data):
        # 0002004600019e0c3137332e3136382e37342e31333700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009f8e
        return super()._decrypt_xsalsa20_poly1305_lite(header, data)
        # except nacl.exceptions.CryptoError as e:
        #     # TODO: Try to figure out how to recover from this error.
        #     self._crashed = True
        #     print(f"Suppressed VoiceClient CryptoError: {e}, data: {data.hex()}")
