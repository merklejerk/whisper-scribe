import discord
import nacl
import traceback

class PatchedVoiceClient(discord.VoiceClient):
    """A subclass of discord.VoiceClient to mitigate instability in py-cord's 2.6.1 `VoiceClient` impl."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._crashed = False

    # Called on a recv thread spun up by `start_recording()`.
    def unpack_audio(self, data):
        if self._crashed:
            print('still receiving...', data.hex())
        try:
            super().unpack_audio(data)
        except IndexError as e:
            print(self.mode, data.hex())
            print(f"Suppressed VoiceClient IndexError: {e}")
            traceback.print_exc()
        except nacl.exceptions.CryptoError as e:
            # 0002004600019e0c3137332e3136382e37342e31333700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000009f8e
            self._crashed = True
            print(f"Suppressed VoiceClient CryptoError: {e}")
            traceback.print_exc()
            print(self.mode, data.hex())

    # def _decrypt_xsalsa20_poly1305_lite(self, header, data):
    #     try:
    #         return super()._decrypt_xsalsa20_poly1305_lite(header, data)
    #     except nacl.exceptions.CryptoError as e:
    #         print(f"Suppressed VoiceClient CryptoError: {e}")
    #         traceback.print_exc()
    #         print(self.mode, data.hex())
    #         print(f"Restarting opus for {self.channel.name}...")
    #         if self.decoder.is_alive():
    #             self.decoder.stop()
    #         self.decoder = discord.opus.DecodeManager(self)
    #         self.decoder.start()
