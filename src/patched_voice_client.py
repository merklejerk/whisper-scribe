import discord
import nacl
import traceback

class PatchedVoiceClient(discord.VoiceClient):
    """A subclass of discord.VoiceClient to mitigate instability in py-cord's 2.6.1 `VoiceClient` impl."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._crashed = False

    # TODO: Doing some experimation here...
    def unpack_audio(self, data):
        if self._crashed:
            print('still receiving...', data.hex())
        try:
            return super().unpack_audio(data)
        except nacl.exceptions.CryptoError as e:
            self._crashed = True
            print(f"Suppressed VoiceClient CryptoError: {e}")
            traceback.print_exc()
            print(self.mode, data.hex())
            print(f"Restarting opus for {self.channel.name}...")
            if self.decoder.is_alive():
                self.decoder.stop()
            self.decoder = discord.opus.DecodeManager(self)
            self.decoder.start()
