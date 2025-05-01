import discord
import nacl
import traceback

class PatchedVoiceClient(discord.VoiceClient):
    """A subclass of discord.VoiceClient to mitigate instability in py-cord's 2.6.1 `VoiceClient` impl."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def unpack_audio(self, data):
        try:
            return super().unpack_audio(data)
        except nacl.exceptions.CryptoError as e:
            print(f"Suppressed VoiceClient CryptoError: {e}")
            traceback.print_exc()
        return None
