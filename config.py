import os
from dotenv import load_dotenv
from typing import Optional, Final, List

# Load environment variables from .env file
load_dotenv()

# Discord Configuration
DISCORD_TOKEN: Final[Optional[str]] = os.getenv("DISCORD_TOKEN")
# Whitelisted Guild IDs (comma-separated string from env, parsed to list of ints)
_allowed_guilds_str: Optional[str] = os.getenv("ALLOWED_GUILDS")
ALLOWED_GUILD_IDS: Final[Optional[List[int]]] = (
    [int(gid.strip()) for gid in _allowed_guilds_str.split(',')]
    if _allowed_guilds_str
    else None # Allow all guilds if not set
)

# Whisper Configuration
WHISPER_MODEL: Final[str] = os.getenv("WHISPER_MODEL", "openai/whisper-large-v3") # Default to large model

# Voice Handling Configuration
SILENCE_THRESHOLD_SECONDS: Final[float] = 2 # Seconds of silence before processing audio

# Validation
def validate_config() -> bool:
    """Checks if essential configuration variables are set."""
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in environment variables or .env file.")
        return False
    # Optional: Add validation/logging for ALLOWED_GUILD_IDS if needed
    if ALLOWED_GUILD_IDS is None:
        print("Warning: ALLOWED_GUILDS not set. Bot will respond in all guilds.")
    else:
        print(f"Bot commands restricted to guilds: {ALLOWED_GUILD_IDS}")
    return True

print(f"Whisper model: {WHISPER_MODEL}")
