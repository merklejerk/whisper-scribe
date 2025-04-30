import os
import sys
from dotenv import load_dotenv
from typing import Optional, Final
import tomli

# Load environment variables from .env file
load_dotenv(override=True)

def _split_comma_separated(value: str) -> list[str]:
    """Helper function to split a comma-separated string into a list."""
    return [item.strip() for item in value.split(",")] if value else []

# Load bot.toml configuration (non-secret values)
BOT_TOML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bot.toml")
if os.path.exists(BOT_TOML_PATH):
    with open(BOT_TOML_PATH, "rb") as f:
        bot_config = tomli.load(f)
else:
    bot_config = {}

# Discord Configuration
DISCORD_TOKEN: Final[Optional[str]] = os.getenv("DISCORD_TOKEN")
# Whitelisted Guild IDs (comma-separated string from env, parsed to list of ints)
ALLOWED_GUILD_IDS: Final[Optional[list[int]]] = bot_config.get("discord", {}).get("allowed_guild_ids")
# Whitelisted command givers.
ALLOWED_COMMANDERS: Final[Optional[list[int]]] = bot_config.get("discord", {}).get("allowed_commanders")
# Username remapping for chatlog
USERNAME_MAP: Final[dict[str, str]] = bot_config.get("username_map", {})

# Whisper Configuration
WHISPER_MODEL: Final[str] = bot_config.get("whisper", {}).get("model", "openai/whisper-small.en")

# Voice Handling Configuration
SILENCE_THRESHOLD_SECONDS: Final[float] = bot_config.get("voice", {}).get("silence_threshold_seconds", 2)

OPENAI_API_KEY: Final[str] = os.getenv("OPENAI_API_KEY")

# OpenAI GPT Model for summarization
OPENAI_GPT_MODEL: Final[str] = bot_config.get("llm", {}).get("model", "gpt-4o-mini")

PROPER_NOUNS: Final[Optional[list[str]]] = bot_config.get("llm", {}).get("proper_nouns")

TIPS: Final[Optional[list[str]]] = bot_config.get("llm", {}).get("tips")

# Validation
def validate_config() -> bool:
    """Checks if essential configuration variables are set."""
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in environment variables or .env file.")
        return False
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables or .env file.")
        return False
    return True

if not validate_config():
    sys.exit("Exiting due to missing configuration.")
