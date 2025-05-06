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

# Load config.toml configuration (non-secret values)
CONFIG_TOML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.toml")
if os.path.exists(CONFIG_TOML_PATH):
    with open(CONFIG_TOML_PATH, "rb") as f:
        bot_config = tomli.load(f)
else:
    bot_config = {}

# Discord Configuration
DISCORD_TOKEN: Final[Optional[str]] = os.getenv("DISCORD_TOKEN")
# Whitelisted Guild IDs (comma-separated string from env, parsed to list of ints)
ALLOWED_GUILD_IDS: Final[Optional[list[int]]] = bot_config.get("discord", {}).get("allowed_guild_ids")
# Whitelisted command givers.
ALLOWED_COMMANDERS: Final[Optional[list[int]]] = bot_config.get("discord", {}).get("allowed_commanders")

# Whisper Configuration
WHISPER_MODEL: Final[str] = bot_config.get("whisper", {}).get("model", "openai/whisper-small.en")
WHISPER_LOGPROB_THRESHOLD: Final[float] = bot_config.get("whisper", {}).get("logprob_threshold", -1.0)
WHISPER_NO_SPEECH_THRESHOLD: Final[float] = bot_config.get("whisper", {}).get("no_speech_threshold", 0.2)
WHISPER_PROMPT: Final[str] = bot_config.get("whisper", {}).get("prompt", "")

# Voice Handling Configuration
SILENCE_THRESHOLD_SECONDS: Final[float] = bot_config.get("voice", {}).get("silence_threshold_seconds", 2)
VAD_THRESHOLD: Final[float] = bot_config.get("voice", {}).get("vad_threshold", 0.5)
# Maximum allowed speech buffer length in seconds
MAX_SPEECH_BUF_SECONDS: Final[int] = bot_config.get("voice", {}).get("max_speech_buf_seconds", 0)

OPENAI_API_KEY: Final[str] = os.getenv("OPENAI_API_KEY")

# OpenAI GPT Model for summarization
OPENAI_GPT_MODEL: Final[str] = bot_config.get("wrapup", {}).get("model", "gpt-4o-mini")
TIPS: Final[Optional[list[str]]] = bot_config.get("wrapup", {}).get("tips")
# Remapping of phrases in the chatlog.
PHRASE_MAP: Final[Optional[list[str]]] = bot_config.get("wrapup.phrase_map", {})
# Username remapping for chatlog
USERNAME_MAP: Final[dict[str, str]] = bot_config.get("wrapup.username_map", {})

# Refiner Configuration
# Default to None if not specified in config
REFINER_OLLAMA_MODEL: Final[Optional[str]] = bot_config.get("refiner", {}).get("model") 
REFINER_CONTEXT_LOG_LINES: Final[int] = bot_config.get("refiner", {}).get("context_log_lines", 10)


# Validation
def validate_config() -> bool:
    """Checks if essential configuration variables are set."""
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in environment variables or .env file.")
        return False
    return True

if not validate_config():
    sys.exit("Exiting due to missing configuration.")
