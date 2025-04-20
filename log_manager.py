import os
import traceback
import json
from typing import List, Optional

class LogManager:
    """Manages logging by writing entries directly to channel-specific files."""

    # Removed save_interval from constructor
    # log_directory is where channel-specific logs will be stored
    def __init__(self, log_directory: str = "logs"):
        # Removed _session_log, _log_lock, _log_save_task
        self._log_directory = log_directory
        # Ensure the base log directory exists
        try:
            if not os.path.exists(self._log_directory):
                os.makedirs(self._log_directory)
                print(f"Created base log directory: {self._log_directory}")
        except OSError as e:
            print(f"Error creating base log directory {self._log_directory}: {e}")
            # Decide how to handle this - maybe raise an exception or log to a default file?
            # For now, we'll print the error and continue; writing might fail later.

    # --- Public Methods --- #

    # Modified to accept channel_id and write immediately
    async def add_log_entry(self, entry: str, channel_id: Optional[int], user_id: Optional[int] = None, user_name: Optional[str] = None) -> None:
        """Writes a log entry directly to the appropriate channel log file."""
        await self._write_entry_to_file(entry, channel_id, user_id, user_name)

    # --- Internal Methods --- #


    # Modified to write a single entry to a channel-specific file
    async def _write_entry_to_file(self, entry: str, channel_id: int, user_id: int , user_name: str) -> None:
        """Appends a single log entry to the appropriate channel file."""
        if not entry:
            return

        # Determine filename based on channel_id
        filename = os.path.join(self._log_directory, f"channel_{channel_id}.ndjson")
        try:
            # Append the single entry to the log file
            # Use 'a' mode for appending, create file if it doesn't exist
            # Use async file I/O if available/needed, but standard I/O is often fine here
            # as it's called from the bot's async loop, but file I/O can still block.
            # For simplicity, using standard blocking I/O for now.
            # Consider using aiofiles if this becomes a bottleneck.
            # Append a JSON record to the session NDJSON log, including user info
            record = {"channel_id": channel_id, "user_id": user_id, "user_name": user_name, "entry": entry}
            with open(filename, "a", encoding="utf-8") as sf:
                sf.write(json.dumps(record) + "\n")
        except IOError as e:
            print(f"Error writing log entry to file {filename}: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error writing log entry to {filename}: {e} {traceback.format_exc()}")
            raise

