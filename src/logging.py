import os
import traceback
from typing import List, Literal, Optional
from datetime import datetime
from pydantic import BaseModel
import aiofiles

class LogEntry(BaseModel):
    """Represents a single log entry."""
    user_id: int
    user_name: str
    timestamp: datetime
    content: str
    medium: Optional[Literal["text", "voice"]] = "voice"

def ensure_log_directory_for_file(log_path: str) -> None:
    """Ensure the directory for the given log file exists."""
    directory = os.path.dirname(log_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Created log directory: {directory}")
        except OSError as e:
            print(f"Error creating log directory {directory}: {e}")

async def add_entry(
        *,
        log_path: str,
        content: str,
        user_id: int,
        medium: Optional[Literal["text", "voice"]] = None,
        user_name: Optional[str] = None,
        timestamp: Optional[datetime] = None
) -> None:
    """Appends a single log entry to the specified log file path."""
    if not content:
        return

    ensure_log_directory_for_file(log_path)
    try:
        ent = LogEntry(
            user_id=user_id,
            user_name=user_name,
            timestamp=(timestamp or datetime.now()).isoformat(),
            content=content,
            medium=medium
        )
        async with aiofiles.open(log_path, "a", encoding="utf-8") as sf:
            await sf.write(ent.model_dump_json() + "\n")
    except IOError as e:
        print(f"Error writing log entry to file {log_path}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error writing log entry to {log_path}: {e} {traceback.format_exc()}")
        raise

async def load_log(log_path: str) -> List[LogEntry]:
    """Loads and returns all log entries from the specified log file as a list of LogEntry objects."""
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
    entries: List[LogEntry] = []
    async with aiofiles.open(log_path, "r", encoding="utf-8") as infile:
        async for line in infile:
            try:
                ent = LogEntry.model_validate_json(line)
                entries.append(ent)
            except Exception:
                continue
    return entries

