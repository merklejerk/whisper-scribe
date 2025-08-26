from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any

def generate_transcript(log_entries: List[Dict[str, Any]], session_name: str) -> str:
	"""Formats session log entries into a plain text transcript file."""
	lines = [f"# Transcript for session: {session_name}\n"]
	for entry in log_entries:
		# The log entries from JS are already formatted Transcription objects
		ts = datetime.fromtimestamp(entry['start_ts']).strftime('%H:%M:%S')
		lines.append(f"[{ts}] {entry['user_name']}: {entry['text']}")
	return "\n".join(lines)

def generate_outline(log_entries: List[Dict[str, Any]], session_name: str) -> str:
	"""Formats session log entries into a Markdown outline."""
	lines = [f"# Outline for session: {session_name}\n"]
	current_user = None
	for entry in log_entries:
		if entry['user_name'] != current_user:
			current_user = entry['user_name']
			lines.append(f"\n## {current_user}\n")
		lines.append(f"- {entry['text']}")
	return "\n".join(lines)
