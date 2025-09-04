from __future__ import annotations

from typing import Optional, Literal, Union, List, Any
from pydantic import BaseModel, Field

PROTO_VERSION = 1

# Shared base
class BaseMessage(BaseModel):
	v: int = Field(default=PROTO_VERSION)

# Node -> Python messages
class PCMFormat(BaseModel):
	sr: int
	channels: int
	sample_width: int

class AudioSegmentMessage(BaseMessage):
	type: Literal['audio.segment'] = 'audio.segment'
	id: str
	index: int
	pcm_format: PCMFormat
	started_ts: float
	capture_ts: float
	data_b64: str
	# Optional per-job ASR prompt override (Whisper system prompt)
	prompt: Optional[str] = None

NodeToPy = Union[AudioSegmentMessage]

# Python -> Node messages
class TranscriptionMessage(BaseMessage):
	type: Literal['transcription'] = 'transcription'
	id: str
	text: str
	capture_ts: float
	end_ts: float

class ErrorMessage(BaseMessage):
	type: Literal['error'] = 'error'
	code: str
	message: str
	details: Optional[str] = None

PyToNode = Union[TranscriptionMessage, ErrorMessage]

Incoming = NodeToPy
Outgoing = PyToNode