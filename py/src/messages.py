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

class AudioChunkMessage(BaseMessage):
	type: Literal['audio.chunk'] = 'audio.chunk'
	user_id: str
	index: int
	pcm_format: PCMFormat
	started_ts: float
	capture_ts: float
	data_b64: str

class WrapupLogEntry(BaseModel):
	"""Wire shape sent by Node for wrapup logs."""
	user_name: str
	start_ts: float
	end_ts: float
	text: str
	user_id: str

class WrapupRequestMessage(BaseMessage):
	type: Literal['wrapup.request'] = 'wrapup.request'
	session_name: str
	log_entries: List[WrapupLogEntry]
	request_id: str

NodeToPy = Union[AudioChunkMessage, WrapupRequestMessage]

# Python -> Node messages
class TranscriptionMessage(BaseMessage):
	type: Literal['transcription'] = 'transcription'
	user_id: str
	text: str
	capture_ts: float
	end_ts: float

class WrapupResponseMessage(BaseMessage):
	type: Literal['wrapup.response'] = 'wrapup.response'
	outline: str
	request_id: str

class ErrorMessage(BaseMessage):
	type: Literal['error'] = 'error'
	code: str
	message: str
	details: Optional[str] = None

PyToNode = Union[TranscriptionMessage, ErrorMessage, WrapupResponseMessage]

Incoming = NodeToPy
Outgoing = PyToNode