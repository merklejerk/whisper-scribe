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

class SummarizeRequestMessage(BaseMessage):
	type: Literal['summarize.request'] = 'summarize.request'
	session_name: str
	log_entries: List[Any]
	request_id: str

NodeToPy = Union[AudioChunkMessage, SummarizeRequestMessage]

# Python -> Node messages
class TranscriptionMessage(BaseMessage):
	type: Literal['transcription'] = 'transcription'
	user_id: str
	text: str

class SummarizeResponseMessage(BaseMessage):
	type: Literal['summarize.response'] = 'summarize.response'
	transcript: str
	outline: str
	request_id: str

class ErrorMessage(BaseMessage):
	type: Literal['error'] = 'error'
	code: str
	message: str
	details: Optional[str] = None

PyToNode = Union[TranscriptionMessage, ErrorMessage, SummarizeResponseMessage]

Incoming = NodeToPy
Outgoing = PyToNode

__all__ = [
	'PROTO_VERSION','BaseMessage','PCMFormat','AudioChunkMessage',
	'SummarizeRequestMessage', 'SummarizeResponseMessage',
	'TranscriptionMessage','ErrorMessage','Incoming','Outgoing'
]
