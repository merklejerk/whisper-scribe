from __future__ import annotations

import os, tomli
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from urllib.parse import urlparse
from dotenv import load_dotenv

# Project root (py/src/config.py -> py -> root)
ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / '.env'
CONFIG_FILE = ROOT / 'config.toml'

if ENV_FILE.exists():
	load_dotenv(dotenv_path=ENV_FILE, override=True)
else:
	load_dotenv(override=True)

@dataclass(frozen=True)
class WhisperCfg:
	model: str = 'openai/whisper-small.en'
	logprob_threshold: float = -1.0
	no_speech_threshold: float = 0.2
	prompt: str = ''

@dataclass(frozen=True)
class VoiceCfg:
	silence_threshold_seconds: float = 1.25
	vad_threshold: float = 0.75
	max_speech_buf_seconds: int = 0

@dataclass(frozen=True)
class WrapupCfg:
	model: str = 'gpt-4o-mini'
	tips: tuple[str, ...] = ()
	prompt: str = ''
	temperature: float = 0.05
	max_output_tokens: int = 10240

@dataclass(frozen=True)
class RefinerCfg:
	model: Optional[str] = None
	context_log_lines: int = 10
	temperature: float = 0.5
	timeout: float = 5.0
	prompt: Optional[str] = None

@dataclass(frozen=True)
class NetCfg:
	ai_service_url: str = 'ws://localhost:8771'
	chunk_ms: int = 1000
	host: str = '0.0.0.0'
	port: int = 8771

@dataclass(frozen=True)
class DiscordCfg:
	allowed_commanders: tuple[int, ...] = ()

@dataclass(frozen=True)
class AppConfig:
	discord: DiscordCfg
	net: NetCfg
	whisper: WhisperCfg
	voice: VoiceCfg
	wrapup: WrapupCfg
	refiner: RefinerCfg
	username_map: Dict[str, str]
	phrase_map: Dict[str, str]
	openai_api_key: Optional[str]
	gemini_api_key: Optional[str]
	device: str

def _coerce_int_list(val) -> tuple[int, ...]:
	if not val:
		return ()
	try:
		return tuple(int(x) for x in val)
	except Exception:
		return ()

@lru_cache
def _raw_toml() -> dict:
	if CONFIG_FILE.exists():
		with CONFIG_FILE.open('rb') as f:
			return tomli.load(f)
	return {}

@lru_cache
def load_app_config() -> AppConfig:
	raw = _raw_toml()
	discord = raw.get('discord', {})
	net = raw.get('net', {})
	whisper = raw.get('whisper', {})
	voice = raw.get('voice', {})
	wrapup = raw.get('wrapup', {})
	refiner = raw.get('refiner', {})
	username_map = dict(raw.get('username_map', {}) or {})
	phrase_map = dict(raw.get('phrase_map', {}) or {})

	# Only secrets (e.g. OPENAI_API_KEY) come from environment; all functional knobs come from config.toml.
	whisper_cfg = WhisperCfg(
		model=whisper.get('model', WhisperCfg.model),
		logprob_threshold=float(whisper.get('logprob_threshold', WhisperCfg.logprob_threshold)),
		no_speech_threshold=float(whisper.get('no_speech_threshold', WhisperCfg.no_speech_threshold)),
		prompt=whisper.get('prompt', WhisperCfg.prompt),
	)
	voice_cfg = VoiceCfg(
		silence_threshold_seconds=float(voice.get('silence_threshold_seconds', VoiceCfg.silence_threshold_seconds)),
		vad_threshold=float(voice.get('vad_threshold', VoiceCfg.vad_threshold)),
		max_speech_buf_seconds=int(voice.get('max_speech_buf_seconds', VoiceCfg.max_speech_buf_seconds)),
	)
	wrapup_cfg = WrapupCfg(
		model=wrapup.get('model', WrapupCfg.model),
		tips=tuple(wrapup.get('tips', [])),
		prompt=wrapup.get('prompt', WrapupCfg.prompt),
		temperature=float(wrapup.get('temperature', WrapupCfg.temperature)),
		max_output_tokens=int(wrapup.get('max_output_tokens', WrapupCfg.max_output_tokens)),
	)
	refiner_cfg = RefinerCfg(
		model=refiner.get('model'),
		context_log_lines=int(refiner.get('context_log_lines', RefinerCfg.context_log_lines)),
		temperature=float(refiner.get('temperature', RefinerCfg.temperature)),
		timeout=float(refiner.get('timeout', RefinerCfg.timeout)),
		prompt=refiner.get('prompt'),
	)
	net_cfg = NetCfg(
		ai_service_url=net.get('ai_service_url', NetCfg.ai_service_url),
		chunk_ms=int(net.get('chunk_ms', NetCfg.chunk_ms)),
		# derive host/port from ai_service_url so callers can use them directly from AppConfig
		host=_host_port_from_url(net.get('ai_service_url', NetCfg.ai_service_url))[0],
		port=_host_port_from_url(net.get('ai_service_url', NetCfg.ai_service_url))[1],
	)
	discord_cfg = DiscordCfg(
		allowed_commanders=_coerce_int_list(discord.get('allowed_commanders')),
	)
	return AppConfig(
		discord=discord_cfg,
		net=net_cfg,
		whisper=whisper_cfg,
		voice=voice_cfg,
		wrapup=wrapup_cfg,
		refiner=refiner_cfg,
		username_map=username_map,
		phrase_map=phrase_map,
		openai_api_key=os.getenv('OPENAI_API_KEY'),
		gemini_api_key=os.getenv('GEMINI_API_KEY'),
	device=os.getenv('DEVICE', 'auto'),  # auto = choose best available (cuda/mps/rocm/cpu)
	)

def _host_port_from_url(url: str) -> Tuple[str, int]:
	"""Parse a websocket URL or host:port string and return (host, port).

	Accepts values like 'ws://host:port', 'wss://host:port', or 'host:port'.
	Falls back to ('0.0.0.0', 8771) when input is falsy or missing parts.
	"""
	if not url:
		return ('0.0.0.0', 8771)
	if '://' not in url:
		url = 'ws://' + url
	parsed = urlparse(url)
	host = parsed.hostname or '0.0.0.0'
	port = parsed.port or 8771
	return (host, port)
