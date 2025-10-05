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
ENV_FILE = ROOT / ".env"
CONFIG_FILE = ROOT / "config.toml"

if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
else:
    load_dotenv(override=True)


@dataclass(frozen=True)
class WhisperCfg:
    model: str = "openai/whisper-small.en"
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.2
    prompt: str = ""
    # Maximum times a single word may repeat consecutively before collapsing.
    max_single_word_repeats: int = 4
    # When a segment is comprised of a single repeated word over the limit, drop it entirely instead of emitting.
    drop_repeated_only_segments: bool = True


@dataclass(frozen=True)
class VoiceCfg:
    silence_threshold_seconds: float = 1.25
    vad_threshold: float = 0.75
    max_speech_buf_seconds: int = 0


# Wrapup configuration removed: wrapup is now handled entirely in the Node.js component.


@dataclass(frozen=True)
class RefinerCfg:
    model: Optional[str] = None
    context_log_lines: int = 10
    temperature: float = 0.5
    timeout: float = 5.0
    prompt: Optional[str] = None


@dataclass(frozen=True)
class NetCfg:
    ai_service_url: str = "ws://localhost:8771"
    chunk_ms: int = 1000
    host: str = "0.0.0.0"
    port: int = 8771
    path: str = "/"


@dataclass(frozen=True)
class DiscordCfg:
    allowed_commanders: tuple[int, ...] = ()


@dataclass(frozen=True)
class AppConfig:
    discord: DiscordCfg
    net: NetCfg
    whisper: WhisperCfg
    voice: VoiceCfg
    refiner: RefinerCfg
    username_map: Dict[str, str]
    phrase_map: Dict[str, str]
    openai_api_key: Optional[str]
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
        with CONFIG_FILE.open("rb") as f:
            return tomli.load(f)
    return {}


@lru_cache
def load_app_config(
    host: Optional[str] = None,
    port: Optional[int] = None,
    ai_service_url: Optional[str] = None,
    path: Optional[str] = None,
) -> AppConfig:
    raw = _raw_toml()
    discord = raw.get("discord", {})
    net = raw.get("net", {})
    whisper = raw.get("whisper", {})
    voice = raw.get("voice", {})
    # wrapup section ignored in Python; handled in Node
    refiner = raw.get("refiner", {})
    username_map = dict(raw.get("username_map", {}) or {})
    phrase_map = dict(raw.get("phrase_map", {}) or {})

    # Only secrets (e.g. OPENAI_API_KEY) come from environment; all functional knobs come from config.toml.
    whisper_cfg = WhisperCfg(
        model=whisper.get("model", WhisperCfg.model),
        logprob_threshold=float(whisper.get("logprob_threshold", WhisperCfg.logprob_threshold)),
        no_speech_threshold=float(whisper.get("no_speech_threshold", WhisperCfg.no_speech_threshold)),
        prompt=whisper.get("prompt", WhisperCfg.prompt),
        max_single_word_repeats=int(whisper.get("max_single_word_repeats", WhisperCfg.max_single_word_repeats)),
        drop_repeated_only_segments=bool(
            whisper.get(
                "drop_repeated_only_segments",
                WhisperCfg.drop_repeated_only_segments,
            )
        ),
    )
    voice_cfg = VoiceCfg(
        silence_threshold_seconds=float(voice.get("silence_threshold_seconds", VoiceCfg.silence_threshold_seconds)),
        vad_threshold=float(voice.get("vad_threshold", VoiceCfg.vad_threshold)),
        max_speech_buf_seconds=int(voice.get("max_speech_buf_seconds", VoiceCfg.max_speech_buf_seconds)),
    )
    # no wrapup config in Python
    refiner_cfg = RefinerCfg(
        model=refiner.get("model"),
        context_log_lines=int(refiner.get("context_log_lines", RefinerCfg.context_log_lines)),
        temperature=float(refiner.get("temperature", RefinerCfg.temperature)),
        timeout=float(refiner.get("timeout", RefinerCfg.timeout)),
        prompt=refiner.get("prompt"),
    )
    # Resolve networking config from config.toml with optional CLI overrides
    net_cfg = _resolve_net_cfg(net, host=host, port=port, ai_service_url=ai_service_url, path=path)
    discord_cfg = DiscordCfg(
        allowed_commanders=_coerce_int_list(discord.get("allowed_commanders")),
    )
    return AppConfig(
        discord=discord_cfg,
        net=net_cfg,
        whisper=whisper_cfg,
        voice=voice_cfg,
        refiner=refiner_cfg,
        username_map=username_map,
        phrase_map=phrase_map,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        device=os.getenv("DEVICE", "auto"),  # auto = choose best available (cuda/mps/rocm/cpu)
    )


def _ensure_leading_slash(path: str) -> str:
    if not path:
        return "/"
    return path if path.startswith("/") else "/" + path


def _parse_ws_url(url: str) -> Tuple[str, int, str]:
    """Parse a websocket URL or host:port string and return (host, port, path).

    Accepts values like 'ws://host:port', 'wss://host:port', or 'host:port'.
    Defaults to host='0.0.0.0', port=8771, path='/' when missing.
    """
    if not url:
        return ("0.0.0.0", 8771, "/")
    if "://" not in url:
        url = "ws://" + url
    parsed = urlparse(url)
    host = parsed.hostname or "0.0.0.0"
    port = parsed.port or 8771
    path = _ensure_leading_slash(parsed.path or "/")
    return (host, port, path)


def _resolve_net_cfg(
    raw_net: dict,
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    ai_service_url: Optional[str] = None,
    path: Optional[str] = None,
) -> NetCfg:
    # Base values from config.toml
    ai_url_cfg = raw_net.get("ai_service_url", NetCfg.ai_service_url)
    # Precedence for URL: CLI override > config
    ai_service_url_eff = ai_service_url or ai_url_cfg
    url_host, url_port, url_path = _parse_ws_url(ai_service_url_eff)
    # Path precedence: CLI path > config 'path' > URL path
    cfg_path = raw_net.get("path")
    if path is not None:
        effective_path = _ensure_leading_slash(str(path))
    elif cfg_path is not None:
        effective_path = _ensure_leading_slash(str(cfg_path))
    else:
        effective_path = url_path
    # Host/port precedence: CLI host/port > parsed URL host/port
    effective_host = host if host is not None else url_host
    effective_port = int(port) if port is not None else int(url_port)

    return NetCfg(
        ai_service_url=ai_service_url_eff,
        chunk_ms=int(raw_net.get("chunk_ms", NetCfg.chunk_ms)),
        host=effective_host,
        port=effective_port,
        path=effective_path,
    )
