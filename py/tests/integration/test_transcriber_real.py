import asyncio
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly

from src.transcriber import AsyncWhisperTranscriber, TranscribeJob, TARGET_SR


def _read_audio_to_pcm16_bytes(path: Path) -> bytes:
    """Read any common audio file and return 16kHz mono PCM16 bytes."""
    ext = path.suffix.lower()
    if ext == ".wav":
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            data = wf.readframes(wf.getnframes())
        if sw == 2 and ch == 1 and sr == TARGET_SR:
            return data  # already PCM16 mono 16k
        # fall through to generic path if not already in desired format

    # Generic path: decode with soundfile, resample, downmix, convert
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # downmix to mono
    mono = audio.mean(axis=1)
    if sr != TARGET_SR:
        # Use polyphase resampling for quality; gcd approximation: up = TARGET_SR, down = sr
        # To keep it simple: resample_poly handles arbitrary rates
        # Note: resample_poly expects integers; use ratio approximation

        g = gcd(int(TARGET_SR), int(sr)) if sr.is_integer() else 1
        up = int(TARGET_SR // g)
        down = int(sr // g) if sr.is_integer() else int(sr)
        mono = resample_poly(mono, up, down).astype(np.float32)
    # clip to [-1,1] and convert to int16
    mono = np.clip(mono, -1.0, 1.0)
    pcm16 = (mono * 32767.0).astype(np.int16)
    return pcm16.tobytes()


@pytest.mark.real
@pytest.mark.asyncio
async def test_real_transcribe_emits_text(monkeypatch):
    # Always use the committed fixture: tests/fixtures/stt.wav
    path = Path(__file__).resolve().parents[1] / "fixtures" / "stt.wav"
    assert path.exists(), f"Audio file not found: {path}"

    pcm_bytes = _read_audio_to_pcm16_bytes(path)
    assert len(pcm_bytes) > 0

    model = "openai/whisper-large-v3-turbo"

    got = {}

    async def emit_cb(seg):
        got["seg"] = seg

    cfg = AsyncWhisperTranscriber.WhisperRuntimeCfg(
        device="cuda",
        model=model,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        prompt="this is a clip from a youtube video.",
    )
    t = AsyncWhisperTranscriber(cfg, emit_cb=emit_cb)
    await t.start()
    try:
        await t.submit(TranscribeJob(id="real", pcm16=pcm_bytes))
        # Wait until processed (with timeout safety)
        try:
            await asyncio.wait_for(t.queue.join(), timeout=60)
        except asyncio.TimeoutError:
            pytest.fail("Transcription timed out")

        assert "seg" in got, "No transcription emitted"
        text = got["seg"].text.strip()
        print(text)
        # Expect some alphabetic characters in result
        assert any(c.isalpha() for c in text), f"Unexpected transcript: {text!r}"
    finally:
        await t.stop()
