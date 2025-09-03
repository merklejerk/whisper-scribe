import asyncio
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

from src.transcriber import AsyncWhisperTranscriber, TranscribeJob


class StubPipeline:
    def __init__(self, result: Any):
        self._result = result

    # New transcriber calls pipeline(audio, return_timestamps=True, generate_kwargs=..., ...)
    def __call__(self, audio_np, return_timestamps: bool = True, generate_kwargs: Dict[str, Any] | None = None, **kwargs):  # type: ignore[override]
        return self._result


class StubWhisperProcessor:
    @classmethod
    def from_pretrained(cls, model_name: str):  # type: ignore[override]
        return cls()

    def get_prompt_ids(self, text: str, return_tensors: str = "pt"):
        return [1, 2, 3]


def write_silent_wav(path: Path, sr: int = 16000, seconds: float = 0.25):
    n_samples = int(sr * seconds)
    data = (np.zeros(n_samples, dtype=np.int16)).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(data)


@pytest.mark.asyncio
async def test_integration_pass_pcm_from_wav(tmp_path, monkeypatch):
    import src.transcriber as tr

    # Create a temp 16kHz mono PCM16 WAV file
    wav_path = Path(tmp_path) / "sample_16k.wav"
    write_silent_wav(wav_path, sr=16000, seconds=0.1)

    # Read raw PCM16 frames from the wav file
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        pcm_bytes = wf.readframes(wf.getnframes())

    # Stub external deps to avoid model load
    monkeypatch.setattr(tr, "resolve_device", lambda device: SimpleNamespace(pipeline_index="cpu"))
    monkeypatch.setattr(tr, "WhisperProcessor", StubWhisperProcessor)
    monkeypatch.setattr(tr, "pipeline", lambda task, model, device: StubPipeline({"text": "integration ok"}))

    emitted: List[Any] = []

    async def emit_cb(seg):
        emitted.append(seg)

    cfg = AsyncWhisperTranscriber.WhisperRuntimeCfg(
        device="auto",
        model="openai/whisper-tiny",
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    t = AsyncWhisperTranscriber(cfg, emit_cb=emit_cb)
    await t.start()
    try:
        await t.submit(TranscribeJob(id="iu1", pcm16=pcm_bytes))
        await t.queue.join()
        assert emitted and emitted[0].text == "integration ok"
    finally:
        await t.stop()
