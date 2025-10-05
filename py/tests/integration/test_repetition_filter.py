import asyncio
from types import SimpleNamespace

import pytest

from src.transcriber import AsyncWhisperTranscriber, TranscribeJob


class StubPipeline:
    def __init__(self, text: str):
        self._text = text

    def __call__(self, audio_np, return_timestamps=True, generate_kwargs=None, **kw):  # type: ignore[override]
        return {"text": self._text}


class StubWhisperProcessor:
    @classmethod
    def from_pretrained(cls, model_name: str):  # type: ignore[override]
        return cls()

    def get_prompt_ids(self, text: str, return_tensors: str = "pt"):
        return [1, 2, 3]


@pytest.mark.asyncio
async def test_repetition_is_collapsed(monkeypatch):
    import src.transcriber as tr

    # Provide single word repeated 20 times
    repeated = "you " * 20
    monkeypatch.setattr(tr, "resolve_device", lambda device: SimpleNamespace(to_arg=lambda: "cpu"))
    monkeypatch.setattr(tr, "WhisperProcessor", StubWhisperProcessor)
    monkeypatch.setattr(tr, "pipeline", lambda task, model, device: StubPipeline(repeated.strip()))

    emitted = {}

    async def emit_cb(seg):
        emitted["text"] = seg.text

    cfg = AsyncWhisperTranscriber.WhisperRuntimeCfg(
        device="cpu",
        model="openai/whisper-tiny",
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        max_single_word_repeats=3,
        drop_repeated_only_segments=False,
    )
    t = AsyncWhisperTranscriber(cfg, emit_cb=emit_cb)
    await t.start()
    try:
        # Provide 100ms of silence bytes (placeholder) -> size doesn't matter for stub
        pcm = b"\x00" * 3200
        await t.submit(TranscribeJob(id="r1", pcm16=pcm))
        await asyncio.wait_for(t.queue.join(), timeout=3)
    finally:
        await t.stop()

    assert "text" in emitted
    # Expect only 3 tokens remain
    assert emitted["text"].split() == ["you", "you", "you"]


@pytest.mark.asyncio
async def test_repetition_segment_dropped(monkeypatch):
    import src.transcriber as tr

    repeated = "echo " * 15
    monkeypatch.setattr(tr, "resolve_device", lambda device: SimpleNamespace(to_arg=lambda: "cpu"))
    monkeypatch.setattr(tr, "WhisperProcessor", StubWhisperProcessor)
    monkeypatch.setattr(tr, "pipeline", lambda task, model, device: StubPipeline(repeated.strip()))

    emitted = {}

    async def emit_cb(seg):
        emitted["text"] = seg.text

    cfg = AsyncWhisperTranscriber.WhisperRuntimeCfg(
        device="cpu",
        model="openai/whisper-tiny",
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        max_single_word_repeats=4,
        drop_repeated_only_segments=True,
    )
    t = AsyncWhisperTranscriber(cfg, emit_cb=emit_cb)
    await t.start()
    try:
        pcm = b"\x00" * 1600
        await t.submit(TranscribeJob(id="r2", pcm16=pcm))
        await asyncio.wait_for(t.queue.join(), timeout=3)
    finally:
        await t.stop()

    # Expect segment suppressed
    assert "text" not in emitted