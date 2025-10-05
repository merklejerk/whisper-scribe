from __future__ import annotations

import asyncio
import typing
import numpy as np
from dataclasses import dataclass
from transformers import pipeline, WhisperProcessor

from .devices import resolve_device
from .debug import make_debug_logger

from . import config


@dataclass
class TranscribeJob:
    id: str
    pcm16: bytes  # 16kHz mono little-endian
    # Profiles can only override the prompt; keep this strongly typed
    prompt: typing.Optional[str] = None


TARGET_SR = 16000  # expected incoming sample rate


@typing.runtime_checkable
class TranscriberLike(typing.Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def submit(self, job: "TranscribeJob") -> None: ...


# New lightweight result returned from the transcriber; not tied to Pydantic models
@dataclass
class TranscriptionResult:
    id: str
    text: str
    # capture_ts and duration omitted intentionally — reconstruction happens upstream if needed


class AsyncWhisperTranscriber:
    @dataclass(frozen=True)
    class WhisperRuntimeCfg:
        device: str
        model: str
        logprob_threshold: float
        no_speech_threshold: float
        # repetition filtering
        max_single_word_repeats: int = 4
        drop_repeated_only_segments: bool = True

    def __init__(
        self,
        cfg: "AsyncWhisperTranscriber.WhisperRuntimeCfg",
        emit_cb: typing.Optional[typing.Callable[[TranscriptionResult], typing.Awaitable[None]]] = None,
    ):
        self.emit_cb = emit_cb
        self.queue: asyncio.Queue[TranscribeJob] = asyncio.Queue(maxsize=64)
        self._task: typing.Optional[asyncio.Task] = None
        self._pipe = None
        self._processor = None
        self._prompt_cache: dict[str, object] = {}
        self._cfg = cfg
        self._dbg = make_debug_logger("transcriber")
        self._on_fatal: typing.Optional[typing.Callable[[BaseException], None]] = None

    def set_emit_callback(self, cb: typing.Callable[[TranscriptionResult], typing.Awaitable[None]]):
        self.emit_cb = cb

    def set_on_fatal(self, cb: typing.Callable[[BaseException], None]):
        """Register a callback invoked when the transcriber encounters a fatal error.

        The callback runs in the event loop thread before the exception is re-raised
        to fail the internal task. Use this to trigger shutdown at the application level.
        """
        self._on_fatal = cb

    async def start(self):
        if self._task:
            return
        # Lazy load model in executor to avoid blocking loop
        loop = asyncio.get_running_loop()
        resolved = resolve_device(self._cfg.device)
        self._dbg(f"loading model {self._cfg.model} on device {resolved.to_arg()}")

        def load_pipe():
            proc = WhisperProcessor.from_pretrained(self._cfg.model)
            self._processor = proc
            return pipeline(
                task="automatic-speech-recognition",
                model=self._cfg.model,
                device=resolved.to_arg(),
            )

        self._pipe = await loop.run_in_executor(None, load_pipe)
        # Do not seed any prompt at startup; prompts come via per-job overrides

        self._task = asyncio.create_task(self._run(), name="whisper-transcriber")

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                # Normal shutdown path
                pass
            self._task = None

    async def submit(self, job: TranscribeJob):
        # Drop incoming job when queue is full, with a debug message
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull:
            self._dbg(f"dropping transcribe job id={job.id} - queue full ({self.queue.qsize()}/{self.queue.maxsize})")

    async def _run(self):
        while True:
            job = await self.queue.get()
            try:
                # Convert bytes to float32 numpy in -1..1 range
                if not job.pcm16:
                    raise ValueError("TranscribeJob.pcm16 is empty")
                audio_np = np.frombuffer(job.pcm16, dtype="<i2").astype("float32") / 32768.0
                if audio_np.size == 0:
                    raise ValueError("Decoded audio array is empty")
                # Build generation kwargs from provided cfg (thresholds are static),
                # and apply per-job prompt override when provided.
                whisper_cfg = self._cfg
                generate_kwargs = {
                    "temperature": (0.0, 0.25, 0.5, 0.75),
                    "logprob_threshold": float(whisper_cfg.logprob_threshold),
                    "no_speech_threshold": float(whisper_cfg.no_speech_threshold),
                    "condition_on_prev_tokens": True,
                    "compression_ratio_threshold": 1.35,
                    "forced_decoder_ids": None,
                }
                # Attach prompt ids based on per-job override only; cache per text
                use_prompt = job.prompt
                if use_prompt and self._processor is not None:
                    ids = self._prompt_cache.get(use_prompt)
                    if ids is None:
                        # Compute and memoize prompt ids on current device
                        resolved = resolve_device(self._cfg.device)
                        ids = self._processor.get_prompt_ids(use_prompt, return_tensors="pt").to(resolved.to_arg())
                        self._prompt_cache[use_prompt] = ids
                    generate_kwargs["prompt_ids"] = ids
                    generate_kwargs["prompt_condition_type"] = "first-segment"
                # Force english if model isn't .en variant like legacy
                model_name_lower = self._cfg.model.lower()
                if not model_name_lower.endswith(".en"):
                    generate_kwargs["language"] = "english"
                    generate_kwargs["task"] = "transcribe"

                # Run pipeline (blocking) in executor
                loop = asyncio.get_running_loop()

                def infer():
                    if self._pipe is None:
                        raise RuntimeError("Transcriber pipeline is not initialized")
                    # TODO: Use model.generate() instead of pipeline for more control and to handle deprecated pipe args.
                    return self._pipe(
                        audio_np,
                        return_timestamps=True,
                        generate_kwargs=generate_kwargs,
                    )

                result = await loop.run_in_executor(None, infer)
                if not result:
                    raise RuntimeError("ASR pipeline returned no result")
                # HF pipeline may return dict or list with segments; normalize
                if isinstance(result, list):
                    # take concatenated text fields
                    texts = [seg.get("text", "") for seg in result if isinstance(seg, dict)]
                    text = " ".join(t.strip() for t in texts).strip()
                elif isinstance(result, dict):
                    text = result.get("text", "").strip()
                else:
                    text = ""
                if text:
                    # Basic repetition suppression: collapse runs beyond threshold and optionally drop
                    text = self._suppress_repetition(
                        text,
                        max_repeats=self._cfg.max_single_word_repeats,
                        drop_only=self._cfg.drop_repeated_only_segments,
                    )
                if text:
                    seg = TranscriptionResult(
                        id=job.id,
                        text=text,
                    )
                    if self.emit_cb:
                        await self.emit_cb(seg)
            except Exception as e:
                self._dbg(f"transcribe job id={getattr(job, 'id', '?')} failed: {e}")
                # Notify application-level handler before propagating
                cb = self._on_fatal
                if cb is not None:
                    try:
                        cb(e)
                    except Exception:
                        pass
                raise
            finally:
                self.queue.task_done()

    # --- internal helpers -------------------------------------------------
    @staticmethod
    def _suppress_repetition(text: str, max_repeats: int, drop_only: bool) -> str:
        """Collapse excessive single-token repetitions.

        If the entire segment is one word repeated over the allowed limit and
        drop_only is True, return an empty string to suppress emission.
        """
        if max_repeats <= 0:
            return text
        parts = text.split()
        if not parts:
            return text
        # Detect if entire segment is same token
        unique_tokens = set(parts)
        if len(unique_tokens) == 1 and len(parts) > max_repeats:
            return "" if drop_only else unique_tokens.pop()
        out: list[str] = []
        last = None
        run = 0
        for w in parts:
            if w == last:
                run += 1
            else:
                last = w
                run = 1
            if run <= max_repeats:
                out.append(w)
        return " ".join(out)
