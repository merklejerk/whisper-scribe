import asyncio
import queue
import threading
import torch
import re
import numpy as np
from typing import Optional, Dict, Any, Tuple, TypeAlias, Generic, TypeVar
from dataclasses import dataclass
import os
from transformers import (
    Pipeline as WhisperPipeline,
    pipeline,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
    WhisperFeatureExtractor,
    AutomaticSpeechRecognitionPipeline,
)
from .debug_utils import save_norm_audio
from .config import WHISPER_LOGPROB_THRESHOLD, WHISPER_NO_SPEECH_THRESHOLD, WHISPER_PROMPT, PHRASE_MAP

# Module-level audio constants
TARGET_SR = 16000  # expected incoming sample rate

MetadataT = TypeVar('MetadataT')

# Define types used in the class, now using MetadataT
AudioPipelineOutput = Dict[str, Any]
# Job now takes MetadataT and audio bytes
AudioProcessingJob: TypeAlias = Tuple[MetadataT, np.ndarray]

@dataclass
class TranscriptionResult(Generic[MetadataT]):
    """Represents the result of a transcription job."""
    metadata: MetadataT
    transcription: str
    mean_logprob: float
    std_logprob: float
    n_tokens: int

class Transcriber(Generic[MetadataT]):
    """Handles audio transcription sequentially, passing opaque metadata through."""

    _pipe: Optional[WhisperPipeline]
    _output_queue: asyncio.Queue[TranscriptionResult[MetadataT]]
    _input_queue: queue.Queue[Optional[AudioProcessingJob[MetadataT]]]
    _thread: Optional[threading.Thread]
    _stop_event: threading.Event
    _model_name: str
    _device: str
    _torch_dtype: torch.dtype
    _prompt_ids: Optional[torch.Tensor]

    # Output queue type hint updated
    def __init__(self, output_queue: asyncio.Queue[TranscriptionResult[MetadataT]], model_name: str, device: str = "cpu"):
        self._pipe = None
        self._output_queue = output_queue
        self._model_name = model_name
        self._input_queue = queue.Queue()
        self._thread = None
        self._stop_event = threading.Event()
        self._device = device
        self._torch_dtype = torch.float16 if self._device != "cpu" and torch.cuda.is_available() else torch.float32
        self._prompt_ids = None
        self._processor = None

    def _initialize_pipeline(self):
        """Initializes the Whisper pipeline using the configured model name."""
        if self._pipe is not None:
            return
        
        model = WhisperModelPatched.from_pretrained(
            self._model_name,
            torch_dtype=self._torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self._processor = WhisperProcessor.from_pretrained(
            self._model_name,
            tokenizer=WhisperTokenizerFast.from_pretrained(self._model_name),
            feature_extractor=WhisperFeatureExtractor.from_pretrained(self._model_name, return_attention_mask=True),
        )

        if WHISPER_PROMPT:
            system_prompt_ids = self._processor.get_prompt_ids(WHISPER_PROMPT, return_tensors="pt")
        else:
            system_prompt_ids = torch.tensor([], dtype=torch.long)
        self._prompt_ids = system_prompt_ids.to(self._device)

        self._pipe = WhisperLogprobPipeline(
            model=model,
            tokenizer=self._processor.tokenizer,
            feature_extractor=self._processor.feature_extractor,
            torch_dtype=self._torch_dtype,
            device=self._device,
        )
        print(f"Transcriber: Whisper pipeline ({self._model_name}) initialized successfully.")

    def is_started(self) -> bool:
        """Checks if the background processing thread is running."""
        return self._thread is not None
    
    def start(self):
        """Starts the background processing thread."""
        self._initialize_pipeline()
        if self._thread is None:
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(self._input_queue,),
                daemon=True,
                name="AudioProcessorThread",
            )
            self._thread.start()
            print("Transcriber thread started.")

    def stop(self):
        """Signals the background thread to stop and waits for it."""
        if self._thread:
            self._stop_event.set()
            # Put sentinel value into the input queue
            self._input_queue.put(None)
            self._thread.join() # Wait for thread to finish
            print("Transcriber thread stopped.")
        self._thread = None

    def submit_job(self, metadata: MetadataT, audio_data: bytes):
        """Submits audio data with associated metadata for processing."""
        if not self._stop_event.is_set():
            # Log received metadata generically if needed, or remove logging details
            self._input_queue.put((metadata, audio_data))
        else:
            print(f"Transcriber: Cannot submit job, processor is stopping.")

    def _run(self, input_queue: queue.Queue[Optional[AudioProcessingJob[MetadataT]]]):
        """The main loop for the background processing thread."""
        while not self._stop_event.is_set():
            item = input_queue.get(block=True)

            try:
                # Stop on sentinel value or if stop event is set.
                if item is None or self._stop_event.is_set():
                    break
                self._process_item(item)
            except Exception as e:
                print(f"Transcriber thread: Exception occurred while processing item: {e}")
                raise
            finally:
                # Ensure task_done is called even if processing fails
                input_queue.task_done()

    def _process_item(
        self,
        item: AudioProcessingJob[MetadataT],
    ):
        metadata, audio_np = item
        if len(audio_np) == 0:
            return

        # Audio should be mono, normalized, 16kHz
        if os.environ.get("SAVE_AUDIO", "0") == "1":
            save_norm_audio(audio_np, "debug.wav", TARGET_SR)

        # Convert to the correct dtype for Whisper
        np_type = np.float32 if self._torch_dtype == torch.float32 else np.float16
        audio_np: np.ndarray = audio_np.astype(np_type)

        kwargs = {
            "temperature": (0., 0.1, 0.25, 0.5),
            "forced_decoder_ids": None,
            "logprob_threshold": WHISPER_LOGPROB_THRESHOLD,
            "compression_ratio_threshold": 1.35,
            "no_speech_threshold": WHISPER_NO_SPEECH_THRESHOLD,
            "condition_on_prev_tokens": True,
        }
        if self._prompt_ids is not None:
            kwargs["prompt_ids"] = self._prompt_ids
            kwargs["prompt_condition_type"] = "first-segment"
        if not self._model_name.endswith(".en"):
            kwargs["language"] = "english"
            kwargs["task"] = "transcribe"

        result: AudioPipelineOutput = self._pipe(
            inputs=audio_np,
            generate_kwargs=kwargs,
            chunk_length_s=30,
            batch_size=1,
            return_timestamps=False,
        )
        transcription: str = _remap_phrases(result["text"].strip(), PHRASE_MAP)

        if transcription:
            try:
                self._output_queue.put_nowait(TranscriptionResult(
                    metadata=metadata,
                    transcription=transcription,
                    mean_logprob=result["mean_logprob"],
                    std_logprob=result["std_logprob"],
                    n_tokens=result["n_tokens"],
                ))
            except asyncio.QueueFull:
                print(f"Transcriber thread: WARNING - Output queue is full. Discarding transcription.")

class WhisperModelPatched(WhisperForConditionalGeneration):
    """A subclass of WhisperForConditionalGeneration to override the forward method to fix
        a bug that occurs when using `no_speech_threshold`."""
    def forward(self, *args, **kwargs):
        kwargs.pop('input_ids', None)
        return super().forward(*args, **kwargs)

class WhisperLogprobPipeline(AutomaticSpeechRecognitionPipeline):
    """A subclass of the Whisper pipeline to also return logprob statistics."""

    def _forward(self, model_inputs, return_timestamps=False, **generate_kwargs):
        return super()._forward(
            model_inputs,
            return_timestamps,
            **{
                **generate_kwargs,
                "output_scores": True,
                "return_dict_in_generate": True,
            },
        )

    def postprocess(
        self,
        model_outputs: list[dict[str,Any]],
        return_timestamps: bool = False,
        return_language: bool = False,
    ) -> dict[str, Any]:
        result = super().postprocess(
            [
                {
                    "tokens": o["tokens"].sequences,
                    "stride": o["stride"],
                } for o in model_outputs
            ],
            return_timestamps=return_timestamps,
            return_language=return_language,
        )
        # Skip past the injected prompt tokens.
        first_token_idxs = [len(o["tokens"].sequences[0]) - len(o["tokens"].scores) for o in model_outputs]
        # Get the token IDs after the prompt.
        token_ids_list = [o["tokens"].sequences[0][f:] for o, f in zip(model_outputs, first_token_idxs)]
        # Get all the logprobs for all the token IDs in every chunk.
        # Due to the way audio is chunked and overlapped (stride), this is easier to just evaluate
        # in aggregate instead of by individual token.
        all_logprobs = np.array([
            torch.log_softmax(s[0], dim=-1)[tok_id].tolist() 
             for tok_ids, scores in zip(token_ids_list, (o["tokens"].scores for o in model_outputs))
             for tok_id, s in zip(tok_ids, scores)
        ], dtype=np.float32)
        return {
            **result,
            "n_tokens": all_logprobs.shape[0],
            "mean_logprob": all_logprobs.mean().item(),
            "std_logprob": all_logprobs.std().item(),
        }

def _remap_phrases(text: str, phrase_map: dict) -> str:
    # For each phrase, do a case-insensitive replacement
    for k, v in phrase_map.items():
        # Use word boundaries if you want to only match whole words, otherwise just re.sub
        pattern = re.compile(r'(?i)'+re.escape(k))
        text = pattern.sub(lambda m: v[0].upper() + v[1:] if m.group(0)[0].isupper() else v, text)
    return text