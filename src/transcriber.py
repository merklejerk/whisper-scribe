import asyncio
import queue
import threading
import numpy as np
from typing import Optional, Dict, Any, Tuple, TypeAlias, Generic, TypeVar
from transformers import Pipeline as WhisperPipeline
from transformers import pipeline
from .debug_utils import debug_play_audio
from .config import WHISPER_LOGPROB_THRESHOLD

# Module-level audio constants
TARGET_SR = 16000  # expected incoming sample rate
SILENCE_TOP_DB = 60  # threshold for trimming silence

MetadataT = TypeVar('MetadataT')

# Define types used in the class, now using MetadataT
AudioPipelineOutput = Dict[str, Any]
# Job now takes MetadataT and audio bytes
AudioProcessingJob: TypeAlias = Tuple[MetadataT, bytes]
# Result now returns MetadataT and transcription string
TranscriptionResult: TypeAlias = Tuple[MetadataT, str]


class Transcriber(Generic[MetadataT]):
    """Handles audio transcription sequentially, passing opaque metadata through."""

    _pipeline: Optional[WhisperPipeline]
    _output_queue: asyncio.Queue[TranscriptionResult[MetadataT]]
    _input_queue: queue.Queue[Optional[AudioProcessingJob[MetadataT]]]
    _thread: Optional[threading.Thread]
    _stop_event: threading.Event
    _model_name: str

    # Output queue type hint updated
    def __init__(self, output_queue: asyncio.Queue[TranscriptionResult[MetadataT]], model_name: str, device: str = "cpu"):
        self._pipeline = None
        self._output_queue = output_queue
        self._model_name = model_name
        self._input_queue = queue.Queue()
        self._thread = None
        self._stop_event = threading.Event()
        self._device = device

    def _initialize_pipeline(self):
        """Initializes the Whisper pipeline using the configured model name."""
        if self._pipeline is not None:
            return

        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self._model_name,
            device=self._device,
            chunk_length_s=30,
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
                # Handle exceptions in processing
                print(f"Transcriber thread: Exception occurred while processing item: {e}")
                raise
            finally:
                # Ensure task_done is called even if processing fails
                input_queue.task_done()

    def _process_item(
        self,
        item: AudioProcessingJob[MetadataT],
    ):
        metadata, audio_data = item
        if not audio_data:
            return

        # audio_data is 16kHz mono PCM int16 bytes
        audio_np: np.ndarray = np.frombuffer(audio_data, dtype=np.int16)
        # normalize to float32 in [-1,1]
        audio_np: np.ndarray = audio_np.astype(np.float32) / 32768.0

        debug_play_audio(audio_np, TARGET_SR)

        kwargs = {
            "temperature": (0., 0.1, 0.2),
            "forced_decoder_ids": None,
            "logprob_threshold": WHISPER_LOGPROB_THRESHOLD,
            "no_speech_threshold": 0.6,
            "compression_ratio_threshold": 2.4,
            # "max_new_tokens": 448,
            "num_beams": 1,
            "condition_on_prev_tokens": False,
            "return_timestamps": True,
        }
        if not self._model_name.endswith(".en"):
            kwargs["language"] = "english"
            kwargs["task"] = "transcribe"

        print(kwargs)
        result: AudioPipelineOutput = self._pipeline(
            inputs={ "raw": audio_np, "sampling_rate": TARGET_SR },
            generate_kwargs=kwargs,
        )
        transcription: str = result["text"].strip()

        if transcription:
            try:
                self._output_queue.put_nowait((metadata, transcription))
            except asyncio.QueueFull:
                print(f"Transcriber thread: WARNING - Output queue is full. Discarding transcription.")
