from __future__ import annotations
import asyncio
import queue
import threading
import numpy as np
import librosa  # for silence trimming
# Updated imports for Generics
from typing import Optional, Dict, Any, Tuple, TypeAlias, Generic, TypeVar
from transformers import Pipeline as WhisperPipeline
from transformers import pipeline

# Module-level audio constants
TARGET_SR = 16000  # expected incoming sample rate
SILENCE_TOP_DB = 60  # threshold for trimming silence

# TODO: Not used yet due to transformers + whisper bug.
WHISPER_NO_SPEECH_THRESHOLD = 0.6
WHISPER_LOG_PROB_THRESHOLD = -1.5
WHISPER_BEAM_SIZE = 3
WHISPER_BEST_OF = 3
WHISPER_TEMPERATURES = (0.0, 0.2, 0.4)

# Define a TypeVar for the generic metadata
MetadataT = TypeVar('MetadataT')

# Define types used in the class, now using MetadataT
AudioPipelineOutput = Dict[str, Any]
# Job now takes MetadataT and audio bytes
AudioProcessingJob: TypeAlias = Tuple[MetadataT, bytes]
# Result now returns MetadataT and transcription string
TranscriptionResult: TypeAlias = Tuple[MetadataT, str]

# Make the class Generic over MetadataT
class AudioProcessor(Generic[MetadataT]):
    """Handles audio transcription sequentially, passing opaque metadata through."""
    # Type hints using MetadataT
    _pipeline: Optional[WhisperPipeline]
    _output_queue: asyncio.Queue[TranscriptionResult[MetadataT]] # Use specific generic type
    _input_queue: queue.Queue[Optional[AudioProcessingJob[MetadataT]]] # Use specific generic type
    _thread: Optional[threading.Thread]
    _stop_event: threading.Event
    _pipeline_lock: threading.Lock
    _model_name: str

    # Output queue type hint updated
    def __init__(self, output_queue: asyncio.Queue[TranscriptionResult[MetadataT]], model_name: str):
        self._pipeline = None
        self._output_queue = output_queue
        self._model_name = model_name
        self._input_queue = queue.Queue()
        self._thread = None
        self._stop_event = threading.Event()
        self._pipeline_lock = threading.Lock()

    def _initialize_pipeline(self) -> bool:
        """Initializes the Whisper pipeline using the configured model name."""
        with self._pipeline_lock:
            if self._pipeline is not None:
                return True

            print(f"AudioProcessor: Lazily initializing Whisper pipeline ({self._model_name})...")
            try:
                self._pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self._model_name,
                    device="cpu",
                    chunk_length_s=30,
                )
                print(f"AudioProcessor: Whisper pipeline ({self._model_name}) initialized successfully.")
                return True
            except Exception as e:
                import traceback
                print(f"AudioProcessor: FATAL ERROR initializing Whisper pipeline ({self._model_name}): {e} {traceback.format_exc()}")
                self._pipeline = None
                return False

    def start(self):
        """Starts the background processing thread."""
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            # Pass the input queue to the thread target
            self._thread = threading.Thread(target=self._run, args=(self._input_queue,), daemon=True, name="AudioProcessorThread")
            self._thread.start()
            print("AudioProcessor thread started.")
        else:
            print("AudioProcessor thread already running.")

    def stop(self):
        """Signals the background thread to stop and waits for it."""
        if self._thread and self._thread.is_alive():
            print("Stopping AudioProcessor thread...")
            self._stop_event.set()
            # Put sentinel value into the input queue
            self._input_queue.put(None)
            self._thread.join() # Wait for thread to finish
            print("AudioProcessor thread stopped.")
        self._thread = None

    # Accept metadata of type MetadataT
    def submit_job(self, metadata: MetadataT, audio_data: bytes):
        """Submits audio data with associated metadata for processing."""
        if not self._stop_event.is_set():
            # Log received metadata generically if needed, or remove logging details
            print(f"AudioProcessor: Submitting job with metadata")
            self._input_queue.put((metadata, audio_data))
        else:
            print(f"AudioProcessor: Cannot submit job, processor is stopping.")

    # Input queue type hint updated
    def _run(self, input_queue: queue.Queue[Optional[AudioProcessingJob[MetadataT]]]):
        """The main loop for the background processing thread."""
        print("AudioProcessor thread entering run loop.")

        # Initialize the pipeline.
        if not self._initialize_pipeline():
            print("AudioProcessor: Pipeline initialization failed. Cannot process audio. Skipping job.")

        while not self._stop_event.is_set():
            try:
                item = input_queue.get(block=True)

                if item is None or self._stop_event.is_set():
                    print("AudioProcessor thread received stop signal or sentinel.")
                    if item is not None:
                         input_queue.task_done()
                    break

                # Unpack metadata (type MetadataT) and audio_data
                metadata, audio_data = item
                # Log generically or remove specific details
                print(f"AudioProcessor thread: Processing job with metadata {metadata}...")

                if not audio_data:
                    print(f"AudioProcessor thread: Skipping empty audio data.")
                    input_queue.task_done()
                    continue

                # --- Perform processing (synchronously in this thread) ---
                try:
                    # audio_data is 16kHz mono PCM int16 bytes
                    audio_np: np.ndarray = np.frombuffer(audio_data, dtype=np.int16)
                    # normalize to float32 in [-1,1]
                    audio_np: np.ndarray = audio_np.astype(np.float32) / 32768.0
                    # Trim leading/trailing silence
                    audio_np, _ = librosa.effects.trim(
                        audio_np,
                        top_db=SILENCE_TOP_DB
                    )

                    # Prepare input
                    pipeline_input = {
                        "raw": audio_np,
                        "sampling_rate": TARGET_SR,
                    }

                    if self._pipeline is None:
                         print(f"AudioProcessor thread: ERROR - Pipeline is None despite initialization check. Skipping.")
                         input_queue.task_done()
                         continue

                    # Transcribe (BLOCKING call within this thread)
                    print(f"AudioProcessor thread: Starting Whisper pipeline...")
                    result: AudioPipelineOutput = self._pipeline(
                        pipeline_input,
                        generate_kwargs={
                            # "language": "english",
                            # "task": "transcribe",
                            "return_timestamps": True,
                            "temperature": 0.01,
                            # TODO: Adding these parameters causes whisper to break atm.
                            # Maybe when https://github.com/huggingface/transformers/pull/36809 is merged, we can use them.
                            # "no_speech_threshold": WHISPER_NO_SPEECH_THRESHOLD,
                            # "log_prob_threshold": WHISPER_LOG_PROB_THRESHOLD,
                            # "beam_size": WHISPER_BEAM_SIZE,
                            # "best_of": WHISPER_BEST_OF,
                            # "temperature": WHISPER_TEMPERATURES,
                        })
                    print(f"AudioProcessor thread: Whisper pipeline finished.")
                    transcription: str = result["text"].strip()

                    if transcription:
                        # --- Put result onto the output queue --- #
                        try:
                            # Pass the original metadata (type MetadataT) along with the transcription
                            self._output_queue.put_nowait((metadata, transcription))
                            # Log generically
                            print(f"AudioProcessor thread: Put transcription with metadata onto output queue.")
                        except asyncio.QueueFull:
                             # Log generically
                             print(f"AudioProcessor thread: WARNING - Output queue is full. Discarding transcription.")
                        # --- End putting result --- #
                    else:
                         # Log generically
                        print(f"AudioProcessor thread: Whisper pipeline produced no transcription.")

                except Exception as e:
                    import traceback
                    # Log generically
                    print(f"AudioProcessor thread: Error processing audio: {e} {traceback.format_exc()}")
                finally:
                    input_queue.task_done() # Signal task completion for this item

            except queue.Empty:
                # This shouldn't happen with block=True unless timeout is used
                continue
            except Exception as e:
                # Catch broader errors in the loop itself
                import traceback
                print(f"AudioProcessor thread: Unexpected error in run loop: {e} {traceback.format_exc()}")
                # Ensure task_done is called if an error occurs after getting an item but before finally
                if 'item' in locals() and item is not None:
                    try:
                        input_queue.task_done()
                    except ValueError: # Already marked done
                        pass

        print("AudioProcessor thread exiting run loop.")
