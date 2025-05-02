import argparse
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple, Optional
import torch
from transformers import (
    Pipeline as WhisperPipeline,
    pipeline,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperForCausalLM,
    WhisperTokenizerFast,
    WhisperFeatureExtractor,
)
from src.debug_utils import debug_play_audio

TARGET_SR: int = 16000
MODEL_NAME: str = "openai/whisper-large-v3-turbo"

def wav_to_fp16(file_path: str, target_sr: int) -> np.ndarray:
    """
    Reads a WAV file, resamples to 16kHz, and converts to float16 NumPy array.

    Args:
        file_path (str): Path to the input WAV file.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The processed audio data as a float16 NumPy array.
    """
    # Read the audio file
    audio, sr = sf.read(file_path, dtype='float32') # Read as float32 first for resampling
    print(f"Original sample rate: {sr} Hz, shape: {audio.shape}, dtype: {audio.dtype}")

    # Ensure mono audio if needed, or handle stereo appropriately
    if audio.ndim > 1:
        print(f"Audio has {audio.shape[1]} channels, converting to mono by taking the mean.")
        audio = np.mean(audio, axis=1)

    # Resample if necessary
    if sr != target_sr:
        print(f"Resampling from {sr} Hz to {target_sr} Hz...")
        # librosa.resample expects float32 or float64, and returns the same type
        resampled_audio: np.ndarray = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        audio = resampled_audio
    else:
        print("Audio is already at the target sample rate.")

    # Convert to float16
    return audio.astype(np.float16)

def main() -> None:
    parser = argparse.ArgumentParser(description="Do stuff.")
    parser.add_argument("wav_file", type=str, help="Path to the input WAV file.")
    args = parser.parse_args()

    audio_fp16 = wav_to_fp16(args.wav_file, TARGET_SR)
    transcribe_2(audio_fp16)

class WhisperOverride(WhisperForConditionalGeneration):
    def forward(self, *args, **kwargs):
        kwargs.pop('input_ids', None)
        return super().forward(*args, **kwargs)
    
def transcribe_2(audio_fp16: np.ndarray) -> None:
    model = WhisperOverride.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        tokenizer=WhisperTokenizerFast.from_pretrained(MODEL_NAME),
        feature_extractor=WhisperFeatureExtractor.from_pretrained(MODEL_NAME, return_attention_mask=True),
    )
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device="cuda:0",
    )

    prompt_ids = processor.get_prompt_ids("saison, beer.", return_tensors="pt").to("cuda:0")

    kwargs = {
        "temperature": (0., 0.1, 0.2),
        "forced_decoder_ids": None,
        "logprob_threshold": -1.5,
        "compression_ratio_threshold": 2.4,
        "no_speech_threshold": 0.33,
        "language": "english",
        "task": "transcribe",
        "condition_on_prev_tokens": True,
        "prompt_ids": prompt_ids,
        "prompt_condition_type": "first-segment",
    }

    
    result = pipe(
        inputs=audio_fp16,
        generate_kwargs=kwargs,
        chunk_length_s=10,
        batch_size=1,
        return_timestamps=False,
    )
    print(result)

if __name__ == "__main__":
    main()
