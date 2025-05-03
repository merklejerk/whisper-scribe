import numpy as np
from numpy import ndarray
import argparse
import soundfile as sf
import librosa
import sounddevice as sd
from src.waveform_utils import bandpass_filter, resample, rms_normalize, pre_emphasis

def wav_to_fp32(file_path: str, target_sr: int) -> np.ndarray:
    """
    Reads a WAV file, resamples to 16kHz, and converts to a normalized float32 NumPy array.

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

    # Convert to float32
    return audio.astype(np.float32)

TARGET_SR: int = 16000

def main() -> None:
    parser = argparse.ArgumentParser(description="Do stuff.")
    parser.add_argument("wav_file", type=str, help="Path to the input WAV file.")
    args = parser.parse_args()

    audio = wav_to_fp32(args.wav_file, TARGET_SR)
    audio = pre_emphasis(audio, coeff=0.8)
    audio = bandpass_filter(audio, TARGET_SR, order=3, lowcut=250, highcut=3300)
    audio = rms_normalize(audio)
    audio = audio.clip(-1.0, 1.0)
    sd.play(audio, samplerate=TARGET_SR)
    sd.wait()

if __name__ == "__main__":
    main()
