import numpy as np
from scipy.signal import resample_poly, butter, lfilter

def has_sound(
    waveform: np.ndarray,
    *,
    threshold=1e-4,
    sample_rate=16000,
    window_ms=50,
    stride_ms=10,
    total_sound_ms=250,
) -> bool:
    """ Checks if the waveform contains sound above a certain threshold. """
    window_size = int(sample_rate * window_ms / 1000) 
    stride = int(sample_rate * stride_ms / 1000)

    active_samples = 0

    for i in range(0, len(waveform) - window_size + 1, stride):
        window = waveform[i:i + window_size]
        energy = np.sqrt(np.mean(window**2))
        if energy > threshold:
            active_samples += window_size

    return active_samples >= int(total_sound_ms / 1000 * sample_rate)

def pcm16_to_norm_waveform(audio: bytes, source_sr: int, target_sr: int) -> np.ndarray:
    """Converts PCM int16 audio bytes to a normalized waveform."""
    samples = np.frombuffer(audio, dtype=np.int16).reshape(-1, 2)
    mono = np.mean(samples.astype(np.float32), axis=1)
    if source_sr != target_sr:
        resampled = resample_poly(mono, target_sr, source_sr)
    else:
        resampled = mono
    return np.clip(resampled, -32768, 32767).astype(np.float32) / 32768.0

def bandpass_filter(
    signal: np.ndarray,
    sr: int,
    lowcut: float = 300.0,
    highcut: float = 3400.0,
    order: int = 6
) -> np.ndarray:
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, signal)

def pre_emphasis(signal: np.ndarray, coeff: float = 0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def rms_normalize(signal: np.ndarray, target_db: float = -20.0):
    rms = np.sqrt(np.mean(signal**2))
    if rms == 0:
        return signal  # silent
    target_rms = 10 ** (target_db / 20.0)
    return signal * (target_rms / rms)