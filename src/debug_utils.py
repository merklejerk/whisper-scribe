import os
import numpy as np
from scipy.signal import resample_poly

DEBUG = os.getenv('DEBUG', '0') == '1'
if os.getenv('DEBUG'):
    import soundfile as sf
    import sounddevice as sd

def debug_play_audio(audio_np: np.ndarray, input_sr: int):
    """Plays audio using sounddevice if DEBUG env var is set.
    Allows specifying the output device via DEBUG_AUDIO_DEVICE env var (using device index).
    """
    if DEBUG:
        # print("Available output devices:")
        # print(sd.query_devices())

        # Determine the target device index
        target_device_index_str = os.getenv('DEBUG_AUDIO_DEVICE')
        target_device_index = None
        if target_device_index_str:
            try:
                target_device_index = int(target_device_index_str)
                print(f"Debug: DEBUG_AUDIO_DEVICE specified: {target_device_index}")
                # Basic validation: Check if index exists (sounddevice might raise error later if invalid kind)
                sd.query_devices(target_device_index)
            except ValueError:
                print(f"Debug: Invalid DEBUG_AUDIO_DEVICE value '{target_device_index_str}'. Must be an integer. Falling back to default.")
                target_device_index = None
            except sd.PortAudioError:
                    print(f"Debug: Invalid device index {target_device_index} specified in DEBUG_AUDIO_DEVICE. Falling back to default.")
                    target_device_index = None
            except Exception as e: # Catch potential errors from query_devices
                print(f"Debug: Error querying specified device index {target_device_index}: {e}. Falling back to default.")
                target_device_index = None

        if target_device_index is None:
            default_output_index = sd.default.device[1]
            if default_output_index == -1:
                raise ValueError("Debug: No default output device found by sounddevice!")
            target_device_index = default_output_index

        # Query the selected device
        output_device_info = sd.query_devices(target_device_index, kind='output')
        output_sr = int(output_device_info['default_samplerate'])
        print(f"Debug: Using device: {output_device_info['name']} (Index: {target_device_index})")
        print(f"Debug: Playing audio... Input SR: {input_sr}, Output SR: {output_sr}")

        # Resample if necessary
        if input_sr != output_sr:
            common_multiple = np.lcm(input_sr, output_sr)
            up = common_multiple // input_sr
            down = common_multiple // output_sr
            audio_resampled = resample_poly(audio_np, up, down)
        else:
            audio_resampled = audio_np # No resampling needed

        # Check audio data stats
        if len(audio_resampled) == 0:
            print("Debug: Audio data is empty after resampling (or initially).")
            return

        # Play on the selected device
        sd.play(audio_resampled, samplerate=output_sr, device=target_device_index)
        sd.wait()

def save_norm_audio(
    audio: np.ndarray,
    file_path: str,
    sample_rate: int = 16000,
) -> None:
    """Normalizes and saves audio as a mono WAV file (float32, -1.0 to 1.0) at the given sample rate."""
    # Ensure audio is float32
    audio = audio.astype(np.float32)
    # Normalize to -1.0 to 1.0 if not already
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    # Save as mono WAV
    sf.write(file_path, audio, sample_rate, subtype='PCM_16')
