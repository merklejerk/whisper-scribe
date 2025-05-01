import os
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

DEBUG = os.getenv('DEBUG', '0') == '1'
if os.getenv('DEBUG'):
    import sounddevice as sd

def debug_play_audio(audio_np: np.ndarray, input_sr: int):
    """Plays audio using sounddevice if DEBUG env var is set.
    Allows specifying the output device via DEBUG_AUDIO_DEVICE env var (using device index).
    """
    if os.getenv('DEBUG'):
        try:
            print("\n--- Debug Audio Playback Start ---")
            print("Available output devices:")
            print(sd.query_devices())

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
                    print("Debug: No default output device found by sounddevice!")
                    print("--- Debug Audio Playback End ---\n")
                    return
                target_device_index = default_output_index
                print(f"Debug: Using default output device index: {target_device_index}")

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
                print(f"Debug: Resampled audio from {len(audio_np)} to {len(audio_resampled)} samples (up={up}, down={down}).")
            else:
                audio_resampled = audio_np # No resampling needed

            # Check audio data stats
            if len(audio_resampled) > 0:
                print(f"Debug: Audio data stats before playback - Min: {np.min(audio_resampled):.4f}, Max: {np.max(audio_resampled):.4f}, Mean: {np.mean(audio_resampled):.4f}, Length: {len(audio_resampled)}")
            else:
                print("Debug: Audio data is empty after resampling (or initially).")
                print("--- Debug Audio Playback End ---\n")
                return

            # Play on the selected device
            sd.play(audio_resampled, samplerate=output_sr, device=target_device_index)
            sd.wait() # Wait for playback to finish
            print("Debug: Playback finished.")
        except sd.PortAudioError as pae:
             print(f"Debug: PortAudioError during playback setup or execution: {pae}")
             print(f"Debug: Check if device index {target_device_index} is a valid OUTPUT device.")
        except Exception as e:
            print(f"Debug: Error playing audio: {e}")
        finally:
            print("--- Debug Audio Playback End ---\n")
