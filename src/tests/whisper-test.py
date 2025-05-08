import argparse
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple, Optional, Any
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
    AutomaticSpeechRecognitionPipeline,
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

class WhisperModelPatched(WhisperForConditionalGeneration):
    """A subclass of WhisperForConditionalGeneration to override the forward method to fix
        a bug that occurs when using `no_speech_threshold`."""
    def forward(self, *args, **kwargs):
        kwargs.pop('input_ids', None)
        return super().forward(*args, **kwargs)

class WhisperLogprobPipeline(AutomaticSpeechRecognitionPipeline):
    """A subclass of the Whisper pipeline to also return logprobs, if present."""

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
        # for o in model_outputs:
        #     if isinstance(o, dict):
        #         print("dict", o.keys())
        #         tokens = o["tokens"]
        #         if isinstance(tokens, torch.Tensor):
        #             print(tokens, tokens.shape, tokens.dtype)
        #         else:
        #             print(tokens.sequences, tokens.scores, print(o["stride"]))
        #     elif isinstance(o, torch.Tensor):
        #         print(o.shape, o.dtype)
        #     else:
        #         print("ModelOutput", dir(o))
        # outputs = [{
        #     "tokens": model_outputs[0]["tokens"].sequences,
        #     "stride": model_outputs[0]["stride"],
        # }]
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
        first_token_idxs = [len(o["tokens"].sequences[0]) - len(o["tokens"].scores) for o in model_outputs]
        token_ids_list = [o["tokens"].sequences[0][f:] for o, f in zip(model_outputs, first_token_idxs)]
        logprobs_list = [
            [
                torch.log_softmax(s[0], dim=-1)[tok_id] for tok_id, s in zip(tok_ids, scores)
            ] for tok_ids, scores in zip(token_ids_list, (o["tokens"].scores for o in model_outputs))
        ]
        print([
            [(self.tokenizer.decode(tok_id), logprob.tolist()) for tok_id, logprob in zip(tok_ids, logprobs)] for tok_ids, logprobs in zip(token_ids_list, logprobs_list)
        ])

        # logprobs = []
        # for o in model_outputs:
            
        # print([(len(o["tokens"].sequences[0]), len(o["tokens"].scores)) for o in model_outputs])
        # logprobs = [
        #     [
        #         torch.log_softmax(s, dim=-1)[o["tokens"].sequences[0][i]] if 
        #         for i, s in enumerate(o["tokens"].scores)
        #     ] for o in model_outputs
        # ]
        # print(logprobs, result)

        # # Skip if no scores (e.g., logprobs not requested)
        # scores = model_outputs.get("scores", None)
        # sequences = model_outputs.get("sequences", None)
        # if scores is None or sequences is None:
        #     return result

        # logprobs = [torch.log_softmax(score, dim=-1) for score in scores]

        # # Estimate prompt length
        # prompt_len = self.model.config.forced_decoder_ids and len(self.model.config.forced_decoder_ids) or 0
        # token_ids = sequences[0][prompt_len:]

        # token_logprobs = [lp[token_id].item() for lp, token_id in zip(logprobs, token_ids)]
        # tokens = [self.tokenizer.decode([tid]).strip() for tid in token_ids]

        # # Add logprobs to output
        # result["tokens"] = tokens
        # result["logprobs"] = token_logprobs

        return result
    
def transcribe_2(audio_fp16: np.ndarray) -> None:
    model = WhisperModelPatched.from_pretrained(
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

    system_prompt_ids = processor.get_prompt_ids("saison, beer.", return_tensors="pt")
    if not MODEL_NAME.endswith(".en"):
        lang_task_prompt_ids = torch.tensor(
            [id for _, id in processor.get_decoder_prompt_ids(language="en", task="transcribe")],
            dtype=torch.long,
        )
    else:
        lang_task_prompt_ids = torch.tensor([], dtype=torch.long)
    prompt_ids = torch.cat([lang_task_prompt_ids, system_prompt_ids], dim=-1).to("cuda:0")
    print(processor.decode(prompt_ids.tolist()), len(prompt_ids))

    pipe = WhisperLogprobPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device="cuda:0",
    )

    kwargs = {
        "temperature": (0., 0.1, 0.25, 0.5),
        "forced_decoder_ids": None,
        "logprob_threshold": -1.25,
        "compression_ratio_threshold": 2.4,
        "no_speech_threshold": 0.33,
        "condition_on_prev_tokens": True,
        "prompt_ids": prompt_ids,
        "prompt_condition_type": "first-segment",
        # "return_segments": True,
    }
    
    result = pipe(
        inputs=audio_fp16,
        generate_kwargs=kwargs,
        chunk_length_s=30,
        batch_size=1,
        return_timestamps=False,
    )
    print(result)

if __name__ == "__main__":
    main()
