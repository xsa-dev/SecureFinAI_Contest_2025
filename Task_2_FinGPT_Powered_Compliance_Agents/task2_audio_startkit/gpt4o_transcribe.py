import os
import sys
sys.path.append('.')
import logging
import fire
import numpy as np
import base64
from dotenv import load_dotenv

import librosa
import soundfile as sf
import torch
from tqdm import tqdm
from openai import OpenAI

from prompt import asr_instructions
from utils import setup_logger, load_config, prepare_model_input
from jiwer import wer, process_words
from text_normalizer.preprocess_text import preprocess_text_asr

# Load environment variables from .env file
load_dotenv()

# Device (not strictly needed here, but keep for parity)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_SR = 16000


def audio_to_wav_bytes(
    audio_array: np.ndarray,
    sampling_rate: int,
    target_sr: int = TARGET_SR
) -> bytes:
    """
    Clean + convert arbitrary audio to mono, 16kHz, 16-bit PCM WAV bytes.
    Ensures: finite samples, mono mixdown, resampled to 16k, PCM_16 WAV.
    """
    import io

    # Ensure numpy array
    x = np.asarray(audio_array)
    if x.ndim == 0 or x.size == 0:
        raise ValueError("Audio array is empty")

    # Remove NaN/Inf
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert to float32 in [-1, 1]
    if np.issubdtype(x.dtype, np.integer):
        maxv = float(np.iinfo(x.dtype).max)
        if maxv <= 0:
            raise ValueError(f"Invalid integer dtype range for audio: {x.dtype}")
        x = x.astype(np.float32) / maxv
    else:
        x = x.astype(np.float32)

    # Downmix to mono if multi-channel
    if x.ndim > 1:
        # (T, C) or (C, T) -> heuristically average over channel axis
        if x.shape[-1] <= 8 and x.shape[0] > x.shape[-1]:
            # likely (T, C)
            x = np.mean(x, axis=-1)
        else:
            # likely (C, T)
            x = np.mean(x, axis=0)

    # Resample to target_sr if needed
    sr = int(sampling_rate)
    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Clip to [-1, 1]
    x = np.clip(x, -1.0, 1.0)

    # Write PCM_16 WAV to bytes
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def wav_bytes_to_base64(wav_bytes: bytes) -> str:
    return base64.b64encode(wav_bytes).decode("utf-8")


def transcribe_with_gpt4o(client: OpenAI, audio_b64: str, model: str = "gpt-4o-audio-preview") -> str:
    """
    Transcribe audio using GPT-4o audio via Chat Completions API.
    Expects base64-encoded PCM16 WAV data at 16 kHz.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please transcribe this audio to text. Only provide the transcription without any additional commentary."
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": "wav"
                            }
                        }
                    ]
                }
            ],
        )

        # Most SDKs return a string in message.content for chat.completions
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            logging.error("No transcription in response")
            return ""

    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        if hasattr(e, 'response'):
            logging.error(f"Response details: {e.response}")
        return ""


def main(config_path: str = None):
    # Load config
    configs = load_config(config_path)
    dataset_name = configs['dataset']
    model_name = "gpt-4o-audio-preview"
    task = configs['task']

    # Prepare log
    log_file_path = configs["log_path"]
    os.makedirs(log_file_path, exist_ok=True)
    log_file_full = os.path.join(log_file_path, f"{dataset_name}_GPT4o_{task}.log")
    logger = setup_logger(log_file_full)
    logger.info("= = " * 20)
    logger.info("Dataset: {}".format(dataset_name))
    logger.info("Model: gpt-4o-audio-preview")
    logger.info("= = " * 20)

    # API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        return

    # Client
    client = OpenAI(api_key=openai_api_key)

    # Prepare data
    input_data = prepare_model_input(dataset_name, asr_instructions)
    input_data = input_data[:100]
    logger.info(f"Number of samples to process in {dataset_name}: {len(input_data)}")

    predictions = []
    references = []

    for inputs in tqdm(input_data, leave=False):
        sample_id = inputs["sample_id"]
        reference = inputs["answer"]
        audio_array = inputs["audio"]["array"]
        sampling_rate = inputs["audio"]["sampling_rate"]

        # Validate audio data
        if audio_array is None or len(audio_array) == 0:
            logger.warning(f"sample_id={sample_id}, skipping empty audio")
            predictions.append("")
            references.append(preprocess_text_asr(reference))
            continue

        audio_duration = len(audio_array) / max(1, sampling_rate)
        logger.info(
            f"Processing sample_id={sample_id}, duration={audio_duration:.2f}s, "
            f"sample_rate={sampling_rate}Hz, dtype={np.asarray(audio_array).dtype}, "
            f"shape={np.asarray(audio_array).shape}, finite={np.isfinite(np.asarray(audio_array)).all()}"
        )

        try:
            # Convert to clean PCM_16, mono, 16 kHz WAV bytes
            wav_bytes = audio_to_wav_bytes(audio_array, sampling_rate)
            audio_b64 = wav_bytes_to_base64(wav_bytes)

            # Transcribe
            output = transcribe_with_gpt4o(client, audio_b64, model=model_name)
        except Exception as e:
            logger.error(f"Failed to process sample_id={sample_id}: {e}")
            output = ""

        if output:
            output = output.strip()
            model_prediction = preprocess_text_asr(output)
            answer = preprocess_text_asr(reference)
            predictions.append(model_prediction)
            references.append(answer)
            logger.info(
                f"sample_id={sample_id}, final_transcription='{output}', "
                f"reference='{reference}', Processed_output='{model_prediction}', "
                f"Processed_reference='{answer}'"
            )
        else:
            logger.warning(f"sample_id={sample_id}, transcription failed")
            predictions.append("")
            references.append(preprocess_text_asr(reference))

    # Calculate WER
    sample_wer = []
    incorrect = 0
    total = 0
    sample_idx = 0

    for prediction, reference in zip(predictions, references):
        if prediction:  # Only calculate if we got a transcription
            measures = process_words(reference, prediction)
            incorrect += measures.substitutions + measures.deletions + measures.insertions
            total += measures.substitutions + measures.deletions + measures.hits

            wer_score = wer(reference, prediction)
            logging.info(f"Number of samples {sample_idx}: WER is {wer_score}")
        else:
            wer_score = 1.0  # Max WER for failed transcriptions
            logging.info(f"Number of samples {sample_idx}: WER is {wer_score} (transcription failed)")

        sample_wer.append({
            "reference": reference,
            "prediction": prediction,
            "wer": wer_score,
        })
        sample_idx += 1

    total_wer = incorrect / total if total > 0 else 1.0
    print(f"Final WER: {total_wer}")
    logging.info(f"Final WER: {total_wer}")


if __name__ == '__main__':
    fire.Fire(main)
