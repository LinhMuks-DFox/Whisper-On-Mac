#!/usr/bin/env python3

import argparse
from ConvertModel import get_converted_model  # Your model conversion logic
import whisper
import os

def transcribe_audio(audio_path, model_name="base", device="mps"):
    """
    Main function for audio transcription.
    :param audio_path: Path to the audio file.
    :param model_name: Name of the Whisper model.
    :param device: Device to run the model ("mps", "cpu", etc.).
    :return: Transcription text.
    """
    # Check if the audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"The audio file {audio_path} does not exist.")

    # Load and convert the model
    print(f"Loading model {model_name} to {device}...")
    model = get_converted_model(model_name)
    model.to(device)

    # Load and transcribe audio
    print(f"Transcribing audio {audio_path}...")
    audio = whisper.load_audio(audio_path)
    result = model.transcribe(audio)
    return result["text"]

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Transcribe audio using the Whisper model.")
    parser.add_argument("-ap", type=str, help="Path to the audio file.")
    parser.add_argument("--model", type=str, default="base", help="Name of the Whisper model (default: base).")
    parser.add_argument("--device", type=str, default="mps", help="Device to run the model (default: mps).")
    parser.add_argument("-f", "--format", type=str, choices=["txt", "none"], default="none",
                        help="Output format for the transcription. Choose 'txt' to save the result to a .txt file.")
    parser.add_argument("--output", type=str, default="transcription.txt",
                        help="Path to save the transcription result (default: transcription.txt).")

    # Parse arguments
    args = parser.parse_args()

    try:
        # Call the transcription function
        transcription = transcribe_audio(args.ap, args.model, args.device)
        print("\nTranscription result:")
        print(transcription)

        # Save the result if the format is 'txt'
        if args.format == "txt":
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(transcription)
            print(f"\nTranscription result saved to {args.output}")
    except Exception as e:
        print(f"An error occurred: {e}")