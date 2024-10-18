#!/usr/bin/env python3

import os
import whisper
from pathlib import Path

def transcribe_audio_to_text(input_path: str, model_size: str = "medium", language: str = "fr"):
    """
    Transcribes audio file(s) to text using OpenAI's Whisper model. If input_path is a directory,
    all audio files in the directory will be transcribed.

    Parameters:
    - input_path: Path to the audio file or directory containing audio files to be transcribed.
    - model_size: Size of the Whisper model to use. Defaults to "medium".
    - language: Language of the audio file(s). Defaults to "fr" for French.
    """
    # Load the specified Whisper model
    model = whisper.load_model(model_size)

    # Define a function to process individual files
    def process_file(audio_file_path):
        # Determine the output file path
        output_file_path = f"{audio_file_path}.txt"
        # Transcribe the audio, specifying the language for better accuracy
        result = model.transcribe(audio_file_path, language=language)
        # Extract the transcription text
        transcription = result["text"]
        # Save the transcription to a file
        with open(output_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(transcription)
        print(f"Transcription saved to {output_file_path}")

    # Check if input_path is a file or directory
    if Path(input_path).is_file():
        process_file(input_path)
    elif Path(input_path).is_dir():
        # Iterate over all files in the directory
        for root, dirs, files in os.walk(input_path):
            for file in files:
                # Update this if condition to match your audio files extensions
                if file.endswith(('.mp3','.aac', '.wav', '.m4a', '.flac')):
                    audio_file_path = os.path.join(root, file)
                    process_file(audio_file_path)
    else:
        print(f"The specified path '{input_path}' is neither a file nor a directory.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: script.py <audio_file_path_or_directory>")
    else:
        input_path = sys.argv[1]
        transcribe_audio_to_text(input_path)
