#!/usr/bin/env python3

import whisper

def transcribe_audio_to_text(audio_file_path: str, output_file_path: str, model_size: str = "medium", language: str = "fr"):
    """
    Transcribes an audio file to text using OpenAI's Whisper model and saves the output to a text file.

    Parameters:
    - audio_file_path: Path to the audio file to be transcribed.
    - output_file_path: Path where the transcription text file will be saved.
    - model_size: Size of the Whisper model to use. This script defaults to "medium".
    - language: Language of the audio file. Default is "fr" for French.
    """
    # Load the specified Whisper model
    model = whisper.load_model(model_size)

    # Transcribe the audio, specifying the language for better accuracy
    result = model.transcribe(audio_file_path, language=language)

    # Extract the transcription text
    transcription = result["text"]

    # Save the transcription to a file
    with open(output_file_path, "w", encoding="utf-8") as text_file:
        text_file.write(transcription)

    print(f"Transcription saved to {output_file_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: ./transcribe.py <audio_file_path> <output_file_path>")
    else:
        audio_file_path = sys.argv[1]
        output_file_path = sys.argv[2]
        transcribe_audio_to_text(audio_file_path, output_file_path)
