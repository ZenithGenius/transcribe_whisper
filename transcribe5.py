#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Attempt to import necessary modules and handle the case where whisper is not installed
try:
    import whisper
    import warnings
    import argparse
    import logging
    # Suppress warning messages from libraries
    warnings.filterwarnings("ignore")
except ModuleNotFoundError:
    print("Error: 'whisper' module not found. It seems the Whisper package is not installed.")
    user_response = input("Would you like to receive guidance on setting up a Python virtual environment and installing Whisper? (y/n): ").strip().lower()
    if user_response == 'y':
        print("\nGuide to setting up a Python virtual environment and installing Whisper:")
        print("1. Ensure Python 3 is installed and accessible from your terminal.")
        print("2. Create a new virtual environment:\n   python3 -m venv whisper_env")
        print("3. Activate the virtual environment:\n   - On Windows: whisper_env\\Scripts\\activate\n   - On Unix or MacOS: source whisper_env/bin/activate")
        print("4. Install Whisper using pip:\n   pip install whisper")
        install_now = input("Would you like to install Whisper now? (y/n): ").strip().lower()
        if install_now == 'y':
            os.system("python3 -m venv whisper_env && source whisper_env/bin/activate && pip install whisper")
            print("Whisper has been installed. Please activate the virtual environment with 'source whisper_env/bin/activate' before running this script.")
            sys.exit()
        else:
            print("Installation aborted. Please follow the manual instructions to install Whisper.")
            sys.exit()
    else:
        print("Please install Whisper before running this script.")
        sys.exit()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transcribe_audio_to_text(input_path: str, output_folder: str, model_size: str = "medium", language: str = "fr"):
    """
    Transcribes audio file(s) to text using OpenAI's Whisper model and saves the transcriptions to a specified output folder.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(model_size)

    def process_file(audio_file_path):
        try:
            output_file_name = f"{Path(audio_file_path).stem}.txt"
            output_file_path = Path(output_folder) / output_file_name
            result = model.transcribe(audio_file_path, language=language)
            transcription = result["text"]

            with open(output_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(transcription)

            logging.info(f"Transcription saved to {output_file_path}")
        except Exception as e:
            logging.error(f"Failed to process {audio_file_path}: {e}")

    if Path(input_path).is_file():
        process_file(input_path)
    elif Path(input_path).is_dir():
        audio_files = [os.path.join(root, file) for root, _, files in os.walk(input_path) for file in files if file.endswith(('.mp3', '.aac', '.wav', '.m4a', '.flac'))]
        for audio_file_path in audio_files:
            process_file(audio_file_path)
    else:
        logging.error(f"The specified path '{input_path}' is neither a file nor a directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribes audio files to text using OpenAI's Whisper model.")
    parser.add_argument("input_path", help="Path to the audio file or directory to be transcribed.")
    parser.add_argument("output_folder", help="Path to the output folder where transcriptions will be saved.")
    parser.add_argument("--model_size", default="medium", help="Size of the Whisper model to use. Defaults to 'medium'.")
    parser.add_argument("--language", default="fr", help="Language of the audio file(s). Defaults to 'fr' for French.")

    args = parser.parse_args()

    transcribe_audio_to_text(args.input_path, args.output_folder, args.model_size, args.language)
