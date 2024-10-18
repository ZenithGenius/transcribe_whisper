import whisper
from pyannote.audio import Pipeline
import torch

# pip install whisper pyannote.audio

# Ensure CUDA is available for faster processing, if possible
device = "cuda" if torch.cuda.is_available() else "cpu"

def transcribe_audio(audio_path):
    model = whisper.load_model("base", device=device)
    result = model.transcribe(audio_path)
    return result['text']

def speaker_diarization(audio_path):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    diarization_result = pipeline({'uri': 'example', 'audio': audio_path})
    return diarization_result

def insert_speaker_changes(transcription, diarization_result):
    speaker_changes = []
    for segment, _, speaker in diarization_result.itertracks(yield_label=True):
        speaker_changes.append((segment.middle, speaker))
    speaker_changes.sort(key=lambda x: x[0])

    # Simple method to estimate insertion points based on time
    words_per_minute = 150  # Average speaking speed
    words = transcription.split()
    total_duration = diarization_result.get_timeline().duration()
    words_per_second = words_per_minute / 60
    total_words = len(words)
    seconds_per_word = total_duration / total_words
    
    # Insert markers
    for change_time, speaker in speaker_changes:
        word_position = int(change_time / seconds_per_word)
        if word_position < len(words):
            words.insert(word_position, f"\n[Speaker Change: Speaker {speaker}]\n")

    return ' '.join(words)

def main(audio_path):
    print("Transcribing audio...")
    transcription = transcribe_audio(audio_path)
    
    print("Performing speaker diarization...")
    diarization_result = speaker_diarization(audio_path)
    
    print("Merging transcription with speaker changes...")
    final_transcription = insert_speaker_changes(transcription, diarization_result)
    
    print("\nFinal Transcription with Speaker Changes:\n")
    print(final_transcription)

if __name__ == "__main__":
    audio_path = input("Enter the path to your audio file : ")
    main(audio_path)
