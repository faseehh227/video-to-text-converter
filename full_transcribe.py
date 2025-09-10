import ffmpeg
import whisper
import os

# Step 1: Extract audio from video
def extract_audio(video_path, audio_path):
    print("🔄 Extracting audio...")
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .overwrite_output()
        .run()
    )
    print(f"✅ Audio saved to {audio_path}")

# Step 2: Transcribe audio with optional language (auto-detect fallback)
def transcribe_audio(audio_path, language=None):
    model = whisper.load_model("small")

    if not language:
        print("🌐 Auto-detecting language...")
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        language = max(probs, key=probs.get)
        print(f"✅ Detected language: {language}")
        detected = language
    else:
        print(f"🌍 Using selected language: {language}")
        detected = None

    result = model.transcribe(audio_path, language=language)
    transcript_text = " ".join([seg["text"] for seg in result["segments"]])

    return transcript_text, detected

# Run both steps manually (if testing from CLI)
if __name__ == "__main__":
    video_file = "video.mp4"
    audio_file = "audio.wav"

    if not os.path.exists(video_file):
        print("❌ video.mp4 not found.")
    else:
        extract_audio(video_file, audio_file)
        transcript, lang = transcribe_audio(audio_file)
        print("📝 Final Transcript:\n")
        print(transcript)
        if lang:
            print(f"\n🌍 Auto-detected language: {lang}")
