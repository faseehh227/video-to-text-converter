from flask import Flask, render_template, request, jsonify
import os
import ffmpeg
import whisper
from transformers import pipeline, MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

LANGUAGE_NAMES = {
    "en": ("English", "🇺🇸"),
    "es": ("Spanish", "🇪🇸"),
    "fr": ("French", "🇫🇷"),
    "de": ("German", "🇩🇪"),
    "hi": ("Hindi", "🇮🇳"),
    "ur": ("Urdu", "🇵🇰"),
    "pa": ("Punjabi", "🇮🇳🇵🇰"),
    "zh": ("Chinese", "🇨🇳"),
    "ja": ("Japanese", "🇯🇵"),
    "ar": ("Arabic", "🇸🇦"),
    "it": ("Italian", "🇮🇹"),
    "ru": ("Russian", "🇷🇺"),
}

# Load core models once
whisper_model = whisper.load_model("small")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# ❌ FIXED: Paraphraser with slow tokenizer to avoid SentencePiece error
paraphrase_model_name = "Vamsi/T5_Paraphrase_Paws"

paraphrase_tokenizer = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(paraphrase_model_name)
paraphraser = pipeline("text2text-generation", model=paraphrase_model, tokenizer=paraphrase_tokenizer)

title_generator = pipeline("text2text-generation", model="google/flan-t5-large")

def get_translation_pipeline(target_lang):
    model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name, use_safetensors=True)
    return pipeline("translation", model=model, tokenizer=tokenizer)

def translate_text(text, target_lang_code):
    translator = get_translation_pipeline(target_lang_code)
    chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
    translated = [translator(chunk)[0]['translation_text'] for chunk in chunks]
    return " ".join(translated)

def extract_audio(video_path, audio_path):
    ffmpeg.input(video_path).output(
        audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000'
    ).overwrite_output().run()

def transcribe_audio(audio_path, language=None):
    if not language:
        print("🌐 Auto-detecting language...")
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
        _, probs = whisper_model.detect_language(mel)
        language = max(probs, key=probs.get)
        print(f"✅ Detected language: {language}")
    else:
        print(f"🌍 Using selected language: {language}")

    result = whisper_model.transcribe(audio_path, language=language)
    transcript_text = " ".join([seg["text"] for seg in result["segments"]])
    return transcript_text, language

def summarize_text(text):
    chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
    summaries = []
    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=80,
            min_length=20,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )[0]['summary_text']
        summaries.append(summary.strip())
    return " ".join(summaries)

def paraphrase_text(text):
    prompt = f"paraphrase: {text} </s>"
    outputs = paraphraser(prompt, max_new_tokens=256, num_return_sequences=1, do_sample=True)
    return outputs[0]['generated_text'].strip()

def generate_title(text):
    prompt = f"Generate a short, relevant and creative title for this content:\n\n{text.strip()}"
    try:
        response = title_generator(prompt, max_new_tokens=20, num_return_sequences=1)
        return response[0]['generated_text'].strip()
    except Exception as e:
        print("⚠️ Title generation failed:", str(e))
        return ""

@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    summary = ""
    key_points = []
    detected_language_name = ""
    audio_url = ""
    title = ""

    if request.method == "POST":
        video = request.files.get("video")
        language = request.form.get("language") or None

        if video:
            video_path = os.path.join(UPLOAD_FOLDER, video.filename)
            audio_filename = os.path.splitext(video.filename)[0] + ".wav"
            audio_path = os.path.join(STATIC_FOLDER, audio_filename)

            video.save(video_path)
            extract_audio(video_path, audio_path)
            transcript, detected_code = transcribe_audio(audio_path, language)

            if detected_code:
                name_flag = LANGUAGE_NAMES.get(detected_code)
                detected_language_name = f"{name_flag[1]} {name_flag[0]}" if name_flag else detected_code

            if transcript:
                summary = summarize_text(transcript)
                key_points = [f"{i + 1}. {point.strip()}" for i, point in enumerate(summary.split('.')) if point.strip()]
                title = generate_title(transcript[:1000])

            audio_url = f"/static/{audio_filename}"

    return render_template("index.html",
                           transcript=transcript,
                           summary=summary,
                           key_points=key_points,
                           detected_language=detected_language_name,
                           audio_url=audio_url,
                           title=title)

@app.route("/resummarize", methods=["POST"])
def resummarize():
    summary = request.form.get("summary")
    print("Original summary received:", bool(summary))

    if not summary or summary.strip() == "":
        return jsonify({"error": "Summary is missing or empty"}), 400

    try:
        paraphrased = paraphrase_text(summary)
        key_points = [f"{i + 1}. {point.strip()}" for i, point in enumerate(paraphrased.split('.')) if point.strip()]
        return jsonify({
            "summary": paraphrased,
            "key_points": key_points
        })
    except Exception as e:
        print("❌ Error during paraphrasing:", str(e))
        return jsonify({"error": "Paraphrasing failed: " + str(e)}), 500

@app.route("/translate", methods=["POST"])
def translate():
    text = request.form.get("text")
    target_lang = request.form.get("target_lang")

    if not text or not target_lang:
        return jsonify({"error": "Missing text or target language"}), 400

    try:
        translated = translate_text(text, target_lang)
        return jsonify({"translated": translated})
    except Exception as e:
        print("❌ Translation failed:", str(e))
        return jsonify({"error": "Translation failed: " + str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
