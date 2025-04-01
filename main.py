import sounddevice as sd
import numpy as np
import whisper
import scipy.io.wavfile as wavfile
import os

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# ========== SETTINGS ==========
duration = 5  # Seconds to record
samplerate = 16000
filename = "recorded_audio.wav"

# ========== DEVICE SETUP ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ========== LOAD MODELS ==========
try:
    print("🔁 Loading Whisper...")
    whisper_model = whisper.load_model("base")

    print("🧠 Loading correction model...")
    tokenizer = T5Tokenizer.from_pretrained("models/t5_correction_model")
    t5_model = T5ForConditionalGeneration.from_pretrained("models/t5_correction_model").to(device)
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# ========== RECORD AUDIO ==========
try:
    print("🎤 Start speaking...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording finished.")
except Exception as e:
    print(f"❌ Error during audio recording: {e}")
    exit()

# ========== SAVE AUDIO ==========
try:
    wavfile.write(filename, samplerate, recording)
except Exception as e:
    print(f"❌ Error saving audio file: {e}")
    exit()

# ========== WHISPER TRANSCRIPTION ==========
try:
    print("🔍 Transcribing with Whisper...")
    result = whisper_model.transcribe(filename)
    raw_text = result["text"].strip()
    print(f"\n📝 Whisper Output: {raw_text}")
except Exception as e:
    print(f"❌ Error during Whisper transcription: {e}")
    os.remove(filename)
    exit()

# ========== CORRECT WITH T5 MODEL ==========
def correct_text(text):
    try:
        input_text = "fix: " + text
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=64, truncation=True).to(device)
        outputs = t5_model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"❌ Error during text correction: {e}")
        return text  # If correction fails, return the original transcription

# Generate corrected text
corrected_text = correct_text(raw_text)
print(f"✨ Corrected Transcription: {corrected_text}\n")

# ========== CLEANUP ==========
try:
    os.remove(filename)
except Exception as e:
    print(f"❌ Error deleting audio file: {e}")