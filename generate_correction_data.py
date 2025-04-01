from datasets import load_dataset, Audio
import whisper
import pandas as pd
from tqdm import tqdm
import os
from scipy.io.wavfile import write
import logging

logging.basicConfig(filename='error_log.txt', level=logging.ERROR)

# Load Whisper model
print("🔁 Loading Whisper model...")
model = whisper.load_model("base")

# Load TORGO dataset
print("📥 Loading TORGO dysarthric subset...")
dataset = load_dataset("abnerh/TORGO-database")
print("🔎 First few samples:")
for i in range(5):
    print(dataset['train'][i]['speech_status'])
dataset = dataset['train'].filter(lambda x: x['speech_status'] == 'dysarthria')

# Prepare audio for Whisper
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Create CSV list
data_pairs = []

# Directory to store temporary WAV files
os.makedirs("temp_audio", exist_ok=True)

print("🧠 Running Whisper on samples...")

for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
    try:
        audio = sample['audio']['array']
        transcription = sample['transcription']
        temp_path = f"temp_audio/sample_{i}.wav"

        # Save audio temporarily
        import numpy as np
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
        write(temp_path, 16000, audio)

        # Transcribe using Whisper
        result = model.transcribe(temp_path)
        whisper_output = result['text']

        # Add to correction dataset
        data_pairs.append({
            'whisper_output': whisper_output.strip(),
            'ground_truth': transcription.strip()
        })

        os.remove(temp_path)

    except Exception as e:
        logging.error(f"Error at sample {i}: {e}")

# Cleanup temporary files
import shutil
shutil.rmtree("temp_audio", ignore_errors=True)

# Save to CSV
df = pd.DataFrame(data_pairs)
df.to_csv("whisper_correction_data.csv", index=False)

print("✅ Correction dataset saved as whisper_correction_data.csv")