import pandas as pd
import evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load correction model
tokenizer = T5Tokenizer.from_pretrained("models/t5_correction_model")
model = T5ForConditionalGeneration.from_pretrained("models/t5_correction_model")

# Load dataset
df = pd.read_csv("whisper_correction_data.csv").dropna()

# Load metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# Store predictions and references
preds = []
refs = []

# Correction function
def correct_text(text):
    input_text = "fix: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=64, truncation=True)
    outputs = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Evaluate all samples
print("🔍 Running evaluation...")
for _, row in df.iterrows():
    whisper_output = row['whisper_output']
    ground_truth = row['ground_truth']
    corrected = correct_text(whisper_output)
    
    preds.append(corrected)
    refs.append(ground_truth)

# Compute WER and CER
wer = wer_metric.compute(predictions=preds, references=refs)
cer = cer_metric.compute(predictions=preds, references=refs)

print(f"✅ WER: {wer:.3f}")
print(f"✅ CER: {cer:.3f}")