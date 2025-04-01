from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd

# Load CSV
df = pd.read_csv("whisper_correction_data.csv")
df = df.dropna()

# Format for T5: add "fix: " before input
inputs = ["fix: " + i for i in df["whisper_output"].tolist()]
targets = df["ground_truth"].tolist()

# Create Hugging Face Dataset
dataset = Dataset.from_dict({"input_text": inputs, "target_text": targets})

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Tokenization function
def preprocess(example):
    model_inputs = tokenizer(example["input_text"], max_length=64, truncation=True, padding="max_length")
    labels = tokenizer(example["target_text"], max_length=64, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the dataset
tokenized = dataset.map(preprocess)

# Define training arguments
args = TrainingArguments(
    output_dir="models/t5_correction_model",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    save_strategy="epoch",
    evaluation_strategy="no",
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized
)

# Fine-tune the model
trainer.train()

# Save the model
trainer.save_model("models/t5_correction_model")
print("✅ Correction model saved to models/t5_correction_model")