
# Inclusive Speech Recognition: A Real-Time Transcription System for Individuals with Speech Impediments

This project aims to build a real-time transcription system optimized for individuals with speech impediments, using OpenAI Whisper for transcription and a fine-tuned T5 model for correction.

---

## 📌 Project Overview

This project provides an inclusive speech-to-text transcription system by improving automatic speech recognition (ASR) performance for individuals with speech impediments like dysarthria. The approach combines **Whisper** for transcription and **T5** for error correction.

---

## 📂 Project Structure

```
InclusiveSpeechRecognition/
├── venv/                           # Python virtual environment (Not pushed to GitHub)
├── models/                         # Directory for trained models
│   └── t5_correction_model/        # Fine-tuned T5 model for transcription correction
├── main.py                         # Real-time transcription and correction script
├── train_correction_model.py       # Training script for fine-tuning T5 model
├── generate_correction_data.py     # Generates Whisper correction dataset
├── evaluate_corrections.py         # Evaluates model performance using WER and CER
├── whisper_correction_data.csv     # Generated training data (Ignored by .gitignore)
├── requirements.txt                # Python dependencies
├── README.md                       # Project description and usage guide
├── .gitignore                      # Files and folders to exclude from version control
```

---

## 🔍 Models Used

1. **Whisper (Base)**: For transcription.  
2. **T5-small (Fine-tuned)**: For correcting transcription errors.

---

## 📊 Evaluation Metrics

- **Word Error Rate (WER)**
- **Character Error Rate (CER)**

---

## 📦 Installation

Clone the repository and install dependencies.

```bash
git clone https://github.com/YourUsername/InclusiveSpeechRecognition.git
cd InclusiveSpeechRecognition
python3 -m venv venv
source venv/bin/activate  # On Windows, use .\venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Generate Correction Data
To prepare the training dataset from the TORGO dysarthric subset:
```bash
python generate_correction_data.py
```

### 2. Train the Model
To fine-tune the T5 model on the correction dataset:
```bash
python train_correction_model.py
```

### 3. Evaluate the Model
To calculate WER and CER on the test set:
```bash
python evaluate_corrections.py
```

### 4. Run the Real-Time Application
To use the trained model for real-time transcription and correction:
```bash
python main.py
```

---

## 📌 Future Improvements

- Adding a GUI using **Streamlit or Tkinter** for ease of use.
- Training on larger datasets for better performance.
- Model personalization for specific speakers with severe speech impediments.

---

## 💻 Author

- **Mohini Vashisth**  

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
