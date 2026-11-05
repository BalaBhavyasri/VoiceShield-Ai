# 🛡️ VoiceShield AI: Voice Scam Threat Intelligence Platform
**An AI-Powered Voice Scam Detection, Tactic Analysis, and Regional Awareness System**
*Submission for the Great Indian AI Internship Challenge*

---

## 📌 Executive Summary
Phone scams in India have evolved rapidly—from standard bank KYC fraud to high-pressure **"Digital Arrest"** extortion, **AnyDesk remote device takeovers**, and **fake courier narcotics detention**. Furthermore, fraudulent calls in India frequently leverage **code-mixed regional languages** (such as Telugu-English *Tanglish* and Hindi-English *Hinglish*).

**VoiceShield AI** addresses these challenges directly:
1. **Voice Scam Analysis**: Converts consented call recordings or live voice calls into transcripts using the **Gnani AI Vachana Speech-to-Text API**.
2. **Multi-Vector Scam Tactic Extraction**: Analyzes transcripts against 7 critical attack vectors (OTP theft, payment pressure, digital arrest, remote access, impersonation, lottery fraud, parcel release).
3. **Indian-Language Accessibility**: Features deep support for English, **Telugu-English (Tanglish)**, and **Hindi-English (Hinglish)**.
4. **Explainable Warnings**: Highlights the exact suspicious phrases in the transcript and explains the underlying social engineering tactic.
5. **Spoken Safety Guidance**: Reads out spoken safety advice in **Telugu**, **Hindi**, and **English** using Text-to-Speech so that users of any literacy level can protect themselves.
6. **Hybrid Machine Learning Architecture**: Fuses rule-based regex heuristics with a lightweight NumPy TF-IDF classifier, achieving **99.1% benchmark accuracy** across 108 Kaggle-scale test cases.
7. **Privacy by Design**: Explicit consent confirmation, ephemeral audio processing, and zero retention of raw audio files.

---

## 🏗️ Technology Stack

| Component | Technology / Library |
| :--- | :--- |
| **Frontend** | Streamlit (Custom Glassmorphism Cybersecurity Theme) |
| **Backend** | Python 3.10 |
| **Speech-to-Text (STT)** | **Gnani AI Vachana REST API v3** (`api.vachana.ai/stt/v3`) with local fallback |
| **Text-to-Speech (TTS)** | **Gnani AI TTS** + multilingual `gTTS` audio synthesis (Telugu, Hindi, English) |
| **Scam Detection** | **Hybrid Ensemble**: Multilingual Rule Heuristics + NumPy TF-IDF Naive Bayes ML |
| **Interactive Visualizations** | Plotly (Threat Vector Radar, Score Gauge, Confusion Matrix, Language Breakdown) |
| **Data Storage** | Local SQLite (`voiceshield.db`) for privacy-first audit logging (no raw audio stored) |
| **Audio Processing** | `imageio-ffmpeg` (16kHz mono PCM conversion) & `wave` |

---

## 📁 Repository Structure

```
C:\Users\VENKATASUBBA\Desktop\VoiceShieldAI\
├── app.py                      # Main Streamlit application
├── speech.py                   # Gnani STT API client, audio converter & TTS synthesizer
├── detector.py                 # Multilingual rule heuristics (English, Tanglish, Hinglish)
├── ml_detector.py              # Lightweight NumPy TF-IDF + Naive Bayes ML classifier
├── audio_utils.py              # Uncertainty handling & audio quality inspection
├── database.py                 # SQLite privacy-first audit store
├── requirements.txt            # Python dependencies
├── .env                        # Gnani AI API credentials & settings
├── test_detector.py            # Comprehensive unit test suite
├── evaluate_detector.py        # Benchmark evaluation script
│
├── evaluation/
│   ├── scam_test_data.csv      # Kaggle-style 108 balanced multi-lingual call transcripts
│   ├── evaluate_detector.py    # Subgroup benchmarking script
│   └── evaluation_results.csv  # Detailed per-sample evaluation matrix
│
└── archive/                    # Archived legacy backups
```

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites
Ensure you have Python 3.10 installed on your system.

### 2. Configure Environment Variables (`.env`)
Your `.env` file should contain your Gnani AI API credentials:
```env
STT_PROVIDER=gnani
GNANI_STT_API_KEY_ID=your_gnani_vachana_key_here
GNANI_STT_URL=https://api.vachana.ai/stt/v3
GNANI_LANGUAGE_CODE=en-IN
GNANI_TTS_URL=https://api.gnani.ai/v1/tts/inference
GNANI_TTS_VOICE=Yashvi
```

### 3. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
The application will launch automatically in your browser at `http://localhost:8501`.

### 5. Run Automated Unit Tests & Benchmarks
```bash
# Run unit tests (all 9 tests verifying heuristics, Tanglish, and ML)
python test_detector.py

# Run benchmark evaluation across 108 multi-lingual samples
python evaluation/evaluate_detector.py
```

---

## 📊 Evaluation & Benchmark Results

Evaluated on **108 balanced call transcripts** spanning English, Telugu-English, and Hindi-English:

| Metric | Rule Baseline | Hybrid Model (Rule + ML) | Improvement |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 89.8% | **99.1%** | **+9.3%** |
| **Precision** | 0.980 | **0.984** | +0.004 |
| **Recall (Sensitivity)** | 0.833 | **1.000** | **+16.7% (Zero missed scams)** |
| **Specificity** | 0.979 | **0.979** | High true negative rate |
| **False Positive Rate (FPR)** | 0.021 | **0.021** | Very low false alarms |
| **F1-Score** | 0.901 | **0.992** | **+0.091** |

### Language Subgroup Breakdown
- **English**: 100.0% Accuracy | 1.000 Recall | 1.000 Precision (N=63)
- **Telugu-English (Tanglish)**: 96.2% Accuracy | 1.000 Recall | 0.938 Precision (N=26)
- **Hindi-English (Hinglish)**: 100.0% Accuracy | 1.000 Recall | 1.000 Precision (N=19)

---

## 🔒 Privacy & Safety Principles
1. **No Raw Audio Retention**: Temporary audio files are converted into 16kHz PCM WAV in memory/ephemeral storage, transcribed via Gnani STT, and immediately wiped.
2. **Explicit Consent Gate**: Users must confirm that recordings are submitted with consent for scam prevention.
3. **Calibrated Labeling**: Outputs are transparently labeled as **"potentially suspicious"** rather than definitively fraudulent, preventing false accusations in ambiguous business interactions.
4. **Data Purge Capability**: A single-click **"Purge Audit History"** button allows complete removal of local session logs.
