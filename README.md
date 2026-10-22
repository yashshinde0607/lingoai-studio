# 🎙️ LingoAI Studio
## Multilingual Text Processing and Analysis System Using Voice Commands

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-8B5CF6?logo=python&logoColor=white)](https://www.python.org/)
[![Sarvam AI](https://img.shields.io/badge/Engine-Sarvam%20AI%20Indic%20Models-FF786B)](https://www.sarvam.ai/)
[![Voice Pipeline](https://img.shields.io/badge/Voice-STT%20%2B%20TTS-BCE8D1)](https://www.sarvam.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F4C95D.svg)](LICENSE)

An end-to-end natural language engineering and voice processing platform tailored for Indian languages (**English**, **Hindi (हिंदी)**, and **Marathi (मराठी)**). Built as a final year college engineering exhibition project, **LingoAI Studio** unifies voice commands, speech-to-text transcription, text-to-speech synthesis, Devanagari regex preprocessing, high-accuracy neural translation, abstractive summarization, conversational AI, sentiment analysis with linguistic explanations, topic classification, named entity recognition (NER), and an authentic mathematical evaluation framework.

---

## 🎨 Color Palette & Visual Identity

The user interface follows a modern academic exhibition design language adhering strictly to the warm visual theme:
- **Warm Ivory (`#FAF7F2`)**: Base canvas & app background
- **White (`#FFFFFF`)**: Interactive card surfaces & containers
- **Charcoal (`#252522`)**: Typography and borders
- **Purple (`#8B5CF6`)**: Primary action buttons, topic pills, and focus states
- **Coral (`#FF786B`)**: Negative sentiment alerts, word count highlights
- **Mint (`#BCE8D1`)**: Positive sentiment indicators and success states
- **Gold (`#F4C95D`)**: Neutral sentiment markers and warning indicators

*(Strictly avoids blue, cyan, teal, navy, and blue-grey as per design specifications).*

---

## 🏗️ System Architecture & Voice Pipeline

```
┌────────────────────────────────────────────────────────┐
│                   Audio / Text Input                   │
│   (Browser Microphone, WAV/MP3 Audio, or Typed Text)   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│             Sarvam Speech-to-Text (STT)                │
│    (saaras:v3 Multilingual Transcription & Language ID) │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│          Devanagari & Latin Regex Normalizer           │
│     (NFC Normalization, Danda/Punctuation Repair,      │
│         Whitespace Collapse, Token Extraction)         │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│            Sarvam AI Conversational & NLP Engine       │
│  • sarvam-105b-conversations (Chatbot & Commands)      │
│  • mayura:v1 (Indic Neural Machine Translation)        │
│  • Abstractive Summarizer & Sentiment Polarity         │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│             Sarvam Text-to-Speech (TTS)                │
│  • bulbul:v3 Indic Voice Synthesis (WAV Audio Output)   │
│  • Real-time Audio Playback & Download                 │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│            Interactive Plotly Visualization            │
│  • Confusion Matrix Heatmaps   • Entity Donut Charts   │
│  • Script Composition Bars     • Word Frequency Plots  │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Modules & Capabilities

### 1. 🎙️ Voice Assistant & Multilingual Chatbot
- **Voice Ingestion:** Speak voice commands directly using the browser microphone (`st.audio_input`) or upload audio files.
- **Speech-to-Text (`saaras:v3`):** Transcribes Devanagari and Indian English audio with high fidelity.
- **Intent Execution:** Understands voice commands such as:
  - *"Translate 'Knowledge is power' to Marathi"*
  - *"Analyze the sentiment of 'हे ॲप खरोखरच खूप छान आणि उपयुक्त आहे!'"*
  - *"Summarize: [text]"*
- **Conversational Chatbot (`sarvam-105b-conversations`):** Natural dialogue in English, Hindi, and Marathi.
- **Voice Playback (`bulbul:v3`):** Every AI response includes a **"🔊 Listen to Response (TTS)"** button to synthesize speech in the native language.

### 2. 🔊 Voice Studio (Dedicated STT & TTS Hub)
- **Speech-to-Text (STT):** Convert recorded speech or uploaded `.wav`/`.mp3` files into clean text with auto language detection.
- **Text-to-Speech (TTS):** Convert text into natural speech with multiple Indic personas (`aditya`, `priya`, `ritu`, `neha`, `rahul`, `rohan`, `simran`, `kavya`, `soham`), pace adjustment, and downloadable `.wav` files.

### 3. 🔍 Multilingual Text Analyzer
- **Devanagari Normalization:** Enforces Unicode NFC standard, normalizes zero-width non-joiners, converts pipe characters to traditional single danda (`।`) and double danda (`॥`), and cleans redundant spacing.
- **Deep Statistical Analytics:** Calculates character count, word count, sentence count, Type-Token Ratio (Lexical Richness %), average word length, and script breakdown.
- **Sentiment Analysis:** Categorizes polarity into **Positive**, **Negative**, or **Neutral** accompanied by model confidence and in-depth native linguistic reasoning.
- **Topic Classification:** Automatically maps documents into academic/industry categories (*Technology, Politics, Business, Sports, Entertainment, Healthcare, Education, Other*) with contextual explanations.
- **Named Entity Recognition (NER):** Extracts and visually chips identifiable entities (*PERSON, LOCATION, ORGANIZATION, DATE, OTHER*).
- **Voice Output:** One-click **"🔊 Read Analyzed Text Aloud (TTS)"** audio playback.

### 4. 🔄 Neural Translation Studio
- Bidirectional translation across **English (`en-IN`)**, **Hindi (`hi-IN`)**, and **Marathi (`mr-IN`)**.
- Powered by Sarvam's flagship **`mayura:v1`** neural model.
- Features one-click source/target language swapping, pre-translation normalization, side-by-side text synchronization, and **"🔊 Speak Translation"** voice playback.

### 5. 📝 Abstractive Summarization & Audio
- Condenses extensive multilingual articles into concise executive paragraphs or bulleted takeaways.
- Supports length customization (*Short, Medium, Detailed*) while preserving native script syntax.
- Displays real-time **Compression Ratio (%)** and word reduction statistics with voice synthesis.

### 6. 📊 Mathematical Evaluation Framework
- **No Fabricated Metrics:** Computes authentic mathematical values directly from predictions against labeled ground truth.
- **Preloaded Benchmark Dataset:** Includes 15 diverse college-level test cases in English, Hindi, and Marathi.
- **Custom Dataset Upload:** Allows importing any CSV dataset containing text and ground truth labels.
- **Metrics Computed:**
  $$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Samples}}$$
  $$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$
  $$\text{Macro F1} = \frac{1}{|C|} \sum_{c \in C} 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
- **Confusion Matrix:** Rendered as an interactive Plotly heatmap with warm ivory-to-purple gradients.
- **Audit Table:** Filterable by "All" or "Misclassified Only" for error analysis during project presentations.

### 7. 📜 Session History & Export
- Logs every operation (*Voice Commands, Analysis, Translation, Summarization, Evaluation*) in memory.
- Allows previewing historical payloads, viewing JSON inspection trees, clearing history, and exporting session activity to CSV.

---

## 💻 Local Setup & Execution (Windows PowerShell)

```powershell
# 1. Navigate to the project directory
cd "c:\Users\yash shinde\multilingual-nlp-system"

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Launch the Streamlit application
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## ☁️ Deployment Guide (GitHub & Streamlit Community Cloud)

1. **Upload Files to GitHub:**
   - Create a repository on [github.com](https://github.com) named `lingoai-studio`.
   - Upload: `app.py`, `nlp_utils.py`, `benchmark_data.py`, `requirements.txt`, `README.md`, `Dockerfile`, `Procfile`.
   - *Never upload `.env` or `.venv/`.*
2. **Deploy to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io) ➔ Click **"New app"**.
   - Select your repository and set Main file path to `app.py`.
   - Under **Advanced settings ➔ Secrets**, add:
     ```toml
     SARVAM_API_KEY = "your_actual_sarvam_api_key_here"
     ```
   - Click **Deploy!**
