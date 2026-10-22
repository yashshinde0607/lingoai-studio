# 🌐 LingoAI Studio
## Multilingual Text Processing and Analysis System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-8B5CF6?logo=python&logoColor=white)](https://www.python.org/)
[![Sarvam AI](https://img.shields.io/badge/Engine-Sarvam%20AI%20Indic%20Models-FF786B)](https://www.sarvam.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F4C95D.svg)](LICENSE)

An end-to-end natural language engineering platform tailored for Indian languages (**English**, **Hindi (हिंदी)**, and **Marathi (मराठी)**). Built as a final year college engineering exhibition project, **LingoAI Studio** unifies Devanagari regex preprocessing, high-accuracy neural translation, abstractive summarization, sentiment analysis with linguistic explanations, topic classification, named entity recognition (NER), and a rigorous mathematical evaluation framework.

---

## 🎨 Color Palette & Design System

The application interface follows a modern, academic exhibition design language adhering strictly to the warm visual identity:
- **Warm Ivory (`#FAF7F2`)**: Canvas and app background
- **White (`#FFFFFF`)**: Card containers and interactive surfaces
- **Charcoal (`#252522`)**: High-contrast typography and borders
- **Purple (`#8B5CF6`)**: Primary action controls, accents, and topic pills
- **Coral (`#FF786B`)**: Negative sentiment alerts and translation badges
- **Mint (`#BCE8D1`)**: Positive sentiment indicators and success states
- **Gold (`#F4C95D`)**: Neutral sentiment markers and warning badges

*(Strictly avoids blue, cyan, teal, navy, and blue-grey as per design specifications).*

---

## 🏗️ System Architecture & Workflow

```
┌────────────────────────────────────────────────────────┐
│                   Input Ingestion                      │
│   (English / Hindi / Marathi Text or Benchmark CSV)    │
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
│                 Sarvam AI Model Stack                  │
│  • mayura:v1 (Indic Neural Machine Translation)        │
│  • sarvam-105b-conversations (Multilingual LLM)       │
│  • identify_language (Indic Script Identification)     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│            Interactive Plotly Visualization            │
│  • Confusion Matrix Heatmaps   • Entity Donut Charts   │
│  • Script Composition Bars     • Word Frequency Plots  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                 Export & Audit Engine                  │
│       • Formatted TXT   • Structured JSON   • CSV      │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Modules & Capabilities

### 1. 🔍 Multilingual Text Analyzer
- **Devanagari Normalization:** Enforces Unicode NFC standard, normalizes zero-width non-joiners, converts pipe characters to traditional single danda (`।`) and double danda (`॥`), and cleans redundant spacing.
- **Deep Statistical Analytics:** Calculates character count (total vs. space-excluded), word count, sentence count, Type-Token Ratio (Lexical Richness %), average word length, and script breakdown (Devanagari vs. Latin vs. Numerals).
- **Sentiment Analysis:** Categorizes polarity into **Positive**, **Negative**, or **Neutral** accompanied by model confidence and in-depth native linguistic reasoning explaining *why* specific words triggered the classification.
- **Topic Classification:** Automatically maps documents into academic/industry categories (*Technology, Politics, Business, Sports, Entertainment, Healthcare, Education, Other*) with contextual explanations.
- **Named Entity Recognition (NER):** Extracts and visually chips identifiable entities (*PERSON, LOCATION, ORGANIZATION, DATE, OTHER*) with frequency distributions.
- **Plotly Visuals:** Character distribution bar charts and Top 10 frequent tokens bar charts.

### 2. 🔄 Neural Translation Studio
- Bidirectional translation across **English (`en-IN`)**, **Hindi (`hi-IN`)**, and **Marathi (`mr-IN`)**.
- Powered by Sarvam's flagship **`mayura:v1`** neural model.
- Features one-click source/target language swapping, pre-translation normalization, side-by-side text synchronization, and export.

### 3. 📝 Abstractive Summarization
- Condenses extensive multilingual articles into concise executive paragraphs or bulleted takeaways.
- Supports length customization (*Short, Medium, Detailed*) while preserving native script syntax.
- Displays real-time **Compression Ratio (%)** and word reduction statistics.

### 4. 📊 Mathematical Evaluation Framework
- **No Fabricated Metrics:** Computes authentic mathematical values directly from predictions against labeled ground truth.
- **Preloaded Benchmark Dataset:** Includes 15 diverse college-level test cases in English, Hindi, and Marathi.
- **Custom Dataset Upload:** Allows importing any CSV dataset containing text and ground truth labels.
- **Metrics Computed:**
  $$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Samples}}$$
  $$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$
  $$\text{Macro F1} = \frac{1}{|C|} \sum_{c \in C} 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
- **Confusion Matrix:** Rendered as an interactive Plotly heatmap with warm ivory-to-purple gradients.
- **Audit Table:** Filterable by "All" or "Misclassified Only" for error analysis during project presentations.

### 5. 📜 Session History & Export
- Logs every operation (*Analysis, Translation, Summarization, Evaluation*) in memory.
- Allows previewing historical payloads, viewing JSON inspection trees, clearing history, and exporting session activity to CSV.

---

## 💻 Local Setup & Execution (Windows PowerShell)

### Step 1: Clone or Open the Workspace
```powershell
cd "c:\Users\yash shinde\multilingual-nlp-system"
```

### Step 2: Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Verify Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure API Key in `.env`
Ensure your `.env` file contains your Sarvam AI subscription key:
```env
SARVAM_API_KEY=your_actual_sarvam_api_key_here
```
*(The key can also be entered or updated securely directly in the app's sidebar during live demonstrations).*

### Step 5: Launch the Streamlit Application
```powershell
streamlit run app.py
```
The application will open in your default browser at **`http://localhost:8501`**.

---

## ☁️ Deployment Guide

### A. GitHub Setup
1. Initialize git and commit the codebase (ensure `.env` and `.venv/` are ignored):
   ```powershell
   git init
   git add .
   git commit -m "Initial release of LingoAI Studio"
   ```
2. Create a repository on GitHub (e.g., `lingoai-studio`) and push your code:
   ```powershell
   git remote add origin https://github.com/your-username/lingoai-studio.git
   git branch -M main
   git push -u origin main
   ```

### B. Deploy on Streamlit Community Cloud
1. Sign in to [share.streamlit.io](https://share.streamlit.io).
2. Click **"New app"**, select your GitHub repository, branch (`main`), and set **Main file path** to `app.py`.
3. Expand **"Advanced settings"** and navigate to the **Secrets** section.
4. Add your Sarvam API Key securely:
   ```toml
   SARVAM_API_KEY = "your_actual_sarvam_api_key_here"
   ```
5. Click **"Deploy"**. Streamlit Cloud will build the environment and serve the live application with free HTTPS hosting.

---

## 📁 Repository Structure

```
multilingual-nlp-system/
├── app.py               # Streamlit application UI and flow
├── nlp_utils.py         # NLP engine, regex normalizers, Sarvam AI client, metrics
├── benchmark_data.py    # Preloaded sample texts and labeled benchmark dataset
├── requirements.txt     # Python package dependencies
├── .env                 # Environment credentials (excluded from git)
├── .gitignore           # Git ignore rules
└── README.md            # Comprehensive project documentation
```

---

## 🎓 College Exhibition Demo Tips

1. **Overview Tab:** Show the system architecture and explain the pipeline flow from Devanagari Unicode normalization to LLM reasoning.
2. **Text Analyzer:** Click **"Marathi Sample (मराठी)"**, click **"Analyze Text"**, and show the evaluator the Marathi entity chips and sentiment explanation.
3. **Translation Studio:** Load an English sample, select **Marathi** as target, and showcase the speed and fidelity of `mayura:v1`.
4. **Summarization:** Load a long Hindi passage, select **"Key Takeaways (Bullet Points)"**, and demonstrate the 60%+ compression ratio.
5. **Evaluation Framework:** Navigate to **"Evaluation"**, click **"Run Live Benchmark Evaluation"**, and demonstrate the real Plotly Confusion Matrix heatmap, explaining Precision, Recall, and Macro-F1.
