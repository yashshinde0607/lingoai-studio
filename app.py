"""
app.py
LingoAI Studio: Multilingual Text Processing and Analysis System
A college-level NLP engineering project powered by Streamlit and Sarvam AI.
"""

import os
import sys
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# Local imports
import nlp_utils
import benchmark_data

# Ensure UTF-8 output
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==============================================================================

st.set_page_config(
    page_title="LingoAI Studio | Multilingual NLP System",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS adhering STRICTLY to the requested palette:
# Warm ivory (#FAF7F2), White (#FFFFFF), Charcoal (#252522), Purple (#8B5CF6),
# Coral (#FF786B), Mint (#BCE8D1), Gold (#F4C95D).
# Strictly avoiding blue, cyan, teal, navy, and blue-grey.
CUSTOM_CSS = """
<style>
/* Global App Background and Text */
.stApp {
    background-color: #FAF7F2;
    color: #252522;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #EBE5DC;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
    color: #252522;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #252522 !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

/* Card Container */
.lingo-card {
    background-color: #FFFFFF;
    border: 1px solid #EAE3D9;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 4px 12px rgba(37, 37, 34, 0.04);
    margin-bottom: 1.2rem;
}

/* Hero Section */
.hero-box {
    background: linear-gradient(135deg, #FFFFFF 0%, #FAF7F2 100%);
    border: 1px solid #E2D9CC;
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    box-shadow: 0 8px 24px rgba(37, 37, 34, 0.05);
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #252522;
    margin-bottom: 0.35rem;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: #635E55;
    font-weight: 500;
    margin-bottom: 1.2rem;
}

.hero-badge {
    display: inline-block;
    background-color: #8B5CF6;
    color: #FFFFFF;
    font-size: 0.8rem;
    font-weight: 700;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Metric Display Cards */
.metric-box {
    background-color: #FFFFFF;
    border: 1px solid #EBE4DA;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    box-shadow: 0 2px 6px rgba(37, 37, 34, 0.03);
}

.metric-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: #8B5CF6;
    margin-bottom: 0.2rem;
}

.metric-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #706B62;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

/* Sentiment Badges */
.badge-sentiment-positive {
    display: inline-block;
    background-color: #BCE8D1;
    color: #164E33;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.4rem 1.2rem;
    border-radius: 30px;
    border: 1px solid #9FD4B8;
}

.badge-sentiment-negative {
    display: inline-block;
    background-color: #FF786B;
    color: #FFFFFF;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.4rem 1.2rem;
    border-radius: 30px;
    border: 1px solid #E86558;
}

.badge-sentiment-neutral {
    display: inline-block;
    background-color: #F4C95D;
    color: #403000;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.4rem 1.2rem;
    border-radius: 30px;
    border: 1px solid #DEB140;
}

/* Topic Badge */
.badge-topic {
    display: inline-block;
    background-color: #8B5CF6;
    color: #FFFFFF;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 0.35rem 1rem;
    border-radius: 20px;
}

/* Entity Chips */
.entity-chip {
    display: inline-block;
    margin: 0.25rem 0.35rem;
    padding: 0.3rem 0.75rem;
    border-radius: 8px;
    font-size: 0.88rem;
    font-weight: 600;
    border: 1px solid transparent;
}
.entity-PERSON { background-color: #BCE8D1; color: #164E33; border-color: #A2D9BC; }
.entity-LOCATION { background-color: #F4C95D; color: #403000; border-color: #E0B445; }
.entity-ORGANIZATION { background-color: #8B5CF6; color: #FFFFFF; }
.entity-DATE { background-color: #FF786B; color: #FFFFFF; }
.entity-OTHER { background-color: #FAF7F2; color: #252522; border-color: #D6CEC1; }

/* Buttons Customization */
button[kind="primary"] {
    background-color: #8B5CF6 !important;
    border-color: #8B5CF6 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease-in-out !important;
}

button[kind="primary"]:hover {
    background-color: #7C3AED !important;
    border-color: #7C3AED !important;
}

button[kind="secondary"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D8CFC2 !important;
    color: #252522 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

button[kind="secondary"]:hover {
    border-color: #8B5CF6 !important;
    color: #8B5CF6 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #FFFFFF;
    border-radius: 8px 8px 0 0;
    color: #252522;
    padding: 8px 16px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: #FAF7F2 !important;
    border-bottom: 3px solid #8B5CF6 !important;
    color: #8B5CF6 !important;
}

/* Plotly chart container styling */
.js-plotly-plot {
    border-radius: 10px;
    overflow: hidden;
}

/* Footer note */
.app-footer {
    text-align: center;
    color: #857F75;
    font-size: 0.85rem;
    padding: 2.5rem 0 1rem 0;
    border-top: 1px solid #EBE5DC;
    margin-top: 3rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. SESSION STATE MANAGEMENT
# ==============================================================================

if "history" not in st.session_state:
    st.session_state["history"] = []

if "text_analyzer_input" not in st.session_state:
    st.session_state["text_analyzer_input"] = benchmark_data.SAMPLE_TEXTS["Marathi"]["text"]

if "translation_input" not in st.session_state:
    st.session_state["translation_input"] = benchmark_data.SAMPLE_TEXTS["English"]["text"]

if "summarization_input" not in st.session_state:
    st.session_state["summarization_input"] = benchmark_data.SAMPLE_TEXTS["Hindi"]["text"]


def log_history_entry(op_type: str, input_text: str, output_summary: str, details: dict):
    """Logs an operation into the session history list."""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operation": op_type,
        "input_preview": (input_text[:120] + "...") if len(input_text) > 120 else input_text,
        "output_summary": output_summary,
        "input_length": len(input_text),
        "details": details
    }
    st.session_state["history"].insert(0, entry)


# ==============================================================================
# 3. SIDEBAR NAVIGATION & SETTINGS
# ==============================================================================

with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem;">
        <span style="font-size: 2rem;">🌐</span>
        <div>
            <h2 style="margin: 0; font-size: 1.35rem; color: #252522;">LingoAI Studio</h2>
            <p style="margin: 0; font-size: 0.8rem; color: #736D64;">Multilingual NLP System</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px solid #EAE3D9;'>", unsafe_allow_html=True)
    
    # Navigation Radio
    navigation_options = [
        "🏛️ Overview",
        "🔍 Text Analyzer",
        "🔄 Translation",
        "📝 Summarization",
        "📊 Evaluation",
        "📜 History"
    ]
    
    selected_page = st.radio(
        "Navigation",
        navigation_options,
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("<hr style='margin: 1.2rem 0; border: none; border-top: 1px solid #EAE3D9;'>", unsafe_allow_html=True)
    
    # API Configuration
    st.markdown("<h4 style='font-size: 0.95rem; margin-bottom: 0.4rem;'>🔑 API Configuration</h4>", unsafe_allow_html=True)
    default_key = os.getenv("SARVAM_API_KEY", "")
    has_env_key = bool(default_key and default_key.strip())
    
    custom_key = st.text_input(
        "Sarvam AI Subscription Key",
        value=default_key,
        type="password",
        help="Reads SARVAM_API_KEY from environment/.env by default. Never exposed in code or commits."
    )
    
    if custom_key and custom_key.strip():
        st.markdown(
            "<span style='background-color: #BCE8D1; color: #164E33; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 12px;'>● API Key Active</span>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<span style='background-color: #FF786B; color: #FFFFFF; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 12px;'>● Key Not Configured</span>",
            unsafe_allow_html=True
        )

    st.markdown("<hr style='margin: 1.2rem 0; border: none; border-top: 1px solid #EAE3D9;'>", unsafe_allow_html=True)
    
    # College Project Metadata
    st.markdown("""
    <div style="font-size: 0.8rem; color: #6E685E; line-height: 1.45;">
        <strong style="color: #252522;">Project Information:</strong><br>
        • <strong>College Exhibition:</strong> Final Year NLP Engineering<br>
        • <strong>Supported Languages:</strong> English, Hindi (हिंदी), Marathi (मराठी)<br>
        • <strong>Engine:</strong> Sarvam AI Indic Models (`mayura:v1`, `sarvam-105b`)<br>
        • <strong>UI Stack:</strong> Streamlit + Plotly
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 4. COLOR PALETTE HELPERS FOR PLOTLY
# ==============================================================================
THEME_PURPLE = "#8B5CF6"
THEME_CORAL = "#FF786B"
THEME_MINT = "#BCE8D1"
THEME_GOLD = "#F4C95D"
THEME_CHARCOAL = "#252522"
THEME_BG = "#FAF7F2"
THEME_WHITE = "#FFFFFF"

PALETTE_SEQUENCE = [THEME_PURPLE, THEME_CORAL, THEME_GOLD, THEME_MINT, "#D97706", "#7C3AED", "#E11D48"]


# ==============================================================================
# 5. PAGE: OVERVIEW
# ==============================================================================
if "Overview" in selected_page:
    st.markdown("""
    <div class="hero-box">
        <span class="hero-badge">🎓 Engineering Exhibition Project</span>
        <div class="hero-title">LingoAI Studio</div>
        <div class="hero-subtitle">Multilingual Text Processing and Analysis System for Indian Languages</div>
        <p style="color: #4A463F; font-size: 1.05rem; line-height: 1.6; max-width: 900px; margin-bottom: 0;">
            LingoAI Studio is an end-to-end natural language engineering platform tailored for Indian languages.
            Supporting <strong>English</strong>, <strong>Hindi (हिंदी)</strong>, and <strong>Marathi (मराठी)</strong>, 
            it unifies Devanagari regex preprocessing, high-accuracy neural translation, abstractive summarization, 
            sentiment classification with linguistic reasoning, topic modeling, and named entity extraction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Architecture Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">3</div>
            <div class="metric-label">Languages Supported</div>
            <div style="font-size: 0.75rem; color: #8A847A; margin-top: 4px;">English • Hindi • Marathi</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">6</div>
            <div class="metric-label">NLP Pipeline Tasks</div>
            <div style="font-size: 0.75rem; color: #8A847A; margin-top: 4px;">Preprocess, Translate, Summarize, NER, Sentiment, Topic</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #FF786B;">Mayura</div>
            <div class="metric-label">Translation Engine</div>
            <div style="font-size: 0.75rem; color: #8A847A; margin-top: 4px;">Indic Neural Model (`mayura:v1`)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #F4C95D;">105B</div>
            <div class="metric-label">Multilingual LLM</div>
            <div style="font-size: 0.75rem; color: #8A847A; margin-top: 4px;">Sarvam AI Conversational Model</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Capabilities Grid
    st.markdown("### 🛠️ Core Capabilities & Features")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="lingo-card">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 1.4rem;">🧹</span>
                <h4 style="margin: 0; font-size: 1.1rem;">1. Devanagari Text Preprocessing & Analytics</h4>
            </div>
            <p style="color: #57524A; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                Normalizes Unicode (NFC), cleans extraneous whitespace, handles Devanagari danda (।) and double danda (॥), 
                and computes deep statistical metrics including Lexical Richness (Type-Token Ratio), script composition, 
                and sentence length dynamics.
            </p>
        </div>
        <div class="lingo-card">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 1.4rem;">🔄</span>
                <h4 style="margin: 0; font-size: 1.1rem;">2. Neural Multilingual Translation</h4>
            </div>
            <p style="color: #57524A; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                Seamless bidirectional translation between English, Hindi, and Marathi utilizing Sarvam's flagship 
                <code>mayura:v1</code> translation model, preserving grammatical subtleties and native vocabulary.
            </p>
        </div>
        <div class="lingo-card">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 1.4rem;">📝</span>
                <h4 style="margin: 0; font-size: 1.1rem;">3. Abstractive Summarization</h4>
            </div>
            <p style="color: #57524A; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                Distills extensive paragraphs into executive summaries or actionable bullet points with controllable 
                conciseness levels while maintaining native script fidelity and quantifying word reduction percentage.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="lingo-card">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 1.4rem;">💡</span>
                <h4 style="margin: 0; font-size: 1.1rem;">4. Sentiment Analysis with Linguistic Reasoning</h4>
            </div>
            <p style="color: #57524A; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                Classifies emotional polarity as Positive, Negative, or Neutral with confidence scoring and detailed 
                linguistic explanations explaining precisely which native terms influenced the judgment.
            </p>
        </div>
        <div class="lingo-card">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 1.4rem;">🏷️</span>
                <h4 style="margin: 0; font-size: 1.1rem;">5. Topic Modeling & Named Entity Recognition</h4>
            </div>
            <p style="color: #57524A; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                Categorizes texts into standard academic/industry topics (Technology, Politics, Business, Sports, etc.) 
                and tags named entities (Persons, Organizations, Locations, Dates) across Devanagari and Latin scripts.
            </p>
        </div>
        <div class="lingo-card">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 1.4rem;">📊</span>
                <h4 style="margin: 0; font-size: 1.1rem;">6. Mathematical Evaluation Framework</h4>
            </div>
            <p style="color: #57524A; font-size: 0.92rem; line-height: 1.5; margin: 0;">
                Benchmarking engine with preloaded college demonstration datasets. Calculates authentic mathematical 
                metrics: Confusion Matrix, Accuracy, Precision, Recall, and Macro-F1 without fabrication.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture Workflow Box
    st.markdown("### 🏗️ System Architecture & Data Flow")
    st.markdown("""
    <div class="lingo-card" style="padding: 1.8rem;">
        <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; text-align: center;">
            <div style="flex: 1; min-width: 140px; background-color: #FAF7F2; padding: 12px; border-radius: 8px; border: 1px solid #EBE5DC;">
                <strong style="color: #252522;">1. Input Ingestion</strong><br>
                <span style="font-size: 0.78rem; color: #706B62;">English / Hindi / Marathi Text or CSV</span>
            </div>
            <span style="font-size: 1.5rem; color: #8B5CF6;">➔</span>
            <div style="flex: 1; min-width: 140px; background-color: #FAF7F2; padding: 12px; border-radius: 8px; border: 1px solid #EBE5DC;">
                <strong style="color: #252522;">2. Regex Normalizer</strong><br>
                <span style="font-size: 0.78rem; color: #706B62;">NFC, Danda & Whitespace Standardization</span>
            </div>
            <span style="font-size: 1.5rem; color: #8B5CF6;">➔</span>
            <div style="flex: 1; min-width: 140px; background-color: #FAF7F2; padding: 12px; border-radius: 8px; border: 1px solid #EBE5DC;">
                <strong style="color: #252522;">3. Sarvam AI Models</strong><br>
                <span style="font-size: 0.78rem; color: #706B62;">`mayura:v1` & `sarvam-105b`</span>
            </div>
            <span style="font-size: 1.5rem; color: #8B5CF6;">➔</span>
            <div style="flex: 1; min-width: 140px; background-color: #FAF7F2; padding: 12px; border-radius: 8px; border: 1px solid #EBE5DC;">
                <strong style="color: #252522;">4. Analytics & Plotly</strong><br>
                <span style="font-size: 0.78rem; color: #706B62;">Interactive Charts & Metric Badges</span>
            </div>
            <span style="font-size: 1.5rem; color: #8B5CF6;">➔</span>
            <div style="flex: 1; min-width: 140px; background-color: #FAF7F2; padding: 12px; border-radius: 8px; border: 1px solid #EBE5DC;">
                <strong style="color: #252522;">5. Export & Audit</strong><br>
                <span style="font-size: 0.78rem; color: #706B62;">Session History & CSV/TXT Downloads</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 6. PAGE: TEXT ANALYZER
# ==============================================================================
elif "Text Analyzer" in selected_page:
    st.markdown("## 🔍 Multilingual Text Analyzer")
    st.markdown("Analyze grammatical structure, script statistics, sentiment polarity, topic category, and named entities.")

    # Sample Text Loader Buttons
    st.markdown("<div style='font-size: 0.88rem; font-weight: 600; color: #696359; margin-bottom: 6px;'>LOAD EXHIBITION SAMPLES:</div>", unsafe_allow_html=True)
    c_s1, c_s2, c_s3, c_s4 = st.columns([1, 1, 1, 1])
    with c_s1:
        if st.button("📄 English Sample", use_container_width=True):
            st.session_state["text_analyzer_input"] = benchmark_data.SAMPLE_TEXTS["English"]["text"]
            st.rerun()
    with c_s2:
        if st.button("📄 Hindi Sample (हिंदी)", use_container_width=True):
            st.session_state["text_analyzer_input"] = benchmark_data.SAMPLE_TEXTS["Hindi"]["text"]
            st.rerun()
    with c_s3:
        if st.button("📄 Marathi Sample (मराठी)", use_container_width=True):
            st.session_state["text_analyzer_input"] = benchmark_data.SAMPLE_TEXTS["Marathi"]["text"]
            st.rerun()
    with c_s4:
        if st.button("🗑️ Clear Text", use_container_width=True):
            st.session_state["text_analyzer_input"] = ""
            st.rerun()

    # Preprocessing Options Expander
    with st.expander("⚙️ Text Preprocessing Controls", expanded=False):
        c_opt1, c_opt2, c_opt3, c_opt4 = st.columns(4)
        with c_opt1:
            clean_spaces = st.checkbox("Normalize Whitespace", value=True)
        with c_opt2:
            norm_devanagari = st.checkbox("Devanagari Normalization (NFC / Danda)", value=True)
        with c_opt3:
            std_punct = st.checkbox("Standardize Quotes & Dashes", value=True)
        with c_opt4:
            lower_eng = st.checkbox("Lowercase English Words", value=False)

    # Input Text Area
    user_text = st.text_area(
        "Enter or paste text (English, Hindi, or Marathi):",
        value=st.session_state.get("text_analyzer_input", ""),
        height=140,
        placeholder="Type or paste Devanagari or Latin text here to analyze..."
    )
    # Keep session state synced
    st.session_state["text_analyzer_input"] = user_text

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        analyze_btn = st.button("🚀 Analyze Text", type="primary", use_container_width=True)

    if analyze_btn:
        if not user_text.strip():
            st.warning("Please enter or load some text before running the analysis.")
        else:
            with st.spinner("Processing text and querying Sarvam AI NLP models..."):
                # Step 1: Preprocessing
                processed_text = nlp_utils.clean_text(
                    user_text,
                    remove_extra_spaces=clean_spaces,
                    normalize_devanagari=norm_devanagari,
                    standardize_punctuation=std_punct,
                    lowercase_english=lower_eng
                )
                
                # Step 2: Compute real text statistics
                stats = nlp_utils.calculate_text_statistics(processed_text)
                
                # Step 3: Call Sarvam AI NLP API for Sentiment, Topic, and NER
                nlp_res = nlp_utils.analyze_nlp(processed_text, api_key=custom_key)
                
                # Cache results in session state for rendering and export
                st.session_state["last_analysis"] = {
                    "raw_text": user_text,
                    "processed_text": processed_text,
                    "stats": stats,
                    "nlp_res": nlp_res
                }

                # Log to session history
                sentiment_label = nlp_res.get("sentiment", {}).get("label", "N/A") if nlp_res.get("success") else "Error"
                topic_label = nlp_res.get("classification", {}).get("topic", "N/A") if nlp_res.get("success") else "Error"
                log_history_entry(
                    op_type="Text Analysis",
                    input_text=user_text,
                    output_summary=f"Sentiment: {sentiment_label} | Topic: {topic_label} | Words: {stats['word_count']}",
                    details={"stats": stats, "nlp": nlp_res}
                )

    # Render Results if available
    if "last_analysis" in st.session_state:
        analysis_data = st.session_state["last_analysis"]
        stats = analysis_data["stats"]
        nlp_res = analysis_data["nlp_res"]

        st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #E8E2D8;'>", unsafe_allow_html=True)
        st.markdown("### 📊 Text Statistics & Linguistics")

        # 4 Key Stat Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{stats['char_count_total']}</div>
                <div class="metric-label">Total Characters</div>
                <div style="font-size: 0.75rem; color: #8A847A;">{stats['char_count_no_spaces']} without spaces</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #FF786B;">{stats['word_count']}</div>
                <div class="metric-label">Total Words</div>
                <div style="font-size: 0.75rem; color: #8A847A;">{stats['unique_word_count']} unique tokens</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #F4C95D;">{stats['sentence_count']}</div>
                <div class="metric-label">Sentences</div>
                <div style="font-size: 0.75rem; color: #8A847A;">Avg {stats['avg_sentence_length']} words/sentence</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #252522;">{stats['lexical_diversity']}%</div>
                <div class="metric-label">Lexical Richness (TTR)</div>
                <div style="font-size: 0.75rem; color: #8A847A;">Type-Token Ratio</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # AI Insights Cards: Sentiment & Topic
        col_res_l, col_res_r = st.columns(2)
        
        with col_res_l:
            st.markdown("""<div class="lingo-card">""", unsafe_allow_html=True)
            st.markdown("#### 💬 Sentiment Analysis")
            if nlp_res.get("success"):
                sent = nlp_res["sentiment"]
                badge_class = f"badge-sentiment-{sent['label'].lower()}"
                st.markdown(f"""
                <div style="margin: 0.5rem 0 1rem 0;">
                    <span class="{badge_class}">● {sent['label']}</span>
                    <span style="font-size: 0.85rem; color: #787268; margin-left: 10px;">Confidence: <strong>{int(sent['confidence'] * 100)}%</strong></span>
                </div>
                <div style="background-color: #FAF7F2; padding: 12px 14px; border-radius: 8px; border: 1px solid #ECE6DC; font-size: 0.92rem; line-height: 1.5; color: #38342E;">
                    <strong>Linguistic Explanation:</strong><br>{sent['explanation']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(nlp_res.get("error", "Sentiment analysis could not be completed."))
            st.markdown("</div>", unsafe_allow_html=True)

        with col_res_r:
            st.markdown("""<div class="lingo-card">""", unsafe_allow_html=True)
            st.markdown("#### 🏷️ Topic Classification")
            if nlp_res.get("success"):
                cls_info = nlp_res["classification"]
                st.markdown(f"""
                <div style="margin: 0.5rem 0 1rem 0;">
                    <span class="badge-topic">📂 {cls_info['topic']}</span>
                    <span style="font-size: 0.85rem; color: #787268; margin-left: 10px;">Primary Subject</span>
                </div>
                <div style="background-color: #FAF7F2; padding: 12px 14px; border-radius: 8px; border: 1px solid #ECE6DC; font-size: 0.92rem; line-height: 1.5; color: #38342E;">
                    <strong>Classification Rationale:</strong><br>{cls_info['explanation']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(nlp_res.get("error", "Topic classification could not be completed."))
            st.markdown("</div>", unsafe_allow_html=True)

        # Named Entity Recognition (NER)
        st.markdown("### 🧩 Named Entity Recognition (NER)")
        if nlp_res.get("success"):
            entities = nlp_res.get("entities", [])
            if entities:
                st.markdown("""<div class="lingo-card">""", unsafe_allow_html=True)
                chips_html = ""
                for ent in entities:
                    ent_text = ent.get("text", "")
                    ent_type = ent.get("type", "OTHER")
                    chips_html += f'<span class="entity-chip entity-{ent_type}">{ent_text} <span style="font-size: 0.72rem; opacity: 0.8;">[{ent_type}]</span></span>'
                
                st.markdown(f"<div style='margin-bottom: 1rem;'>{chips_html}</div>", unsafe_allow_html=True)
                
                # Breakdown table and chart
                ent_df = pd.DataFrame(entities)
                if not ent_df.empty and "type" in ent_df.columns:
                    c_ent_tbl, c_ent_pie = st.columns([1, 1])
                    with c_ent_tbl:
                        type_counts = ent_df["type"].value_counts().reset_index()
                        type_counts.columns = ["Entity Type", "Frequency"]
                        st.dataframe(type_counts, use_container_width=True, hide_index=True)
                    with c_ent_pie:
                        fig_ent = px.pie(
                            type_counts,
                            values="Frequency",
                            names="Entity Type",
                            color_discrete_sequence=[THEME_PURPLE, THEME_GOLD, THEME_CORAL, THEME_MINT],
                            hole=0.45
                        )
                        fig_ent.update_layout(
                            paper_bgcolor=THEME_WHITE,
                            plot_bgcolor=THEME_WHITE,
                            margin=dict(t=20, b=20, l=20, r=20),
                            font=dict(color=THEME_CHARCOAL)
                        )
                        st.plotly_chart(fig_ent, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No specific named entities detected in the provided snippet.")
        else:
            st.warning("Entity extraction was not returned.")

        # Plotly Visualizations: Script Breakdown & Word Frequency
        st.markdown("### 📈 Visual NLP Analytics")
        v_col1, v_col2 = st.columns(2)
        
        with v_col1:
            st.markdown("""<div class="lingo-card">""", unsafe_allow_html=True)
            st.markdown("#### Script & Character Distribution")
            script_data = {
                "Character Category": ["Devanagari Chars", "Latin Chars", "Numerals", "Punctuation"],
                "Count": [stats["devanagari_chars"], stats["latin_chars"], stats["digit_chars"], stats["punct_chars"]]
            }
            sdf = pd.DataFrame(script_data)
            fig_script = px.bar(
                sdf,
                x="Character Category",
                y="Count",
                color="Character Category",
                color_discrete_sequence=[THEME_PURPLE, THEME_CORAL, THEME_GOLD, THEME_MINT]
            )
            fig_script.update_layout(
                paper_bgcolor=THEME_WHITE,
                plot_bgcolor=THEME_WHITE,
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
                font=dict(color=THEME_CHARCOAL),
                xaxis=dict(gridcolor="#EBE5DC"),
                yaxis=dict(gridcolor="#EBE5DC")
            )
            st.plotly_chart(fig_script, use_container_width=True)
            st.markdown(f"<div style='font-size: 0.85rem; color: #696359;'>Primary Script Identified: <strong>{stats['primary_script']}</strong></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with v_col2:
            st.markdown("""<div class="lingo-card">""", unsafe_allow_html=True)
            st.markdown("#### Top 10 Most Frequent Tokens")
            if stats["top_words"]:
                top_w_df = pd.DataFrame(list(stats["top_words"].items()), columns=["Token", "Frequency"])
                fig_words = px.bar(
                    top_w_df,
                    x="Frequency",
                    y="Token",
                    orientation="h",
                    color_discrete_sequence=[THEME_PURPLE]
                )
                fig_words.update_layout(
                    paper_bgcolor=THEME_WHITE,
                    plot_bgcolor=THEME_WHITE,
                    margin=dict(t=20, b=20, l=20, r=20),
                    font=dict(color=THEME_CHARCOAL),
                    yaxis=dict(autorange="reversed", gridcolor="#EBE5DC"),
                    xaxis=dict(gridcolor="#EBE5DC")
                )
                st.plotly_chart(fig_words, use_container_width=True)
            else:
                st.info("No frequent tokens found.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Export Analysis Results
        st.markdown("### 💾 Export Analysis Report")
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        
        # Prepare text report
        txt_report = f"""LINGOAI STUDIO - NLP ANALYSIS REPORT
Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
------------------------------------------------------------
TEXT INPUT:
{user_text}

TEXT STATISTICS:
- Total Characters: {stats['char_count_total']} (No spaces: {stats['char_count_no_spaces']})
- Total Words: {stats['word_count']} (Unique: {stats['unique_word_count']})
- Sentences: {stats['sentence_count']}
- Lexical Diversity (TTR): {stats['lexical_diversity']}%
- Primary Script: {stats['primary_script']}

SENTIMENT ANALYSIS:
- Polarity: {nlp_res.get('sentiment', {}).get('label', 'N/A')}
- Confidence: {nlp_res.get('sentiment', {}).get('confidence', 0.0)}
- Explanation: {nlp_res.get('sentiment', {}).get('explanation', 'N/A')}

TOPIC CLASSIFICATION:
- Topic Category: {nlp_res.get('classification', {}).get('topic', 'N/A')}
- Rationale: {nlp_res.get('classification', {}).get('explanation', 'N/A')}

NAMED ENTITIES EXTRACTED:
{json.dumps(nlp_res.get('entities', []), indent=2, ensure_ascii=False)}
"""
        with exp_col1:
            st.download_button(
                "📥 Download Report (.txt)",
                data=txt_report,
                file_name="lingoai_analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )
        with exp_col2:
            st.download_button(
                "📥 Download Data (.json)",
                data=json.dumps(analysis_data, indent=2, ensure_ascii=False, default=str),
                file_name="lingoai_analysis_data.json",
                mime="application/json",
                use_container_width=True
            )
        with exp_col3:
            # Stats CSV
            stat_df = pd.DataFrame([stats])
            st.download_button(
                "📥 Download Stats (.csv)",
                data=stat_df.to_csv(index=False),
                file_name="lingoai_stats.csv",
                mime="text/csv",
                use_container_width=True
            )


# ==============================================================================
# 7. PAGE: TRANSLATION
# ==============================================================================
elif "Translation" in selected_page:
    st.markdown("## 🔄 Multilingual Translation Studio")
    st.markdown("High-fidelity neural machine translation between English, Hindi, and Marathi powered by Sarvam AI `mayura:v1`.")

    # Sample Buttons
    st.markdown("<div style='font-size: 0.88rem; font-weight: 600; color: #696359; margin-bottom: 6px;'>LOAD EXHIBITION SAMPLES:</div>", unsafe_allow_html=True)
    c_tr_s1, c_tr_s2, c_tr_s3, c_tr_s4 = st.columns([1, 1, 1, 1])
    with c_tr_s1:
        if st.button("📄 English Text", use_container_width=True):
            st.session_state["translation_input"] = benchmark_data.SAMPLE_TEXTS["English"]["text"]
            st.rerun()
    with c_tr_s2:
        if st.button("📄 Hindi Text (हिंदी)", use_container_width=True):
            st.session_state["translation_input"] = benchmark_data.SAMPLE_TEXTS["Hindi"]["text"]
            st.rerun()
    with c_tr_s3:
        if st.button("📄 Marathi Text (मराठी)", use_container_width=True):
            st.session_state["translation_input"] = benchmark_data.SAMPLE_TEXTS["Marathi"]["text"]
            st.rerun()
    with c_tr_s4:
        if st.button("🗑️ Clear Box", use_container_width=True):
            st.session_state["translation_input"] = ""
            st.rerun()

    # Controls Row
    c_lang1, c_swap, c_lang2, c_model = st.columns([2.5, 0.8, 2.5, 2.2])
    
    with c_lang1:
        source_lang_choice = st.selectbox(
            "Source Language",
            options=["auto", "en-IN", "hi-IN", "mr-IN"],
            format_func=lambda x: "🌐 Auto Detect" if x == "auto" else nlp_utils.SUPPORTED_LANGUAGES.get(x, x),
            index=1
        )
    with c_swap:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("⇄", help="Swap Source and Target Languages", use_container_width=True):
            if source_lang_choice != "auto":
                st.session_state["swap_temp"] = True
                st.rerun()
    with c_lang2:
        target_lang_choice = st.selectbox(
            "Target Language",
            options=["mr-IN", "hi-IN", "en-IN"],
            format_func=lambda x: nlp_utils.SUPPORTED_LANGUAGES.get(x, x),
            index=0
        )
    with c_model:
        model_choice = st.selectbox(
            "Translation Engine",
            options=["mayura:v1", "sarvam-translate:v1"],
            help="`mayura:v1` is Sarvam AI's primary Indic neural translation model."
        )

    preprocess_before_tr = st.checkbox("Preprocess & Normalize text before translating", value=True)

    # Two-Column Layout for Translation
    col_src, col_tgt = st.columns(2)
    
    with col_src:
        st.markdown("#### Source Text")
        tr_input_text = st.text_area(
            "Source Text Input",
            value=st.session_state.get("translation_input", ""),
            height=200,
            placeholder="Enter text to translate...",
            label_visibility="collapsed"
        )
        st.session_state["translation_input"] = tr_input_text
        src_words = len(nlp_utils.tokenize_words(tr_input_text))
        st.markdown(f"<div style='font-size: 0.8rem; color: #706B62; margin-top: 4px;'>Length: {len(tr_input_text)} characters | {src_words} words</div>", unsafe_allow_html=True)

    with col_tgt:
        st.markdown("#### Translated Output")
        translated_display = st.session_state.get("last_translated_text", "")
        tr_output_area = st.text_area(
            "Translated Output Text",
            value=translated_display,
            height=200,
            placeholder="Translation will appear here upon clicking Translate...",
            label_visibility="collapsed"
        )
        tgt_words = len(nlp_utils.tokenize_words(translated_display))
        st.markdown(f"<div style='font-size: 0.8rem; color: #706B62; margin-top: 4px;'>Length: {len(translated_display)} characters | {tgt_words} words</div>", unsafe_allow_html=True)

    # Action Buttons
    b_col1, b_col2, b_col3 = st.columns([1.5, 1.5, 3])
    with b_col1:
        translate_btn = st.button("🚀 Translate Now", type="primary", use_container_width=True)
    with b_col2:
        if translated_display:
            st.download_button(
                "📥 Download Translation (.txt)",
                data=translated_display,
                file_name=f"translation_{target_lang_choice}.txt",
                mime="text/plain",
                use_container_width=True
            )

    if translate_btn:
        if not tr_input_text.strip():
            st.warning("Please provide input text to translate.")
        else:
            with st.spinner(f"Translating to {nlp_utils.SUPPORTED_LANGUAGES.get(target_lang_choice, target_lang_choice)}..."):
                input_to_send = tr_input_text
                if preprocess_before_tr:
                    input_to_send = nlp_utils.clean_text(input_to_send)

                # Real translation API call
                res = nlp_utils.translate_text(
                    text=input_to_send,
                    source_lang=source_lang_choice,
                    target_lang=target_lang_choice,
                    model=model_choice,
                    api_key=custom_key
                )

                if res.get("success"):
                    translated_text = res.get("translated_text", "")
                    st.session_state["last_translated_text"] = translated_text
                    
                    # Log into session history
                    log_history_entry(
                        op_type="Translation",
                        input_text=tr_input_text,
                        output_summary=f"{source_lang_choice} ➔ {target_lang_choice}: {translated_text[:80]}...",
                        details=res
                    )
                    st.rerun()
                else:
                    st.error(res.get("error", "Translation request failed."))


# ==============================================================================
# 8. PAGE: SUMMARIZATION
# ==============================================================================
elif "Summarization" in selected_page:
    st.markdown("## 📝 Multilingual Text Summarization")
    st.markdown("Generate concise executive summaries or key actionable takeaways while retaining native linguistic context.")

    # Sample Buttons
    st.markdown("<div style='font-size: 0.88rem; font-weight: 600; color: #696359; margin-bottom: 6px;'>LOAD EXHIBITION SAMPLES:</div>", unsafe_allow_html=True)
    c_sm_s1, c_sm_s2, c_sm_s3, c_sm_s4 = st.columns([1, 1, 1, 1])
    with c_sm_s1:
        if st.button("📄 English Passage", use_container_width=True):
            st.session_state["summarization_input"] = benchmark_data.SAMPLE_TEXTS["English"]["text"]
            st.rerun()
    with c_sm_s2:
        if st.button("📄 Hindi Passage (हिंदी)", use_container_width=True):
            st.session_state["summarization_input"] = benchmark_data.SAMPLE_TEXTS["Hindi"]["text"]
            st.rerun()
    with c_sm_s3:
        if st.button("📄 Marathi Passage (मराठी)", use_container_width=True):
            st.session_state["summarization_input"] = benchmark_data.SAMPLE_TEXTS["Marathi"]["text"]
            st.rerun()
    with c_sm_s4:
        if st.button("🗑️ Clear Text Box", use_container_width=True):
            st.session_state["summarization_input"] = ""
            st.rerun()

    # Controls
    c_style, c_len = st.columns([1, 1])
    with c_style:
        summary_style = st.radio(
            "Summary Presentation Format",
            ["Executive Paragraph", "Key Takeaways (Bullet Points)"],
            horizontal=True
        )
    with c_len:
        summary_length = st.select_slider(
            "Desired Summary Length",
            options=["short", "medium", "detailed"],
            value="medium",
            format_func=lambda x: {"short": "⚡ Short (1-2 lines)", "medium": "⚖️ Medium (Balanced)", "detailed": "📖 Detailed"}[x]
        )

    # Input Area
    sum_input = st.text_area(
        "Input text to summarize:",
        value=st.session_state.get("summarization_input", ""),
        height=160,
        placeholder="Paste an article, document, or passage here..."
    )
    st.session_state["summarization_input"] = sum_input

    orig_word_count = len(nlp_utils.tokenize_words(sum_input))
    st.markdown(f"<div style='font-size: 0.8rem; color: #706B62; margin-bottom: 12px;'>Original Text: {len(sum_input)} characters | {orig_word_count} words</div>", unsafe_allow_html=True)

    if st.button("🚀 Generate Summary", type="primary"):
        if not sum_input.strip():
            st.warning("Please supply text to summarize.")
        else:
            with st.spinner("Synthesizing multilingual abstractive summary..."):
                style_key = "bullets" if "Bullet" in summary_style else "paragraph"
                res = nlp_utils.generate_summary(
                    text=sum_input,
                    style=style_key,
                    length_level=summary_length,
                    api_key=custom_key
                )

                if res.get("success"):
                    st.session_state["last_summary"] = res
                    log_history_entry(
                        op_type="Summarization",
                        input_text=sum_input,
                        output_summary=f"Reduced by {res.get('reduction_percentage')}%: {res.get('summary')[:80]}...",
                        details=res
                    )
                else:
                    st.error(res.get("error", "Summarization failed."))

    # Render Summary Result
    if "last_summary" in st.session_state:
        sum_res = st.session_state["last_summary"]
        st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #E8E2D8;'>", unsafe_allow_html=True)
        st.markdown("### 📋 Generated Summary")

        # Compression Statistics
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{sum_res['original_word_count']}</div>
                <div class="metric-label">Original Word Count</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #FF786B;">{sum_res['summary_word_count']}</div>
                <div class="metric-label">Summary Word Count</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #BCE8D1; -webkit-text-stroke: 1px #164E33;">{sum_res['reduction_percentage']}%</div>
                <div class="metric-label">Text Compression Ratio</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Summary Presentation Box
        st.markdown("""<div class="lingo-card" style="border-left: 5px solid #8B5CF6;">""", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.05rem; line-height: 1.6; color: #252522;'>{sum_res['summary']}</p>", unsafe_allow_html=True)
        
        bullets = sum_res.get("bullet_points", [])
        if bullets and len(bullets) > 0:
            st.markdown("<h5 style='margin-top: 1rem; font-size: 0.95rem;'>Key Bullet Takeaways:</h5>", unsafe_allow_html=True)
            for b in bullets:
                st.markdown(f"<li style='color: #403C35; margin-bottom: 4px;'>{b}</li>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Download button
        st.download_button(
            "📥 Download Summary (.txt)",
            data=sum_res["summary"],
            file_name="lingoai_summary.txt",
            mime="text/plain"
        )


# ==============================================================================
# 9. PAGE: EVALUATION
# ==============================================================================
elif "Evaluation" in selected_page:
    st.markdown("## 📊 Mathematical Evaluation Framework")
    st.markdown("Evaluate Sarvam AI NLP predictions against ground-truth labeled datasets. Computes authentic **Accuracy**, **Precision**, **Recall**, **Macro-F1**, and the **Confusion Matrix**.")

    tab_eval_bench, tab_eval_custom = st.tabs(["🏛️ College Exhibition Benchmark", "📁 Upload Custom CSV"])

    with tab_eval_bench:
        st.markdown("""
        <div class="lingo-card">
            <h4 style="margin: 0 0 6px 0;">Pre-loaded Multilingual Benchmark Dataset</h4>
            <p style="color: #5E584E; font-size: 0.9rem; margin-bottom: 10px;">
                Curated set of 15 diverse real-world samples covering English, Hindi, and Marathi with verified ground truth for Sentiment Analysis and Topic Classification.
            </p>
        </div>
        """, unsafe_allow_html=True)

        c_task, c_samples = st.columns([1, 1])
        with c_task:
            bench_task = st.selectbox(
                "Evaluation Task",
                options=["sentiment", "topic"],
                format_func=lambda x: "Sentiment Analysis (Positive / Negative / Neutral)" if x == "sentiment" else "Topic Classification (Tech, Politics, Sports, etc.)",
                key="bench_task_select"
            )
        with c_samples:
            bench_count = st.slider("Number of Test Samples to Evaluate", min_value=3, max_value=15, value=6, step=1)

        bench_df = pd.DataFrame(benchmark_data.BENCHMARK_EVAL_DATA)
        label_col_name = "true_sentiment" if bench_task == "sentiment" else "true_topic"

        st.markdown(f"<div style='font-size: 0.85rem; font-weight: 600; color: #6E685D; margin-bottom: 4px;'>Benchmark Preview ({bench_count} selected):</div>", unsafe_allow_html=True)
        st.dataframe(bench_df[["id", "language", "text", label_col_name]].head(bench_count), use_container_width=True, hide_index=True)

        if st.button("🚀 Run Live Benchmark Evaluation", type="primary", key="btn_run_bench"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(curr, tot):
                progress_bar.progress(curr / tot)
                status_text.text(f"Evaluating sample {curr} of {tot} using Sarvam AI...")

            with st.spinner("Executing live model evaluation..."):
                eval_results = nlp_utils.run_batch_evaluation(
                    df=bench_df,
                    task=bench_task,
                    text_col="text",
                    label_col=label_col_name,
                    max_items=bench_count,
                    api_key=custom_key,
                    progress_callback=update_progress
                )
                progress_bar.empty()
                status_text.empty()
                st.session_state["last_eval_results"] = eval_results

                log_history_entry(
                    op_type="Benchmark Evaluation",
                    input_text=f"Batch Evaluation: {bench_task} ({bench_count} samples)",
                    output_summary=f"Accuracy: {eval_results['metrics']['accuracy'] * 100:.1f}% | Macro-F1: {eval_results['metrics']['macro_f1']:.3f}",
                    details={"metrics": eval_results["metrics"]}
                )

    with tab_eval_custom:
        st.markdown("""
        <div class="lingo-card">
            <h4 style="margin: 0 0 6px 0;">Upload Labeled Dataset for Rigorous Testing</h4>
            <p style="color: #5E584E; font-size: 0.9rem; margin-bottom: 0;">
                Supply a CSV containing a text column and a ground truth label column.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            custom_df = pd.read_csv(uploaded_file)
            st.dataframe(custom_df.head(5), use_container_width=True)

            c_c1, c_c2, c_c3, c_c4 = st.columns(4)
            with c_c1:
                custom_task = st.selectbox("Task", ["sentiment", "topic"], key="cust_task")
            with c_c2:
                text_col = st.selectbox("Text Column", custom_df.columns)
            with c_c3:
                label_col = st.selectbox("Ground Truth Column", custom_df.columns)
            with c_c4:
                custom_limit = st.slider("Max Samples", 3, min(len(custom_df), 30), 5)

            if st.button("🚀 Evaluate Uploaded Dataset", type="primary"):
                with st.spinner("Processing custom evaluation dataset..."):
                    eval_results = nlp_utils.run_batch_evaluation(
                        df=custom_df,
                        task=custom_task,
                        text_col=text_col,
                        label_col=label_col,
                        max_items=custom_limit,
                        api_key=custom_key
                    )
                    st.session_state["last_eval_results"] = eval_results

    # Render Evaluation Results
    if "last_eval_results" in st.session_state:
        eval_res = st.session_state["last_eval_results"]
        metrics = eval_res["metrics"]
        results_df = eval_res["results_df"]
        task_name = eval_res["task"]

        st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #E8E2D8;'>", unsafe_allow_html=True)
        st.markdown(f"### 📈 Live Evaluation Results ({task_name.capitalize()})")

        # Key Metrics Row
        em1, em2, em3, em4 = st.columns(4)
        with em1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{metrics['accuracy'] * 100:.1f}%</div>
                <div class="metric-label">Overall Accuracy</div>
                <div style="font-size: 0.75rem; color: #8A847A;">{metrics['correct_predictions']}/{metrics['total_samples']} Correct</div>
            </div>
            """, unsafe_allow_html=True)
        with em2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #FF786B;">{metrics['macro_f1']:.3f}</div>
                <div class="metric-label">Macro F1-Score</div>
                <div style="font-size: 0.75rem; color: #8A847A;">Harmonic Mean of P & R</div>
            </div>
            """, unsafe_allow_html=True)
        with em3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #F4C95D;">{metrics['macro_precision']:.3f}</div>
                <div class="metric-label">Macro Precision</div>
                <div style="font-size: 0.75rem; color: #8A847A;">Exactness Ratio</div>
            </div>
            """, unsafe_allow_html=True)
        with em4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #BCE8D1; -webkit-text-stroke: 1px #164E33;">{metrics['macro_recall']:.3f}</div>
                <div class="metric-label">Macro Recall</div>
                <div style="font-size: 0.75rem; color: #8A847A;">Sensitivity Ratio</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Plotly Confusion Matrix & Per-Class Table
        c_hm, c_cls_tbl = st.columns([1.1, 1.2])
        with c_hm:
            st.markdown("""<div class="lingo-card">""", unsafe_allow_html=True)
            st.markdown("#### Real Confusion Matrix Heatmap")
            
            # Confusion Matrix heatmap using palette: Warm Ivory -> Gold -> Coral -> Purple
            fig_cm = px.imshow(
                metrics["confusion_matrix_grid"],
                x=metrics["classes"],
                y=metrics["classes"],
                labels=dict(x="Predicted Class", y="Ground Truth Class", color="Count"),
                color_continuous_scale=[[0, THEME_WHITE], [0.5, THEME_GOLD], [1.0, THEME_PURPLE]],
                text_auto=True
            )
            fig_cm.update_layout(
                paper_bgcolor=THEME_WHITE,
                plot_bgcolor=THEME_WHITE,
                margin=dict(t=30, b=30, l=30, r=30),
                font=dict(color=THEME_CHARCOAL)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c_cls_tbl:
            st.markdown("""<div class="lingo-card">""", unsafe_allow_html=True)
            st.markdown("#### Per-Class Performance Breakdown")
            per_class_list = []
            for c_name, c_data in metrics["per_class"].items():
                if c_data["support"] > 0 or c_data["fp"] > 0:
                    per_class_list.append({
                        "Class": c_name,
                        "Support": c_data["support"],
                        "Precision": c_data["precision"],
                        "Recall": c_data["recall"],
                        "F1-Score": c_data["f1_score"]
                    })
            if per_class_list:
                st.dataframe(pd.DataFrame(per_class_list), use_container_width=True, hide_index=True)
            else:
                st.info("No class support data found.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Detailed Inspection Table
        st.markdown("### 🔍 Sample-by-Sample Prediction Audit")
        
        filter_opt = st.radio("Filter Samples", ["Show All", "Misclassified Only"], horizontal=True)
        display_eval_df = results_df.copy()
        if filter_opt == "Misclassified Only":
            display_eval_df = display_eval_df[~display_eval_df["is_correct"]]

        # Display friendly columns
        audit_cols = [c for c in ["id", "language", "text", "true_sentiment", "true_topic", "predicted", "is_correct", "model_explanation"] if c in display_eval_df.columns]
        st.dataframe(display_eval_df[audit_cols], use_container_width=True, hide_index=True)

        # Download Report
        st.download_button(
            "📥 Download Evaluation Results (.csv)",
            data=results_df.to_csv(index=False),
            file_name=f"evaluation_{task_name}_results.csv",
            mime="text/csv"
        )


# ==============================================================================
# 10. PAGE: HISTORY
# ==============================================================================
elif "History" in selected_page:
    st.markdown("## 📜 Session History & Audit Log")
    st.markdown("Inspect real-time operations performed during your current session.")

    history_list = st.session_state.get("history", [])

    if not history_list:
        st.info("No operations have been recorded in this session yet. Try analyzing, translating, or evaluating some text!")
    else:
        h_col1, h_col2 = st.columns([1, 4])
        with h_col1:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()
        with h_col2:
            hist_df = pd.DataFrame(history_list)
            # Remove nested details from raw CSV export to make it clean
            export_df = hist_df[["timestamp", "operation", "input_preview", "output_summary", "input_length"]]
            st.download_button(
                "📥 Export History as CSV",
                data=export_df.to_csv(index=False),
                file_name="lingoai_session_history.csv",
                mime="text/csv"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        for idx, entry in enumerate(history_list):
            with st.expander(f"#{len(history_list) - idx} [{entry['operation']}] at {entry['timestamp']} — {entry['output_summary'][:60]}..."):
                st.markdown(f"**Timestamp:** `{entry['timestamp']}`")
                st.markdown(f"**Operation:** `{entry['operation']}`")
                st.markdown(f"**Input Preview:**")
                st.code(entry["input_preview"])
                st.markdown(f"**Output Summary:** {entry['output_summary']}")
                st.markdown(f"**Details Payload:**")
                st.json(entry["details"])


# ==============================================================================
# 11. FOOTER
# ==============================================================================
st.markdown("""
<div class="app-footer">
    <strong>LingoAI Studio</strong> — Multilingual Text Processing and Analysis System<br>
    Final Year NLP Engineering Project Exhibition • Powered by Sarvam AI & Streamlit
</div>
""", unsafe_allow_html=True)
