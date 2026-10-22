"""
nlp_utils.py
Core Natural Language Processing utilities, regex preprocessing,
Sarvam AI SDK integration (Translation, Chatbot, STT, TTS, Summarization, Analytics),
and evaluation calculations for LingoAI Studio.
"""

import os
import re
import json
import base64
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional, Union

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supported languages in LingoAI Studio
SUPPORTED_LANGUAGES = {
    "en-IN": "English (India)",
    "hi-IN": "Hindi (हिंदी)",
    "mr-IN": "Marathi (मराठी)",
}

TOPIC_CATEGORIES = [
    "Technology",
    "Politics",
    "Business",
    "Sports",
    "Entertainment",
    "Healthcare",
    "Education",
    "Other"
]

SENTIMENT_CLASSES = ["Positive", "Negative", "Neutral"]

# Valid speakers for bulbul:v3
AVAILABLE_SPEAKERS = [
    "aditya", "ritu", "priya", "neha", "rahul", "pooja", 
    "rohan", "simran", "kavya", "amit", "dev", "ishita", 
    "shreya", "ratan", "varun", "manan", "sumit", "roopa", 
    "kabir", "aayan", "shubh", "advait", "anand", "tanya", 
    "tarun", "sunny", "mani", "gokul", "vijay", "shruti", 
    "suhani", "mohit", "kavitha", "rehan", "soham", "rupali"
]


# ==============================================================================
# 1. TEXT PREPROCESSING & STATISTICS
# ==============================================================================

def clean_text(
    text: str,
    remove_extra_spaces: bool = True,
    normalize_devanagari: bool = True,
    standardize_punctuation: bool = True,
    lowercase_english: bool = False
) -> str:
    """
    Cleans and normalizes multilingual text with specific handling for Devanagari and English.
    """
    if not text:
        return ""

    cleaned = text

    # Unicode normalization (NFC)
    if normalize_devanagari:
        cleaned = unicodedata.normalize('NFC', cleaned)
        # Normalize common Devanagari zero-width chars if misplaced
        cleaned = re.sub(r'[\u200B\uFEFF]', '', cleaned)
        # Standardize Devanagari danda and double danda
        cleaned = cleaned.replace("||", "॥").replace("|", "।")

    # Lowercase English if requested (preserves Devanagari untouched)
    if lowercase_english:
        cleaned = cleaned.lower()

    # Standardize quotation marks and dashes
    if standardize_punctuation:
        cleaned = re.sub(r'["""]', '"', cleaned)
        cleaned = re.sub(r"[''']", "'", cleaned)
        cleaned = re.sub(r'[—–]', '-', cleaned)

    # Whitespace normalization
    if remove_extra_spaces:
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = cleaned.strip()

    return cleaned


def tokenize_words(text: str) -> List[str]:
    """
    Tokenizes words supporting both Latin and Devanagari unicode characters.
    """
    if not text:
        return []
    tokens = re.findall(r'[\u0900-\u097Fa-zA-Z0-9_]+', text)
    return tokens


def tokenize_sentences(text: str) -> List[str]:
    """
    Tokenizes text into sentences handling standard stops (. ! ?) as well as
    Devanagari danda (।) and double danda (॥).
    """
    if not text or not text.strip():
        return []
    
    raw_sentences = re.split(r'([.!?।॥]+(?:\s+|\n+|$))', text.strip())
    sentences = []
    current = ""
    for piece in raw_sentences:
        if not piece:
            continue
        current += piece
        if re.search(r'[.!?।॥]+', piece):
            sent = current.strip()
            if sent:
                sentences.append(sent)
            current = ""
    if current.strip():
        sentences.append(current.strip())
    
    return [s for s in sentences if len(s.strip()) > 0]


def calculate_text_statistics(text: str) -> Dict[str, Any]:
    """
    Calculates comprehensive statistical and script metrics for supplied text.
    """
    if not text or not text.strip():
        return {
            "char_count_total": 0,
            "char_count_no_spaces": 0,
            "word_count": 0,
            "unique_word_count": 0,
            "sentence_count": 0,
            "lexical_diversity": 0.0,
            "avg_word_length": 0.0,
            "avg_sentence_length": 0.0,
            "devanagari_chars": 0,
            "latin_chars": 0,
            "digit_chars": 0,
            "punct_chars": 0,
            "primary_script": "None",
            "top_words": {},
            "word_length_distribution": {}
        }

    words = tokenize_words(text)
    sentences = tokenize_sentences(text)
    
    total_chars = len(text)
    chars_no_space = len(re.sub(r'\s+', '', text))
    word_count = len(words)
    
    normalized_words = [w.lower() if re.match(r'^[a-zA-Z]', w) else w for w in words]
    unique_words = len(set(normalized_words))
    sentence_count = max(len(sentences), 1 if word_count > 0 else 0)
    
    lexical_diversity = round((unique_words / word_count) * 100, 2) if word_count > 0 else 0.0
    avg_word_length = round(sum(len(w) for w in words) / word_count, 2) if word_count > 0 else 0.0
    avg_sentence_len = round(word_count / sentence_count, 2) if sentence_count > 0 else 0.0
    
    devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    digit_chars = len(re.findall(r'[0-9\u0966-\u096F]', text))
    punct_chars = len(re.findall(r'[^\w\s\u0900-\u097F]', text))
    
    if devanagari_chars > latin_chars and devanagari_chars > 0:
        primary_script = "Devanagari (Hindi/Marathi)"
    elif latin_chars > devanagari_chars and latin_chars > 0:
        primary_script = "Latin (English)"
    elif devanagari_chars > 0 and latin_chars > 0:
        primary_script = "Mixed Script"
    else:
        primary_script = "Numeric / Symbols"

    filtered_words = [w for w in normalized_words if len(w) > 1]
    top_words = dict(Counter(filtered_words).most_common(10))
    
    lengths = [len(w) for w in words]
    length_dist = dict(Counter(lengths))

    return {
        "char_count_total": total_chars,
        "char_count_no_spaces": chars_no_space,
        "word_count": word_count,
        "unique_word_count": unique_words,
        "sentence_count": sentence_count,
        "lexical_diversity": lexical_diversity,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_len,
        "devanagari_chars": devanagari_chars,
        "latin_chars": latin_chars,
        "digit_chars": digit_chars,
        "punct_chars": punct_chars,
        "primary_script": primary_script,
        "top_words": top_words,
        "word_length_distribution": length_dist
    }


# ==============================================================================
# 2. SARVAM AI CLIENT & API OPERATIONS
# ==============================================================================

def get_sarvam_client(api_key: Optional[str] = None):
    """
    Initializes and returns the SarvamAI client using the provided or environment key.
    """
    key = api_key or os.getenv("SARVAM_API_KEY")
    if not key or not key.strip():
        return None, "Sarvam API Key is missing. Please set SARVAM_API_KEY in .env or enter it in the app sidebar."
    
    try:
        from sarvamai import SarvamAI
        client = SarvamAI(api_subscription_key=key.strip())
        return client, None
    except Exception as e:
        return None, f"Failed to initialize Sarvam AI SDK: {str(e)}"


def identify_language(text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Detects language using Sarvam AI identify_language endpoint with heuristic fallback.
    """
    if not text or not text.strip():
        return {"language_code": "unknown", "script_code": "unknown", "language_name": "Unknown"}

    client, err = get_sarvam_client(api_key)
    if client:
        try:
            res = client.text.identify_language(input=text[:500])
            code = getattr(res, "language_code", "unknown")
            script = getattr(res, "script_code", "unknown")
            name = SUPPORTED_LANGUAGES.get(code, code)
            return {
                "language_code": code,
                "script_code": script,
                "language_name": name,
                "source": "Sarvam AI API"
            }
        except Exception:
            pass

    # Heuristic fallback
    deva_count = len(re.findall(r'[\u0900-\u097F]', text))
    latn_count = len(re.findall(r'[a-zA-Z]', text))
    if deva_count > latn_count:
        if "ळ" in text or re.search(r'\b(आहे|आणि|नाही|झाले|केले|होते|म्हणून)\b', text):
            return {"language_code": "mr-IN", "script_code": "Deva", "language_name": "Marathi (मराठी)", "source": "Heuristic"}
        return {"language_code": "hi-IN", "script_code": "Deva", "language_name": "Hindi (हिंदी)", "source": "Heuristic"}
    else:
        return {"language_code": "en-IN", "script_code": "Latn", "language_name": "English (India)", "source": "Heuristic"}


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    model: str = "mayura:v1",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Translates text between English, Hindi, and Marathi using Sarvam AI translation API.
    """
    if not text or not text.strip():
        return {"success": False, "translated_text": "", "error": "Input text cannot be empty."}

    if source_lang == target_lang and source_lang != "auto":
        return {
            "success": True,
            "translated_text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "error": None
        }

    client, err = get_sarvam_client(api_key)
    if not client:
        return {"success": False, "translated_text": "", "error": err}

    try:
        res = client.text.translate(
            input=text,
            source_language_code=source_lang,
            target_language_code=target_lang,
            model=model
        )
        translated = getattr(res, "translated_text", "")
        return {
            "success": True,
            "translated_text": translated,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "model_used": model,
            "error": None
        }
    except Exception as e:
        error_msg = str(e)
        if "status_code: 401" in error_msg or "Unauthorized" in error_msg:
            return {"success": False, "translated_text": "", "error": "Authentication failed: Invalid SARVAM_API_KEY. Please verify your key."}
        elif "status_code: 429" in error_msg:
            return {"success": False, "translated_text": "", "error": "Rate limit exceeded. Please wait a moment and try again."}
        return {"success": False, "translated_text": "", "error": f"Translation error: {error_msg}"}


def generate_summary(
    text: str,
    style: str = "paragraph",
    length_level: str = "medium",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a concise summary using Sarvam AI's multilingual LLM (sarvam-105b-conversations).
    """
    if not text or not text.strip():
        return {"success": False, "summary": "", "key_points": [], "error": "Input text cannot be empty."}

    client, err = get_sarvam_client(api_key)
    if not client:
        return {"success": False, "summary": "", "key_points": [], "error": err}

    length_prompt = {
        "short": "Keep it very brief (1-2 sentences or 2 concise bullet points).",
        "medium": "Provide a well-rounded summary (3-4 sentences or 3 bullet points).",
        "detailed": "Provide a comprehensive summary capturing all key insights."
    }.get(length_level, "Provide a concise summary.")

    style_prompt = (
        "Focus on key actionable bullet points." if style == "bullets" 
        else "Write a cohesive, executive paragraph."
    )

    prompt = f"""You are an expert multilingual summarization system.
Read the following text and summarize it in the SAME language as the original input (English, Hindi, or Marathi).
Guidelines:
- {length_prompt}
- {style_prompt}
- Strictly preserve all factual names, numbers, and core message.

Input Text:
"{text}"

Output a valid JSON object with these exact keys:
{{
  "summary": "Full cohesive summary text",
  "bullet_points": ["Key takeaway 1", "Key takeaway 2", "Key takeaway 3"],
  "language": "English" or "Hindi" or "Marathi"
}}
Return ONLY the raw JSON object, without markdown quotes."""

    try:
        res = client.chat.completions(
            model="sarvam-105b-conversations",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = res.choices[0].message.content or ""
        
        parsed_data = {}
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                parsed_data = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        summary_text = parsed_data.get("summary", content.strip())
        bullets = parsed_data.get("bullet_points", [])
        lang = parsed_data.get("language", "Auto")

        orig_words = len(tokenize_words(text))
        summary_words = len(tokenize_words(summary_text))
        reduction = round((1 - (summary_words / max(orig_words, 1))) * 100, 1) if orig_words > 0 else 0.0

        return {
            "success": True,
            "summary": summary_text,
            "bullet_points": bullets,
            "language": lang,
            "original_word_count": orig_words,
            "summary_word_count": summary_words,
            "reduction_percentage": max(reduction, 0.0),
            "error": None
        }
    except Exception as e:
        return {"success": False, "summary": "", "key_points": [], "error": f"Summarization failed: {str(e)}"}


def analyze_nlp(text: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs unified deep NLP analysis:
    1. Sentiment Analysis (Positive, Negative, Neutral) + Confidence + Explanation
    2. Topic Classification (Technology, Politics, Business, etc.) + Explanation
    3. Named Entity Recognition (PERSON, LOCATION, ORGANIZATION, DATE, OTHER)
    """
    if not text or not text.strip():
        return {"success": False, "error": "Input text cannot be empty."}

    client, err = get_sarvam_client(api_key)
    if not client:
        return {"success": False, "error": err}

    prompt = f"""You are a senior multilingual NLP analysis engine supporting English, Hindi, and Marathi.
Analyze the following text with high precision.

Input Text:
"{text}"

Instructions:
1. Sentiment: Must be strictly one of ["Positive", "Negative", "Neutral"].
2. Topic: Must be strictly one of ["Technology", "Politics", "Business", "Sports", "Entertainment", "Healthcare", "Education", "Other"].
3. Entities: Extract named entities with their type (PERSON, LOCATION, ORGANIZATION, DATE, OTHER).
4. Provide concise linguistic explanations for sentiment and topic.

Return ONLY a valid JSON object matching this schema:
{{
  "sentiment": "Positive",
  "sentiment_confidence": 0.95,
  "sentiment_explanation": "Explanation of why this sentiment applies.",
  "topic": "Technology",
  "topic_explanation": "Explanation of why this topic applies.",
  "entities": [
    {{"text": "Extracted Entity", "type": "PERSON"}}
  ]
}}
Do NOT output markdown fences or extra conversational text."""

    try:
        res = client.chat.completions(
            model="sarvam-105b-conversations",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        content = res.choices[0].message.content or ""
        
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            data = json.loads(content)

        raw_sentiment = str(data.get("sentiment", "Neutral")).strip().capitalize()
        if any(w in raw_sentiment for w in ["सकारात्मक", "आनंद", "चांगले"]):
            sentiment = "Positive"
        elif any(w in raw_sentiment for w in ["नकारात्मक", "वाईट", "दुःख", "खराब"]):
            sentiment = "Negative"
        elif raw_sentiment in ["Positive", "Negative", "Neutral"]:
            sentiment = raw_sentiment
        else:
            sentiment = "Neutral"

        raw_topic = str(data.get("topic", "Other")).strip()
        topic = raw_topic if raw_topic in TOPIC_CATEGORIES else "Other"

        entities = data.get("entities", [])
        if not isinstance(entities, list):
            entities = []

        return {
            "success": True,
            "sentiment": {
                "label": sentiment,
                "confidence": float(data.get("sentiment_confidence", 0.90)),
                "explanation": str(data.get("sentiment_explanation", "Standard linguistic analysis."))
            },
            "classification": {
                "topic": topic,
                "explanation": str(data.get("topic_explanation", "Classified based on contextual terminology."))
            },
            "entities": entities,
            "error": None
        }
    except Exception as e:
        return {"success": False, "error": f"NLP analysis failed: {str(e)}"}


# ==============================================================================
# 3. VOICE ENGINE: TEXT-TO-SPEECH (TTS) & SPEECH-TO-TEXT (STT)
# ==============================================================================

def text_to_speech(
    text: str,
    language_code: str = "mr-IN",
    speaker: str = "aditya",
    pace: float = 1.0,
    pitch: float = 0.0,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Converts text to natural human speech using Sarvam AI's 'bulbul:v3' model.
    Returns audio bytes and base64 encoded audio string ready for Streamlit audio playback.
    """
    if not text or not text.strip():
        return {"success": False, "audio_bytes": None, "error": "Input text for speech synthesis cannot be empty."}

    client, err = get_sarvam_client(api_key)
    if not client:
        return {"success": False, "audio_bytes": None, "error": err}

    selected_speaker = speaker if speaker in AVAILABLE_SPEAKERS else "aditya"
    valid_lang = language_code if language_code in SUPPORTED_LANGUAGES else "mr-IN"

    try:
        res = client.text_to_speech.convert(
            text=text[:500],
            language_code=valid_lang,
            model="bulbul:v3",
            speaker=selected_speaker,
            pace=pace,
            pitch=pitch
        )
        if res.audios and len(res.audios) > 0:
            b64_audio = res.audios[0]
            audio_bytes = base64.b64decode(b64_audio)
            return {
                "success": True,
                "audio_bytes": audio_bytes,
                "audio_base64": b64_audio,
                "language_code": valid_lang,
                "speaker": selected_speaker,
                "error": None
            }
        return {"success": False, "audio_bytes": None, "error": "No audio was returned from Sarvam TTS service."}
    except Exception as e:
        return {"success": False, "audio_bytes": None, "error": f"TTS synthesis failed: {str(e)}"}


def speech_to_text(
    audio_data: Union[bytes, Any],
    language_code: str = "unknown",
    model: str = "saaras:v3",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transcribes spoken audio into text using Sarvam AI's 'saaras:v3' speech-to-text model.
    Accepts raw audio bytes or a file-like object from Streamlit audio recorder.
    """
    if not audio_data:
        return {"success": False, "transcript": "", "error": "Audio data is required."}

    client, err = get_sarvam_client(api_key)
    if not client:
        return {"success": False, "transcript": "", "error": err}

    if hasattr(audio_data, "read"):
        raw_bytes = audio_data.read()
    elif isinstance(audio_data, bytes):
        raw_bytes = audio_data
    else:
        return {"success": False, "transcript": "", "error": "Invalid audio data format."}

    if len(raw_bytes) < 100:
        return {"success": False, "transcript": "", "error": "Audio recording is too short or empty."}

    try:
        res = client.speech_to_text.transcribe(
            file=("audio.wav", raw_bytes, "audio/wav"),
            model=model,
            language_code=language_code if language_code in SUPPORTED_LANGUAGES or language_code == "unknown" else "unknown"
        )
        transcript = getattr(res, "transcript", "")
        detected_lang = getattr(res, "language_code", language_code)
        return {
            "success": True,
            "transcript": transcript,
            "detected_language": detected_lang,
            "error": None
        }
    except Exception as e:
        return {"success": False, "transcript": "", "error": f"Speech transcription failed: {str(e)}"}


# ==============================================================================
# 4. CONVERSATIONAL SARVAM AI CHATBOT & VOICE COMMAND PROCESSOR
# ==============================================================================

def chat_with_sarvam(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Converses with Sarvam AI's multilingual chat model (sarvam-105b-conversations).
    Maintains full conversational flow across English, Hindi, and Marathi.
    """
    client, err = get_sarvam_client(api_key)
    if not client:
        return {"success": False, "reply": "", "error": err}

    default_system = (
        "You are LingoAI Assistant, an advanced multilingual AI chatbot for Indian languages "
        "(English, Hindi, Marathi) created for LingoAI Studio. You can converse naturally in English, Hindi, "
        "and Marathi. You can understand voice commands to translate, summarize, analyze sentiment, "
        "or answer general queries. Always be respectful, highly accurate, and reply in the same language "
        "the user uses unless explicitly asked to translate."
    )

    sys_content = system_prompt or default_system
    full_messages = [{"role": "system", "content": sys_content}] + messages

    try:
        res = client.chat.completions(
            model="sarvam-105b-conversations",
            messages=full_messages,
            temperature=0.3
        )
        reply = res.choices[0].message.content or ""
        return {
            "success": True,
            "reply": reply.strip(),
            "error": None
        }
    except Exception as e:
        return {"success": False, "reply": "", "error": f"Chatbot error: {str(e)}"}


def parse_and_execute_voice_command(
    command_text: str,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Interprets user voice command and executes NLP operations or routes to conversational AI.
    """
    cleaned = command_text.strip()
    lower_cmd = cleaned.lower()

    # Pattern 1: Translation Voice Command
    tr_match = re.search(r'(?:translate|भाषांतर करा|अनुवाद करो)\s+[\'"]?(.+?)[\'"]?\s+(?:to|into|मध्ये|में)\s+(marathi|hindi|english|मराठी|हिंदी|इंग्रजी)', cleaned, re.IGNORECASE)
    if tr_match:
        text_to_tr = tr_match.group(1)
        target_name = tr_match.group(2).lower()
        target_code = "mr-IN" if any(x in target_name for x in ["marathi", "मराठी"]) else ("hi-IN" if any(x in target_name for x in ["hindi", "हिंदी"]) else "en-IN")
        tr_res = translate_text(text_to_tr, source_lang="auto", target_lang=target_code, api_key=api_key)
        if tr_res.get("success"):
            return {
                "action": "translate",
                "original": text_to_tr,
                "result": tr_res["translated_text"],
                "target_lang": target_code,
                "reply": f"Here is the translation:\n\n**{tr_res['translated_text']}**"
            }

    # Pattern 2: Sentiment Voice Command
    if any(k in lower_cmd for k in ["sentiment", "भावना", "भावना विश्लेषण"]):
        nlp_res = analyze_nlp(cleaned, api_key=api_key)
        if nlp_res.get("success"):
            sent = nlp_res["sentiment"]
            return {
                "action": "sentiment",
                "result": sent,
                "reply": f"The sentiment of your statement is **{sent['label']}** (Confidence: {int(sent['confidence']*100)}%).\n\n*Explanation:* {sent['explanation']}"
            }

    # Pattern 3: Summarization Voice Command
    if lower_cmd.startswith("summarize") or "सारांश" in lower_cmd:
        passage = re.sub(r'^(?:summarize|सारांश करा|सारांश निकालो)[:\s]*', '', cleaned, flags=re.IGNORECASE)
        if len(tokenize_words(passage)) >= 10:
            sum_res = generate_summary(passage, style="paragraph", api_key=api_key)
            if sum_res.get("success"):
                return {
                    "action": "summarize",
                    "result": sum_res,
                    "reply": f"**Summary:**\n{sum_res['summary']}\n\n*(Reduced by {sum_res['reduction_percentage']}%)*"
                }

    # Default: General Multilingual Chatbot Conversation
    chat_res = chat_with_sarvam([{"role": "user", "content": cleaned}], api_key=api_key)
    return {
        "action": "chat",
        "reply": chat_res.get("reply", "I am listening. How can I help you today?"),
        "error": chat_res.get("error")
    }


# ==============================================================================
# 5. EVALUATION FRAMEWORK & METRICS CALCULATION
# ==============================================================================

def calculate_evaluation_metrics(
    y_true: List[str],
    y_pred: List[str],
    classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculates true mathematical evaluation metrics:
    - Overall Accuracy
    - Per-class Precision, Recall, and F1-Score
    - Macro-averaged Precision, Recall, and F1
    - Full Confusion Matrix
    Never fabricates metrics.
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and of equal length.")

    total_samples = len(y_true)
    y_true_clean = [str(y).strip().capitalize() for y in y_true]
    y_pred_clean = [str(y).strip().capitalize() for y in y_pred]

    if classes is None:
        classes = sorted(list(set(y_true_clean) | set(y_pred_clean)))
    else:
        classes = [str(c).strip().capitalize() for c in classes]

    cm = {c_true: {c_pred: 0 for c_pred in classes} for c_true in classes}
    for t, p in zip(y_true_clean, y_pred_clean):
        if t in cm and p in cm[t]:
            cm[t][p] += 1
        elif t in cm:
            if "Other" in cm[t]:
                cm[t]["Other"] += 1

    correct_predictions = sum(cm[c][c] for c in classes)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

    per_class_metrics = {}
    precisions = []
    recalls = []
    f1s = []

    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[other][c] for other in classes if other != c)
        fn = sum(cm[c][other] for other in classes if other != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_metrics[c] = {
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }

        if (tp + fn) > 0:
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    z_matrix = [[cm[r][col] for col in classes] for r in classes]

    return {
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class_metrics,
        "classes": classes,
        "confusion_matrix_dict": cm,
        "confusion_matrix_grid": z_matrix
    }


def run_batch_evaluation(
    df: pd.DataFrame,
    task: str = "sentiment",
    text_col: str = "text",
    label_col: str = "true_sentiment",
    max_items: int = 15,
    api_key: Optional[str] = None,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Executes actual model predictions over a dataset and computes true evaluation metrics.
    """
    sample_df = df.head(max_items).copy()
    total = len(sample_df)
    
    y_true = []
    y_pred = []
    explanations = []

    for idx, row in sample_df.iterrows():
        text_val = str(row[text_col])
        true_val = str(row[label_col])
        
        analysis = analyze_nlp(text_val, api_key=api_key)
        
        if analysis.get("success"):
            if task == "sentiment":
                pred_label = analysis["sentiment"]["label"]
                exp = analysis["sentiment"]["explanation"]
            else:
                pred_label = analysis["classification"]["topic"]
                exp = analysis["classification"]["explanation"]
        else:
            pred_label = "Neutral" if task == "sentiment" else "Other"
            exp = analysis.get("error", "API request failed")

        y_true.append(true_val)
        y_pred.append(pred_label)
        explanations.append(exp)

        if progress_callback:
            progress_callback(len(y_true), total)

    sample_df["predicted"] = y_pred
    sample_df["model_explanation"] = explanations
    sample_df["is_correct"] = [t.strip().lower() == p.strip().lower() for t, p in zip(y_true, y_pred)]

    target_classes = SENTIMENT_CLASSES if task == "sentiment" else TOPIC_CATEGORIES
    metrics = calculate_evaluation_metrics(y_true, y_pred, classes=target_classes)

    return {
        "results_df": sample_df,
        "metrics": metrics,
        "task": task
    }
