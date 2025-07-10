import streamlit as st
import pandas as pd
import joblib
from gtts import gTTS
from io import BytesIO
from datetime import datetime
import numpy as np
from weather import get_weather, get_weather_telugu, get_weather_hindi
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
import torch
import base64
import re
import tempfile
import os
from dotenv import load_dotenv
import speech_recognition as sr

# ------------------------ Speech Recognition (No PyAudio Needed) ------------------------
def recognize_uploaded_audio(uploaded_file):
    recognizer = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        with sr.AudioFile(tmp_file_path) as source:
            audio = recognizer.record(source)
            return recognizer.recognize_google(audio)
    except Exception as e:
        return f"Speech recognition failed: {str(e)}"

# ------------------------ Translate & Intent ------------------------
def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def detect_intent(text):
    text = text.lower()
    intent_map = {
        'weather': ['weather', 'climate', 'rain', 'forecast', 'temperature'],
        'crop': ['crop', 'plant', 'grow', 'soil', 'recommendation'],
        'pest': ['pest', 'insect', 'disease', 'spray', 'pesticide'],
        'scheme': ['scheme', 'yojana', 'benefit', 'pm', 'government']
    }
    for intent, keywords in intent_map.items():
        if any(re.search(rf'\b{kw}\b', text) for kw in keywords):
            return intent
    return "unknown"

# ------------------------ Text to Speech ------------------------
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    fp = BytesIO()
    tts.write_to_fp(fp)
    st.audio(fp.getvalue(), format='audio/mp3')

# ------------------------ Background ------------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def set_jpg_as_page_bg(jpg_file):
    bin_str = get_base64_of_bin_file(jpg_file)
    page_bg_img = f"""
    <style>
    body {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ------------------------ Loaders ------------------------
@st.cache_resource
def load_models():
    model = joblib.load("crop_model (1).pkl")
    encoder = joblib.load("label_encoder (1).pkl")
    return model, encoder

@st.cache_resource
def load_qna_data():
    df = pd.read_csv("gov_schemes_dataset.csv")
    df['context'] = df[['Scheme Name', 'Description', 'Eligibility', 'Benefits']].astype(str).agg(' '.join, axis=1)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    corpus = df['context'].tolist()
    embeddings = model.encode(corpus, convert_to_tensor=True)
    return df, model, embeddings

# ------------------------ Pest Plan ------------------------
def generate_pest_plan(crop, area):
    df = pd.read_csv("pest_db.csv")
    filtered = df[df["Crop"].str.lower() == crop.lower()].copy()
    if filtered.empty:
        return None
    filtered["Total_Dose"] = filtered["Dose_per_ha"] * area
    return filtered

# ------------------------ MAIN APP ------------------------
def main():
    st.set_page_config("AgriVoice Pro", layout="wide")
    set_jpg_as_page_bg("csp background.jpg")

    st.title("🌿 AgriVoice Pro")
    st.markdown("### Your AI Agriculture Companion")

    tabs = st.tabs(["🏠 Home", "🌤️ Weather", "🌾 Crop", "🏛 Schemes", "🐛 Pests"])
    tab_names = ["home", "weather", "crop", "scheme", "pest"]
    selected_tab = 0  # default

    # ---------- HOME ----------
    with tabs[0]:
        st.subheader("👋 Welcome Farmer!")

        lang = st.radio("Choose language", ["English", "తెలుగు", "हिन्दी"], horizontal=True)

        greeting = "Good Morning" if datetime.now().hour < 12 else "Good Evening"
        st.markdown(f"### {greeting}, Farmer 👨‍🌾")

        user_query = st.text_input("📝 Ask your question:")
        audio_file = st.file_uploader("🎙 Upload your voice (WAV only)", type=["wav"])

        if audio_file:
            transcript = recognize_uploaded_audio(audio_file)
            st.success(f"You said: {transcript}")
            user_query = transcript

        if user_query:
            translated = translate_to_english(user_query)
            intent = detect_intent(translated)

            st.info(f"✅ Detected Intent: **{intent.capitalize()}**")

            # Switch Tab based on intent
            if intent == "weather":
                selected_tab = 1
            elif intent == "crop":
                selected_tab = 2
            elif intent == "scheme":
                selected_tab = 3
            elif intent == "pest":
                selected_tab = 4
            else:
                st.warning("Sorry, I couldn’t understand. Please rephrase.")

    # ---------- WEATHER ----------
    with tabs[1]:
        st.header("🌦️ Weather Forecast")
        city = st.text_input("Enter your city")
        if st.button("Get Weather"):
            if city:
                report = get_weather_telugu(city) if lang == "తెలుగు" else get_weather_hindi(city) if lang == "हिन्दी" else get_weather(city)
                st.success(report)
                text_to_speech(report, 'te' if lang == "తెలుగు" else 'hi' if lang == "हिन्दी" else 'en')

    # ---------- CROP ----------
    with tabs[2]:
        st.header("🌾 Crop Recommendation")
        col1, col2 = st.columns(2)
        with col1:
            n = st.number_input("Nitrogen", 0, 200)
            p = st.number_input("Phosphorus", 0, 200)
            k = st.number_input("Potassium", 0, 200)
        with col2:
            temp = st.number_input("Temperature (°C)", 0.0, 50.0)
            hum = st.number_input("Humidity (%)", 0.0, 100.0)
            ph = st.number_input("pH Level", 0.0, 14.0)
            rain = st.number_input("Rainfall (mm)", 0.0, 500.0)

        if st.button("Get Recommendation"):
            model, le = load_models()
            pred = model.predict([[n, p, k, temp, hum, ph, rain]])[0]
            crop = le.inverse_transform([pred])[0]
            msg = f"Recommended crop: {crop}" if lang == "English" else f"సిఫారసు చేసిన పంట: {crop}" if lang == "తెలుగు" else f"अनुशंसित फसल: {crop}"
            st.success(msg)
            text_to_speech(msg, 'en' if lang == "English" else 'te' if lang == "తెలుగు" else 'hi')

    # ---------- SCHEMES ----------
    with tabs[3]:
        st.header("🏛 Government Schemes Assistant")
        df, model, embeddings = load_qna_data()

        question = st.text_input("Ask about any scheme:")
        if st.button("Get Info"):
            if question:
                question_en = translate_to_english(question)
                q_embed = model.encode(question_en, convert_to_tensor=True)
                best_match = torch.argmax(util.pytorch_cos_sim(q_embed, embeddings)).item()
                answer = df.iloc[best_match]['Description']
                translated_ans = GoogleTranslator(source='en', target='te' if lang == "తెలుగు" else 'hi' if lang == "हिन्दी" else 'en').translate(answer)
                st.success(translated_ans)
                text_to_speech(translated_ans, 'te' if lang == "తెలుగు" else 'hi' if lang == "हिन्दी" else 'en')

    # ---------- PEST ----------
    with tabs[4]:
        st.header("🐛 Pest Management")
        crop = st.text_input("Enter Crop Name")
        area = st.number_input("Area in hectares", min_value=0.1)

        if st.button("Get Pest Plan"):
            if crop and area:
                plan = generate_pest_plan(crop, area)
                if plan is not None:
                    for _, row in plan.iterrows():
                        summary = f"For {row['Crop']} affected by {row['Pest_Disease']}, use {row['Pesticide']} - Dose: {row['Total_Dose']} {row['Unit']}. Note: {row['Notes']}"
                        trans = GoogleTranslator(source='en', target='te' if lang == "తెలుగు" else 'hi' if lang == "हिन्दी" else 'en').translate(summary)
                        st.success(trans)
                        text_to_speech(trans, 'te' if lang == "తెలుగు" else 'hi' if lang == "हिन्दी" else 'en')
                else:
                    st.error("No data found for that crop.")

if __name__ == "__main__":
    load_dotenv()
    main()
