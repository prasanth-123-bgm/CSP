
import streamlit as st
import pandas as pd
import joblib
from gtts import gTTS
from io import BytesIO
import numpy as np
from weather import get_weather, get_weather_telugu, get_weather_hindi
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
import torch
import base64

# 🌄 Background Image Setup
@st.cache_resource
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

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

# 🖼️ Call the function with your image
set_jpg_as_page_bg("csp background.jpg")


# Load environment variables
load_dotenv()

# ---- UI Configuration ----
st.set_page_config(
    page_title="🌿 AgriVoice Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌱"
)

st.markdown("""
<style>
/* === GLOBAL RESET & FONTS === */
* {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    transition: all 0.3s ease-in-out;
    scroll-behavior: smooth;
}

/* === APP CONTAINER WITH GLASS EFFECT === */
.stApp {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 0 30px rgba(0, 255, 255, 0.1);
}

/* === HEADERS === */
h1, h2, h3, .stMarkdown, .stRadio label {
    color: #00fff2;
    font-weight: 700;
    text-shadow: 0 0 4px #00c9a7;
}

/* === INPUT FIELDS === */
input, .stNumberInput input {
    background-color: rgba(0, 0, 0, 0.6) !important;
    color: #00e5ff !important;
    border: 1.5px solid #00ffc3;
    border-radius: 12px;
    padding: 0.6rem;
    font-weight: 600;
}
input::placeholder {
    color: #80cbc4;
    font-weight: 500;
}

/* === BUTTON STYLING === */
.stButton > button {
    background: linear-gradient(135deg, #00e676, #00b0ff);
    color: white;
    border: none;
    border-radius: 16px;
    padding: 0.7rem 1.4rem;
    font-weight: bold;
    box-shadow: 0 0 18px rgba(0, 255, 200, 0.3);
    text-shadow: 0 0 3px rgba(255, 255, 255, 0.3);
    transform: scale(1);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #00b0ff, #ff6f00);
    box-shadow: 0 0 24px rgba(255, 111, 0, 0.4);
    transform: scale(1.05) rotate(-0.5deg);
}

/* === TABS === */
.stTabs [role="tab"] {
    background: #212121;
    color: #00e5ff;
    padding: 0.6rem 1.2rem;
    border-radius: 12px 12px 0 0;
    font-weight: bold;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.stTabs [role="tab"][aria-selected="true"] {
    background: linear-gradient(to right, #00c9a7, #00b0ff);
    color: #ffffff;
    box-shadow: 0 6px 16px rgba(0, 255, 255, 0.3);
}

/* === RESULT CARD === */
.result-card {
    background: rgba(0, 0, 0, 0.6);
    border-left: 6px solid #00e676;
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1.5rem;
    color: #e0f7fa;
    box-shadow: 0 0 20px rgba(0,255,200,0.2);
    animation: floatIn 0.6s ease;
}

/* === CUSTOM ANIMATION === */
@keyframes floatIn {
    from { transform: translateY(30px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

/* === FOOTER === */
.footer {
    background: linear-gradient(to right, #212121, #004d40);
    color: #ffffff;
    padding: 1rem;
    text-align: center;
    border-radius: 16px;
    font-size: 0.9rem;
    box-shadow: 0 -2px 8px rgba(0, 255, 200, 0.1);
}

/* === SCROLLBAR STYLING === */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(#00c9a7, #00b0ff);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# UI Config
st.set_page_config(
    page_title="🌿 AgriVoice Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌱"
)

# 🌟 CSS Styling
st.markdown("""
<style>
/* Include your full CSS styles here */
</style>
""", unsafe_allow_html=True)

# Load ML Models
def load_models():
    try:
        crop_model = joblib.load("crop_model (1).pkl")
        le = joblib.load("label_encoder (1).pkl")
        return crop_model, le
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# Load Scheme Dataset + Model
@st.cache_resource
def load_qna_data():
    df = pd.read_csv("gov_schemes_dataset.csv")
    df['context'] = df[['Scheme Name', 'Description', 'Eligibility', 'Benefits']].astype(str).agg(' '.join, axis=1)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    corpus = df['context'].tolist()
    embeddings = model.encode(corpus, convert_to_tensor=True)
    return df, model, embeddings

# Text-to-Speech
def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        fp = BytesIO()
        tts.write_to_fp(fp)
        st.audio(fp.getvalue(), format='audio/mp3')
    except Exception as e:
        st.warning(f"Speech synthesis failed: {str(e)}")

# Main App
def main():
    st.title("🌿 AgriVoice Pro")
    st.markdown("### AI-Powered Agricultural Assistant")

    lang = st.radio("Language", ["English", "తెలుగు", "हिन्दी"], horizontal=True)

    tab1, tab2, tab3 ,tab4= st.tabs(["🌤️ Weather Forecast", "🌱 Crop Recommendation", "🏛 Government Schemes","🐛 Pest Management"])

    with tab1:
        st.markdown("### Weather Forecast", help="Get real-time weather data")
        city = st.text_input("Enter city name", key="weather_city")
        if st.button("Get Weather", key="weather_btn"):
            if city:
                with st.spinner("Fetching weather..."):
                    if lang == "తెలుగు":
                        report = get_weather_telugu(city)
                    elif lang == "हिन्दी":
                        report = get_weather_hindi(city)
                    else:
                        report = get_weather(city)

                    st.markdown(f"""
                    <div class='card result-card'>
                        <h4>Weather Report</h4>
                        <p>{report}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    text_to_speech(report, 'te' if lang == "తెలుగు" else 'hi' if lang == "हिन्दी" else 'en')

    with tab2:
        st.markdown("### Soil Analysis", help="Enter soil parameters for crop recommendation")
        col1, col2 = st.columns(2)
        with col1:
            n = st.number_input("Nitrogen (N)", min_value=0, max_value=200)
            p = st.number_input("Phosphorus (P)", min_value=0, max_value=200)
            k = st.number_input("Potassium (K)", min_value=0, max_value=200)
        with col2:
            temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1)
            hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
            ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1)
            rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)

        if st.button("Get Recommendation", type="primary"):
            crop_model, le = load_models()
            features = [[n, p, k, temp, hum, ph, rain]]
            try:
                prediction = crop_model.predict(features)[0]
                crop = le.inverse_transform([prediction])[0]
                if lang == "తెలుగు":
                    msg = f"సిఫారసు చేసిన పంట: {crop}"
                elif lang == "हिन्दी":
                    msg = f"अनुशंसित फसल: {crop}"
                else:
                    msg = f"Recommended crop: {crop}"
                st.success(msg)
                text_to_speech(msg, 'te' if lang == "తెలుగు" else 'hi' if lang == "हिन्दी" else 'en')
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

    with tab3:
        st.markdown("### Government Scheme Assistant", help="Ask about any scheme in your preferred language")
        df, qna_model, qna_embeddings = load_qna_data()

        user_question = st.text_input("Ask your question here (in English, తెలుగు, हिन्दी):", key="gov_q")
        if st.button("Get Scheme Info", key="gov_btn"):
            if user_question:
                question_en = GoogleTranslator(source='auto', target='en').translate(user_question)
                q_embedding = qna_model.encode(question_en, convert_to_tensor=True)
                similarities = util.pytorch_cos_sim(q_embedding, qna_embeddings)[0]
                best_idx = torch.argmax(similarities).item()
                answer_en = df.iloc[best_idx]['Description']

                lang_code = 'en'
                if lang == "తెలుగు":
                    lang_code = 'te'
                elif lang == "हिन्दी":
                    lang_code = 'hi'
                answer_native = GoogleTranslator(source='en', target=lang_code).translate(answer_en)

                st.markdown(f"""
                <div class='card result-card'>
                    <h4>📘 Scheme Info:</h4>
                    <p>{answer_native}</p>
                </div>
                """, unsafe_allow_html=True)
                text_to_speech(answer_native, lang=lang_code)
    with tab4:
    st.subheader("🐛 Pest Management Assistant")
    st.markdown("Get eco-friendly pesticide recommendations based on your crop and field area.")

    crop_input = st.text_input("Enter Crop Name (e.g., Rice, Chilli, Cotton)")
    area_input = st.number_input("Enter Field Area (in hectares)", min_value=0.1, step=0.1)
    lang_input = st.radio("Preferred Language for Output", ["English", "తెలుగు", "हिन्दी"], horizontal=True)

    def generate_pest_plan(crop, area):
        df = pd.read_csv("pest_db.csv")
        filtered = df[df["Crop"].str.lower() == crop.lower()].copy()
        if filtered.empty:
            return None
        filtered["Total_Dose"] = filtered.apply(
            lambda row: round(row["Dose_per_ha"] * area, 2) if pd.notnull(row["Dose_per_ha"]) else None,
            axis=1
        )
        return filtered

    def speak_text(text, lang='en'):
        try:
            tts = gTTS(text=text, lang=lang)
            fp = BytesIO()
            tts.write_to_fp(fp)
            st.audio(fp.getvalue(), format='audio/mp3')
        except:
            st.warning("Speech synthesis failed.")

    def translate_text(text, lang_code="te"):
        try:
            return GoogleTranslator(source='en', target=lang_code).translate(text)
        except:
            return text

    if st.button("Get Pest Management Plan"):
        if crop_input and area_input:
            plan = generate_pest_plan(crop_input, area_input)
            if plan is not None:
                for _, row in plan.iterrows():
                    english_text = f"For {row['Crop']} affected by {row['Pest_Disease']}, use {row['Pesticide']}. Required dose: {row['Total_Dose']} {row['Unit']}. Note: {row['Notes']}"
                    if lang_input == "తెలుగు":
                        translated = translate_text(english_text, "te")
                        st.success(translated)
                        speak_text(translated, "te")
                    elif lang_input == "हिन्दी":
                        translated = translate_text(english_text, "hi")
                        st.success(translated)
                        speak_text(translated, "hi")
                    else:
                        st.success(english_text)
                        speak_text(english_text, "en")
            else:
                st.error("No pest management data found for the selected crop.")
        else:
            st.warning("Please enter crop and area details.")


if __name__ == "__main__":
    main()
