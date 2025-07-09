import streamlit as st
import pandas as pd
import joblib
from gtts import gTTS
from io import BytesIO
import numpy as np
from weather import get_weather, get_weather_telugu, get_weather_hindi
import os
from dotenv import load_dotenv

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
    :root {
        --primary: #00C853;   /* Vibrant green (farm tech) */
        --secondary: #2962FF; /* Electric blue */
        --accent: #FFA000;    /* Amber for attention */
        --text: #000000;      /* Rich gray-black */
        --bg: #F1F8E9;        /* Light lime background (eco friendly) */
        --card-bg: #FFFFFF;   /* Clean white for cards */
    }

    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }

    h1, h2, h3 {
        font-weight: 700;
        color: #000000;
    }

    .stApp {
        background-color: var(--bg);
    }

    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border: 1px solid #A5D6A7;
        border-radius: 12px;
        padding: 10px 16px;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0, 200, 83, 0.3);
        transition: 0.3s ease-in-out;
    }

    .stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 12px rgba(0, 200, 83, 0.4);
    }

    .block-container {
        padding: 2rem 2rem 3rem;
    }

    /* Result Cards */
    .result-card {
        background: var(--card-bg);
        border-left: 4px solid var(--primary);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        animation: slideIn 0.4s ease;
    }

    @keyframes slideIn {
        from { transform: translateY(10px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    /* Voice icon style (if used) */
    .voice-btn {
        background-color: var(--secondary);
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(41, 98, 255, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(41, 98, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(41, 98, 255, 0); }
    }

</style>
""", unsafe_allow_html=True)
# ---- Core Functions ----
def load_models():
    try:
        crop_model = joblib.load("crop_model (1).pkl")
        le = joblib.load("label_encoder (1).pkl")
        return crop_model, le
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        fp = BytesIO()
        tts.write_to_fp(fp)
        st.audio(fp.getvalue(), format='audio/mp3')
    except Exception as e:
        st.warning(f"Speech synthesis failed: {str(e)}")

# ---- Main App ----
def main():
    st.title("🌿 AgriVoice Pro")
    st.markdown("### AI-Powered Agricultural Assistant")
    
    # Language selection
    lang = st.radio("Language", ["English", "తెలుగు", "हिन्दी"], horizontal=True)
    
    # Tab layout
    tab1, tab2 = st.tabs(["🌤️ Weather Forecast", "🌱 Crop Recommendation"])
    
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

if __name__ == "__main__":
    main()
