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

# Custom CSS (Minimalist Professional)
st.markdown("""
<style>
    :root {
        --primary: #2E7D32;
        --secondary: #1565C0;
        --accent: #FF8F00;
        --text: #263238;
        --bg: #FAFAFA;
    }
    
    .stApp {
        background-color: var(--bg);
    }
    
    .header {
        color: var(--primary);
        border-bottom: 2px solid var(--primary);
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--primary);
    }
    
    .input-label {
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.5rem;
    }
    
    .result-card {
        background: #E8F5E9;
        border-left: 4px solid var(--primary);
    }
    
    .voice-btn {
        background: var(--secondary) !important;
        color: white !important;
        border-radius: 20px !important;
    }
    
    .tab-button {
        font-weight: 600 !important;
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
