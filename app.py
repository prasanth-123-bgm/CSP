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
        --primary: #2E7D32;        /* Forest Green */
        --secondary: #00B0FF;      /* Aqua Tech Blue */
        --accent: #FF9100;         /* Vibrant Orange */
        --text: #1A1A1A;           /* Deep readable text */
        --bg: #F4FDF4;             /* Soft eco background */
        --card-bg: rgba(255, 255, 255, 0.9);
        --footer-bg: #004D40;
        --input-bg: #ffffff;
        --input-border: #2E7D32;
        --shadow: rgba(0,0,0,0.15);
    }

    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }

    .stApp {
        background-color: var(--bg);
    }

    h1, h2, h3, label, p, span, .stRadio label, .stMarkdown {
        color: var(--text);
        font-weight: 700;
        text-shadow: 0 1px 1px rgba(0,0,0,0.05);
    }

    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: var(--input-bg);
        color: var(--text);
        border: 2px solid var(--input-border);
        border-radius: 12px;
        padding: 12px 18px;
        font-size: 1rem;
        transition: border-color 0.3s ease-in-out;
        box-shadow: 0 2px 6px var(--shadow);
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--secondary);
        outline: none;
    }

    input::placeholder {
        color: #607D8B;  /* Visible gray */
        font-weight: 500;
        opacity: 1;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.4rem;
        font-size: 1rem;
        font-weight: bold;
        box-shadow: 0 6px 12px rgba(0, 150, 136, 0.3);
        transition: all 0.3s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.03);
        background: linear-gradient(135deg, var(--secondary), var(--accent));
        box-shadow: 0 8px 18px rgba(0, 150, 136, 0.45);
    }

    /* Voice Button */
    .voice-btn {
        background-color: var(--accent);
        color: white;
        border-radius: 50%;
        width: 52px;
        height: 52px;
        font-size: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: pulse 1.8s infinite;
        box-shadow: 0 0 14px rgba(255, 145, 0, 0.6);
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 145, 0, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(255, 145, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 145, 0, 0); }
    }

    /* Result Card */
    .result-card {
        background: var(--card-bg);
        border-left: 5px solid var(--primary);
        padding: 1.5rem 2rem;
        border-radius: 18px;
        margin-top: 1.5rem;
        backdrop-filter: blur(6px);
        box-shadow: 0 12px 24px var(--shadow);
        animation: fadeSlideIn 0.5s ease-out;
    }

    @keyframes fadeSlideIn {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    /* Navbar */
    .navbar {
        background: linear-gradient(to right, var(--primary), var(--secondary));
        padding: 1rem 2rem;
        color: white;
        font-size: 1.3rem;
        font-weight: bold;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 16px var(--shadow);
        text-align: center;
    }

    /* Footer */
    .footer {
        background-color: var(--footer-bg);
        color: #E0F2F1;
        text-align: center;
        padding: 1rem;
        font-size: 0.95rem;
        border-radius: 20px 20px 0 0;
        margin-top: 4rem;
        box-shadow: 0 -4px 10px var(--shadow);
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
