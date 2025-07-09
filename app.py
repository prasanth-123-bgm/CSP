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
/* Background Image */
body {
    background-image: url('csp background.jpg');
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}

/* Overlay tint for readability */
.stApp {
    background-color: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 0 40px rgba(0,0,0,0.2);
}

/* Headings and text */
h1, h2, h3, label, p, span, .stMarkdown, .stRadio label {
    color: #1A1A1A;
    font-weight: 700;
    text-shadow: 1px 1px 2px rgba(255,255,255,0.6);
}

/* Navbar */
.navbar {
    background: linear-gradient(to right, #2E7D32, #00B0FF);
    padding: 1rem 2rem;
    color: white;
    font-size: 1.4rem;
    font-weight: bold;
    text-align: center;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}

/* Footer */
.footer {
    background-color: #004D40;
    color: #ffffff;
    padding: 1rem;
    border-radius: 16px;
    text-align: center;
    font-size: 0.9rem;
    margin-top: 3rem;
    box-shadow: 0 -4px 10px rgba(0,0,0,0.2);
}

/* Input styling */
input, .stNumberInput input {
    background-color: #ffffff !important;
    color: #1A1A1A !important;
    border: 2px solid #2E7D32;
    border-radius: 10px;
    padding: 10px;
    font-weight: 600;
    font-size: 1rem;
}

/* Placeholder */
input::placeholder {
    color: #4E944F;
    font-weight: 500;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00C853, #00B0FF);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.3rem;
    font-weight: bold;
    box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    transition: 0.3s ease-in-out;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.03);
    background: linear-gradient(135deg, #00B0FF, #FFA000);
    box-shadow: 0 8px 18px rgba(0,0,0,0.3);
}

/* Result Cards */
.result-card {
    background-color: rgba(255, 255, 255, 0.95);
    padding: 1.2rem 1.6rem;
    border-left: 6px solid #00C853;
    border-radius: 14px;
    margin-top: 1.5rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    animation: slideIn 0.5s ease;
}

@keyframes slideIn {
    from { transform: translateY(20px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
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
