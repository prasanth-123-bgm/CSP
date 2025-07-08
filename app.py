import streamlit as st
import pandas as pd
import pickle
import joblib
from gtts import gTTS
from io import BytesIO
import speech_recognition as sr
import numpy as np
from weather import get_weather, get_weather_telugu, get_weather_hindi
# First uninstall any existing PyAudio
pip uninstall PyAudio

# Install with pipwin (Windows-specific)
pip install pipwin
pipwin install PyAudio
pip install --upgrade pip

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
    }
    
    .input-group {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .voice-button {
        background: linear-gradient(45deg, #FF6B6B, #FF8E8E);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        cursor: pointer;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .voice-button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(255,107,107,0.3);
    }
    
    .result-box {
        background: linear-gradient(135deg, #a8e6cf 0%, #dcedc8 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4caf50;
    }
    
    .listening-indicator {
        background: #ff4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Safe model loading
try:
    crop_model = joblib.load("crop_model (1).pkl")
except Exception as e:
    st.error(f"❌ crop_model.pkl loading failed: {e}")
    st.stop()

try:
    le = joblib.load("label_encoder (1).pkl")
except Exception as e:
    st.error(f"❌ label_encoder.pkl loading failed: {e}")
    st.stop()

# Speech recognition setup
recognizer = sr.Recognizer()

def speak(text, lang='te', autoplay=True):
    """Enhanced speak function with autoplay option"""
    try:
        tts = gTTS(text=text, lang=lang)
        fp = BytesIO()
        tts.write_to_fp(fp)
        
        if autoplay:
            # Auto-play audio
            st.audio(fp.getvalue(), format='audio/mp3', autoplay=True)
        else:
            st.audio(fp.getvalue(), format='audio/mp3')
        return True
    except Exception as e:
        st.warning(f"Voice output failed: {e}")
        return False

def listen_to_speech(lang_code='en-US'):
    """Speech recognition function"""
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            st.info("🎤 Listening... Please speak now")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
        # Convert speech to text
        if lang_code == 'te-IN':
            text = recognizer.recognize_google(audio, language='te-IN')
        elif lang_code == 'hi-IN':
            text = recognizer.recognize_google(audio, language='hi-IN')
        else:
            text = recognizer.recognize_google(audio, language='en-US')
            
        return text
    except sr.RequestError:
        st.error("Could not connect to speech recognition service")
        return None
    except sr.UnknownValueError:
        st.warning("Could not understand audio. Please try again.")
        return None
    except sr.WaitTimeoutError:
        st.warning("No speech detected. Please try again.")
        return None
    except Exception as e:
        st.error(f"Speech recognition error: {e}")
        return None

def voice_input_field(label, key, input_type="text", min_val=None, max_val=None, step=None):
    """Custom voice input field component"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if input_type == "text":
            value = st.text_input(label, key=key)
        elif input_type == "number":
            value = st.number_input(label, min_value=min_val, max_value=max_val, step=step, key=key)
        elif input_type == "float":
            value = st.number_input(label, min_value=min_val, max_value=max_val, step=step, key=key)
    
    with col2:
        if st.button(f"🎤", key=f"voice_{key}", help="Click to speak"):
            with st.spinner("Listening..."):
                lang_code = 'te-IN' if st.session_state.get('language') == 'తెలుగు' else 'hi-IN' if st.session_state.get('language') == 'हिन्दी' else 'en-US'
                speech_text = listen_to_speech(lang_code)
                
                if speech_text:
                    st.success(f"Heard: {speech_text}")
                    
                    # Process the speech input based on type
                    if input_type in ["number", "float"]:
                        try:
                            # Extract numbers from speech
                            import re
                            numbers = re.findall(r'\d+\.?\d*', speech_text)
                            if numbers:
                                value = float(numbers[0]) if input_type == "float" else int(numbers[0])
                                st.session_state[key] = value
                                st.experimental_rerun()
                        except:
                            st.warning("Could not extract number from speech")
                    else:
                        st.session_state[key] = speech_text
                        st.experimental_rerun()
    
    return value

# Page configuration
st.set_page_config(
    page_title="🌿 AI Agri Voice Assistant", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🌿 AI Agri Voice Assistant</h1>
    <p>Weather Forecast & Crop Recommendation with Voice Support</p>
</div>
""", unsafe_allow_html=True)

# Language selection
st.markdown("### 🌐 Choose Your Language")
lang = st.radio("", ["English", "తెలుగు", "हिन्दी"], horizontal=True)
st.session_state['language'] = lang

# Weather Section
st.markdown("""
<div class="section-card">
    <h2>🌤️ Weather Forecast</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    city = voice_input_field("Enter City Name", "city_input", "text")

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🌤️ Get Weather", type="primary"):
        if city:
            with st.spinner("Fetching weather data..."):
                if lang == "తెలుగు":
                    report = get_weather_telugu(city)
                    speak(report, 'te', autoplay=True)
                elif lang == "हिन्दी":
                    report = get_weather_hindi(city)
                    speak(report, 'hi', autoplay=True)
                else:
                    report = get_weather(city)
                    speak(report, 'en', autoplay=True)
                
                st.markdown(f"""
                <div class="result-box">
                    <h4>Weather Information</h4>
                    <p>{report}</p>
                </div>
                """, unsafe_allow_html=True)

# Crop Recommendation Section
st.markdown("""
<div class="section-card">
    <h2>🌱 Crop Recommendation</h2>
</div>
""", unsafe_allow_html=True)

# Input fields in a grid layout
col1, col2, col3 = st.columns(3)

with col1:
    n = voice_input_field("Nitrogen (N)", "nitrogen", "number", 0, 200)
    temp = voice_input_field("Temperature (°C)", "temperature", "float", 10.0, 50.0, 0.1)
    ph = voice_input_field("pH Level", "ph", "float", 0.0, 14.0, 0.1)

with col2:
    p = voice_input_field("Phosphorus (P)", "phosphorus", "number", 0, 200)
    hum = voice_input_field("Humidity (%)", "humidity", "float", 10.0, 100.0, 0.1)

with col3:
    k = voice_input_field("Potassium (K)", "potassium", "number", 0, 200)
    rain = voice_input_field("Rainfall (mm)", "rainfall", "float", 0.0, 500.0, 0.1)

# Recommendation button
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    if st.button("🌾 Get Crop Recommendation", type="primary", use_container_width=True):
        # Validate inputs
        if all([n is not None, p is not None, k is not None, temp is not None, hum is not None, ph is not None, rain is not None]):
            with st.spinner("Analyzing soil conditions..."):
                features = [[n, p, k, temp, hum, ph, rain]]
                prediction = crop_model.predict(features)[0]
                crop = le.inverse_transform([prediction])[0]
                
                if lang == "తెలుగు":
                    output = f"సిఫారసు చేసిన పంట: {crop}"
                    detailed_msg = f"మీ మట్టి పరిస్థితుల ఆధారంగా, {crop} పంట అత్యుత్తమమైనది. ఈ పంట మీ ప్రాంతంలో బాగా పెరుగుతుంది."
                elif lang == "हिन्दी":
                    output = f"अनुशंसित फसल: {crop}"
                    detailed_msg = f"आपकी मिट्टी की स्थिति के आधार पर, {crop} फसल सबसे अच्छी है। यह फसल आपके क्षेत्र में अच्छी तरह से बढ़ेगी।"
                else:
                    output = f"Recommended Crop: {crop}"
                    detailed_msg = f"Based on your soil conditions, {crop} is the optimal choice. This crop will thrive in your region."
                
                st.markdown(f"""
                <div class="result-box">
                    <h3>🎯 Recommendation Result</h3>
                    <h4>{output}</h4>
                    <p>{detailed_msg}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Auto-play recommendation
                lang_code = 'te' if lang == "తెలుగు" else 'hi' if lang == "हिन्दी" else 'en'
                speak(f"{output}. {detailed_msg}", lang_code, autoplay=True)
        else:
            st.warning("Please fill in all the required fields.")

# Voice commands help
with st.expander("🎙️ Voice Commands Help"):
    st.markdown("""
    **How to use voice input:**
    - Click the 🎤 button next to any input field
    - Speak clearly into your microphone
    - For numbers, say them clearly (e.g., "twenty five" for 25)
    - For city names, speak the name clearly
    
    **Supported Languages:**
    - English: Full voice support
    - తెలుగు: Voice input and output
    - हिन्दी: Voice input and output
    
    **Tips:**
    - Ensure your microphone is working
    - Speak in a quiet environment
    - Allow microphone permissions when prompted
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🌿 AI Agri Voice Assistant - Helping farmers make informed decisions
</div>
""", unsafe_allow_html=True)
