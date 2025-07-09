import streamlit as st
import pandas as pd
import pickle
import joblib
from gtts import gTTS
from io import BytesIO
import speech_recognition as sr
import numpy as np
import re
import time
from weather import get_weather, get_weather_telugu, get_weather_hindi

# Custom CSS for modern UI with green and white theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #4CAF50 0%, #81C784 50%, #A5D6A7 100%);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 12px 40px rgba(76, 175, 80, 0.3);
        animation: fadeIn 1s ease-in;
    }
    
    .welcome-card {
        background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 2px solid #4CAF50;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.2);
        animation: slideUp 0.8s ease-out;
    }
    
    .section-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FFF8 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border-left: 5px solid #4CAF50;
        transition: all 0.3s ease;
    }
    
    .section-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(76, 175, 80, 0.2);
    }
    
    .input-group {
        background: linear-gradient(135deg, #F1F8E9 0%, #E8F5E8 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #A5D6A7;
    }
    
    .voice-button {
        background: linear-gradient(45deg, #4CAF50, #66BB6A);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        cursor: pointer;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    .voice-button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
        background: linear-gradient(45deg, #66BB6A, #4CAF50);
    }
    
    .result-box {
        background: linear-gradient(135deg, #C8E6C9 0%, #DCEDC8 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border-left: 5px solid #2E7D32;
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.2);
        animation: resultSlide 0.6s ease-out;
    }
    
    .listening-indicator {
        background: linear-gradient(45deg, #FF5722, #FF7043);
        color: white;
        padding: 1rem 2rem;
        border-radius: 25px;
        text-align: center;
        animation: pulse 1.5s infinite;
        margin: 1rem 0;
    }
    
    .query-input {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FFF8 100%);
        border: 2px solid #4CAF50;
        border-radius: 15px;
        padding: 1rem;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .query-input:focus {
        box-shadow: 0 0 20px rgba(76, 175, 80, 0.3);
        outline: none;
    }
    
    .language-selector {
        background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #A5D6A7;
    }
    
    .step-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FFF8 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes resultSlide {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.02); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #66BB6A);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page_state' not in st.session_state:
    st.session_state.page_state = 'welcome'
if 'language' not in st.session_state:
    st.session_state.language = 'English'
if 'user_query' not in st.session_state:
    st.session_state.user_query = ''
if 'query_type' not in st.session_state:
    st.session_state.query_type = None

# Safe model loading
@st.cache_resource
def load_models():
    try:
        crop_model = joblib.load("crop_model (1).pkl")
        le = joblib.load("label_encoder (1).pkl")
        return crop_model, le
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.info("Please ensure 'crop_model (1).pkl' and 'label_encoder (1).pkl' are in the same directory as this script.")
        st.stop()

# Speech recognition setup
recognizer = sr.Recognizer()

def speak(text, lang='en', autoplay=True):
    """Enhanced speak function with autoplay option"""
    try:
        tts = gTTS(text=text, lang=lang)
        fp = BytesIO()
        tts.write_to_fp(fp)
        
        if autoplay:
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
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            
        # Convert speech to text
        text = recognizer.recognize_google(audio, language=lang_code)
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
                lang_code = 'te-IN' if st.session_state.language == 'తెలుగు' else 'hi-IN' if st.session_state.language == 'हिन्दी' else 'en-US'
                speech_text = listen_to_speech(lang_code)
                
                if speech_text:
                    st.success(f"Heard: {speech_text}")
                    
                    # Process the speech input based on type
                    if input_type in ["number", "float"]:
                        try:
                            # Extract numbers from speech
                            numbers = re.findall(r'\d+\.?\d*', speech_text)
                            if numbers:
                                value = float(numbers[0]) if input_type == "float" else int(numbers[0])
                                st.session_state[key] = value
                                st.rerun()
                        except:
                            st.warning("Could not extract number from speech")
                    else:
                        st.session_state[key] = speech_text
                        st.rerun()
    
    return value

def analyze_query(query):
    """Analyze user query to determine intent"""
    query_lower = query.lower()
    
    weather_keywords = ['weather', 'temperature', 'rain', 'climate', 'forecast', 'వాతావరణం', 'వర్షం', 'మौसम', 'बारिश', 'जलवायु']
    crop_keywords = ['crop', 'farming', 'agriculture', 'plant', 'harvest', 'recommendation', 'పంట', 'వ్యవసాయం', 'फसल', 'कृषि', 'खेती']
    
    if any(keyword in query_lower for keyword in weather_keywords):
        return 'weather'
    elif any(keyword in query_lower for keyword in crop_keywords):
        return 'crop'
    else:
        return 'general'

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
    <p>Your Smart Farming Companion with Voice Support</p>
</div>
""", unsafe_allow_html=True)

# Language selection
st.markdown("""
<div class="language-selector">
    <h3>🌐 Choose Your Language / भाषा चुनें / భాష ఎంచుకోండి</h3>
</div>
""", unsafe_allow_html=True)

lang = st.radio("", ["English", "తెలుగు", "हिन्दी"], horizontal=True, key="language_selector")
st.session_state.language = lang

# Welcome section
if st.session_state.page_state == 'welcome':
    # Welcome message based on language
    if lang == "తెలుగు":
        welcome_msg = """
        ## 🙏 స్వాగతం రైతు మిత్రులు!
        
        **మీ స్మార్ట్ వ్యవసాయ సహాయకుడికి స్వాగతం**
        
        ఈ అప్లికేషన్ మీకు అందిస్తుంది:
        - 🌤️ వాతావరణ సమాచారం
        - 🌱 పంట సిఫార్సులు
        - 🎤 వాయిస్ సపోర్ట్
        
        **మీకు ఏమి అవసరం? దయచేసి టైప్ చేయండి లేదా మాట్లాడండి...**
        """
        placeholder_text = "ఉదాహరణ: వాతావరణం గురించి తెలుసుకోవాలి లేదా పంట సిఫార్సు కావాలి"
    elif lang == "हिन्दी":
        welcome_msg = """
        ## 🙏 स्वागत है किसान मित्रों!
        
        **आपके स्मार्ट कृषि सहायक में आपका स्वागत है**
        
        यह एप्लिकेशन आपको प्रदान करता है:
        - 🌤️ मौसम की जानकारी
        - 🌱 फसल की सिफारिशें
        - 🎤 वॉइस सपोर्ट
        
        **आपको क्या चाहिए? कृपया टाइप करें या बोलें...**
        """
        placeholder_text = "उदाहरण: मौसम के बारे में जानना चाहता हूं या फसल की सिफारिश चाहिए"
    else:
        welcome_msg = """
        ## 🙏 Welcome Farmer Friends!
        
        **Welcome to Your Smart Agriculture Assistant**
        
        This application provides you with:
        - 🌤️ Weather Information
        - 🌱 Crop Recommendations
        - 🎤 Voice Support
        
        **What do you need help with? Please type or speak...**
        """
        placeholder_text = "Example: I want to know about weather or need crop recommendation"
    
    st.markdown(f"""
    <div class="welcome-card">
        {welcome_msg}
    </div>
    """, unsafe_allow_html=True)
    
    # Query input section
    st.markdown("""
    <div class="input-group">
        <h4>🎯 Tell me what you need:</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_query = st.text_input("", placeholder=placeholder_text, key="main_query")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎤", key="voice_main_query", help="Click to speak"):
            with st.spinner("Listening..."):
                lang_code = 'te-IN' if lang == 'తెలుగు' else 'hi-IN' if lang == 'हिन्दी' else 'en-US'
                speech_text = listen_to_speech(lang_code)
                
                if speech_text:
                    st.success(f"Heard: {speech_text}")
                    st.session_state.user_query = speech_text
                    st.session_state.query_type = analyze_query(speech_text)
                    st.session_state.page_state = 'processing'
                    st.rerun()
    
    # Process text input
    if user_query:
        st.session_state.user_query = user_query
        st.session_state.query_type = analyze_query(user_query)
        st.session_state.page_state = 'processing'
        st.rerun()
    
    # Quick action buttons
    st.markdown("### 🚀 Quick Actions:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌤️ Weather Info", type="primary", use_container_width=True):
            st.session_state.query_type = 'weather'
            st.session_state.page_state = 'processing'
            st.rerun()
    
    with col2:
        if st.button("🌱 Crop Recommendation", type="primary", use_container_width=True):
            st.session_state.query_type = 'crop'
            st.session_state.page_state = 'processing'
            st.rerun()
    
    with col3:
        if st.button("🏠 Home", type="secondary", use_container_width=True):
            st.session_state.page_state = 'welcome'
            st.rerun()

# Processing section
elif st.session_state.page_state == 'processing':
    # Back button
    if st.button("🔙 Back to Home", key="back_home"):
        st.session_state.page_state = 'welcome'
        st.rerun()
    
    # Weather section
    if st.session_state.query_type == 'weather':
        st.markdown("""
        <div class="section-card">
            <h2>🌤️ Weather Forecast</h2>
            <p>Get accurate weather information for your location</p>
        </div>
        """, unsafe_allow_html=True)
        
        # City input
        if lang == "తెలుగు":
            city_label = "నగరం లేదా ప్రాంతం పేరు"
            get_weather_text = "వాతావరణ సమాచారం పొందండి"
        elif lang == "हिन्दी":
            city_label = "शहर या क्षेत्र का नाम"
            get_weather_text = "मौसम की जानकारी प्राप्त करें"
        else:
            city_label = "City or Area Name"
            get_weather_text = "Get Weather Information"
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            city = voice_input_field(city_label, "city_input", "text")
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"🌤️ {get_weather_text}", type="primary"):
                if city:
                    with st.spinner("Fetching weather data..."):
                        try:
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
                                <h4>🌤️ Weather Information for {city}</h4>
                                <p>{report}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error fetching weather data: {e}")
                            st.info("Please check the city name and try again.")
    
    # Crop recommendation section
    elif st.session_state.query_type == 'crop':
        crop_model, le = load_models()
        
        st.markdown("""
        <div class="section-card">
            <h2>🌱 Crop Recommendation</h2>
            <p>Get personalized crop recommendations based on soil conditions</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Labels based on language
        if lang == "తెలుగు":
            labels = {
                'nitrogen': 'నైట్రోజన్ (N)',
                'phosphorus': 'ఫాస్ఫరస్ (P)',
                'potassium': 'పొటాషియం (K)',
                'temperature': 'ఉష్ణోగ్రత (°C)',
                'humidity': 'తేమ (%)',
                'ph': 'pH స్థాయి',
                'rainfall': 'వర్షపాతం (mm)',
                'get_recommendation': 'పంట సిఫార్సు పొందండి'
            }
        elif lang == "हिन्दी":
            labels = {
                'nitrogen': 'नाइट्रोजन (N)',
                'phosphorus': 'फॉस्फोरस (P)',
                'potassium': 'पोटेशियम (K)',
                'temperature': 'तापमान (°C)',
                'humidity': 'आर्द्रता (%)',
                'ph': 'pH स्तर',
                'rainfall': 'वर्षा (mm)',
                'get_recommendation': 'फसल की सिफारिश प्राप्त करें'
            }
        else:
            labels = {
                'nitrogen': 'Nitrogen (N)',
                'phosphorus': 'Phosphorus (P)',
                'potassium': 'Potassium (K)',
                'temperature': 'Temperature (°C)',
                'humidity': 'Humidity (%)',
                'ph': 'pH Level',
                'rainfall': 'Rainfall (mm)',
                'get_recommendation': 'Get Crop Recommendation'
            }
        
        # Input fields in a grid layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            n = voice_input_field(labels['nitrogen'], "nitrogen", "number", 0, 200)
            temp = voice_input_field(labels['temperature'], "temperature", "float", 10.0, 50.0, 0.1)
            ph = voice_input_field(labels['ph'], "ph", "float", 0.0, 14.0, 0.1)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            p = voice_input_field(labels['phosphorus'], "phosphorus", "number", 0, 200)
            hum = voice_input_field(labels['humidity'], "humidity", "float", 10.0, 100.0, 0.1)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            k = voice_input_field(labels['potassium'], "potassium", "number", 0, 200)
            rain = voice_input_field(labels['rainfall'], "rainfall", "float", 0.0, 500.0, 0.1)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendation button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button(f"🌾 {labels['get_recommendation']}", type="primary", use_container_width=True):
                # Validate inputs
                if all([n is not None, p is not None, k is not None, temp is not None, hum is not None, ph is not None, rain is not None]):
                    with st.spinner("Analyzing soil conditions..."):
                        try:
                            features = [[n, p, k, temp, hum, ph, rain]]
                            prediction = crop_model.predict(features)[0]
                            crop = le.inverse_transform([prediction])[0]
                            
                            if lang == "తెలుగు":
                                output = f"సిఫార్సు చేసిన పంట: {crop}"
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
                        except Exception as e:
                            st.error(f"Error generating recommendation: {e}")
                else:
                    if lang == "తెలుగు":
                        st.warning("దయచేసి అన్ని అవసరమైన ఫీల్డ్‌లను పూరించండి.")
                    elif lang == "हिन्दी":
                        st.warning("कृपया सभी आवश्यक फ़ील्ड भरें।")
                    else:
                        st.warning("Please fill in all the required fields.")
    
    # General query handling
    else:
        st.markdown("""
        <div class="section-card">
            <h2>🤖 General Query</h2>
            <p>I can help you with weather information and crop recommendations!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if lang == "తెలుగు":
            response = """
            నేను మీకు సహాయం చేయగలను:
            - 🌤️ వాతావరణ సమాచారం కోసం
            - 🌱 పంట సిఫార్సుల కోసం
            
            దయచేసి మీ ప్రశ్నను మరింత స్పష్టంగా చెప్పండి.
            """
        elif lang == "हिन्दी":
            response = """
            मैं आपकी मदद कर सकता हूं:
            - 🌤️ मौसम की जानकारी के लिए
            - 🌱 फसल की सिफारिशों के लिए
            
            कृपया अपना प्रश्न और स्पष्ट रूप से बताएं।
            """
        else:
            response = """
            I can help you with:
            - 🌤️ Weather information
            - 🌱 Crop recommendations
            
            Please specify your query more clearly.
            """
        
        st.markdown(f"""
        <div class="result-box">
            <h4>How can I help you?</h4>
            <p>{response}</p>
        </div>
        """, unsafe_allow_html=True)

# Voice commands help
with st.expander("🎙️ Voice Commands Help"):
    if lang == "తెలుగు":
        help_text = """
        **వాయిస్ ఇన్‌పుట్ ఎలా ఉపయోగించాలి:**
        - ఏదైనా ఇన్‌పుట్ ఫీల్డ్ పక్కన ఉన్న 🎤 బటన్‌ను క్లిక్ చేయండి
        - మైక్రోఫోన్‌లో స్పష్టంగా మాట్లాడండి
        - సంఖ్యల కోసం, వాటిని స్పష్టంగా చెప్పండి (ఉదా: "ఇరవై ఐదు" 25 కోసం)
        - నగర పేర్లను స్పష్టంగా చెప్పండి
        
        **సపోర్ట్ చేయబడిన భాషలు:**
        - English: పూర్తి వాయిస్ సపోర్ట్
        - తెలుగు: వాయిస్ ఇన్‌పుట్ మరియు అవుట్‌పుట్
        - हिन्दी: వాయిస్ ఇన్‌పుట్ మరియు అవుట్‌పుట్
        
        **చిట్కాలు:**
        - మీ మైక్రోఫోన్ పని చేస్తుందని నిర్ధారించుకోండి
        - నిశ్శబ్దమైన వాతావరణంలో మాట్లాడండి
        - ప్రాంప్ట్ చేసినప్పుడు మైక్రోఫోన్ అనుమతులను అనుమతించండి
        """
    elif lang == "हिन्दी":
        help_text = """
        **वॉइस इनपुट का उपयोग कैसे करें:**
        - किसी भी इनपुट फील्ड के बगल में 🎤 बटन पर क्लिक करें
        - अपने माइक्रोफोन में स्पष्ट रूप से बोलें
        - संख्याओं के लिए, उन्हें स्पष्ट रूप से बोलें (जैसे: "पच्चीस" 25 के लिए)
        - शहर के नाम स्पष्ट रूप से बोलें
        
        **समर्थित भाषाएं:**
        - English: पूर्ण वॉइस सपोर्ट
        - తెలుగు: वॉइस इनपुट और आउटपुट
        - हिन्दी: वॉइस इनपुट और आउटपुट
        
        **सुझाव:**
        - सुनिश्चित करें कि आपका माइक्रोफोन काम कर रहा है
        - शांत वातावरण में बोलें
        - प्रॉम्प्ट होने पर माइक्रोफोन अनुमतियां दें
        """
    else:
        help_text = """
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
        """
    
    st.markdown(help_text)

# Footer
st.markdown("---")
if lang == "తెలుగు":
    footer_text = "🌿 AI అగ్రి వాయిస్ అసిస్టెంట్ - రైతులకు తెలివైన నిర్ణయాలు తీసుకోవడంలో సహాయపడుతుంది"
elif lang == "हिन्दी":
    footer_text = "🌿 AI एग्री वॉइस असिस्टेंट - किसानों को सूचित निर्णय लेने में मदद करता है"
else:
    footer_text = "🌿 AI Agri Voice Assistant - Helping farmers make informed decisions"

st.markdown(f"""
<div style="text-align: center; color: #4CAF50; padding: 2rem; font-weight: bold;">
    {footer_text}
</div>
""", unsafe_allow_html=True)
