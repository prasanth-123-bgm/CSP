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
from audio_recorder_streamlit import audio_recorder  # ← Mic input
import re
import os
from dotenv import load_dotenv

load_dotenv()

# --- Background CSS ---
def set_bg(path):
    img = base64.b64encode(open(path,'rb').read()).decode()
    st.markdown(f"""<style>
        body {{ background: url("data:image/jpg;base64,{img}") center/cover fixed; }}
        .stApp {{ background: rgba(255,255,255,0.1); backdrop-filter: blur(12px); }}
        h1,h2,h3,label {{ color:#00fff2; text-shadow:0 0 4px #00c9a7; }}
        input,button {{ border-radius:12px; }}
        .result-card {{ background:rgba(0,0,0,0.6); border-left:6px solid #00e676;
                        border-radius:16px; padding:1rem; color:#e0f7fa; }}
    </style>""", unsafe_allow_html=True)

set_bg("csp background.jpg")
st.set_page_config("AgriVoice Pro", layout="wide", page_icon="🌱")

# --- Helpers ---
def tts(text, lang): tts = gTTS(text=text, lang=lang); fp=BytesIO(); tts.write_to_fp(fp); st.audio(fp.getvalue(), format="audio/mp3")
@st.cache_resource
def load_crop(): return joblib.load("crop_model (1).pkl"), joblib.load("label_encoder (1).pkl")
@st.cache_resource
def load_schemes():
    df = pd.read_csv("gov_schemes_dataset.csv")
    df['context'] = df[['Scheme Name','Description','Eligibility','Benefits']].astype(str).agg(' '.join, axis=1)
    m = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return df, m, m.encode(df['context'].tolist(), convert_to_tensor=True)
def detect_intent(text):
    for k, kws in {'weather':['weather','rain'],'crop':['crop','soil'],'pest':['pest','spray'],'scheme':['scheme','yojana']}.items():
        if any(re.search(fr'\b{kw}\b', text.lower()) for kw in kws): return k
    return None

# --- Layout ---
tabs = st.tabs(["🏠 Home","🌤️ Weather","🌾 Crop","🏛 Schemes","🐛 Pest"])
df_sch, sch_mdl, sch_emb = load_schemes()
crop_mdl, crop_le = load_crop()

# --- HOME ---
with tabs[0]:
    st.header("Welcome Farmer 👨‍🌾")
    lang = st.radio("Language", ["English","తెలుగు","हिन्दी"], horizontal=True)
    greeting = "Good Morning" if datetime.now().hour<12 else "Good Evening"
    st.subheader(f"{greeting}!")
    query = st.text_input("Ask a question:")
    audio = audio_recorder(text="Speak now", recording_color="#0f0", neutral_color="#888")
    if audio:
        query = audio  # record<string> from wav bytes
        st.success(f"You said: {query}")
    if query:
        intent = detect_intent(GoogleTranslator(source='auto', target='en').translate(query))
        st.info(f"Detected intent: {intent}")
        if intent: tabs[['weather','crop','scheme','pest'].index(intent)+1].select()

# --- WEATHER ---
with tabs[1]:
    st.subheader("Weather Forecast")
    city = st.text_input("City name")
    if st.button("Get Weather"):
        if city:
            rep = (get_weather_telugu if lang=="తెలుగు" else get_weather_hindi if lang=="हिन्दी" else get_weather)(city)
            st.markdown(f"<div class='result-card'>{rep}</div>", unsafe_allow_html=True)
            tts(rep, {'English':'en','తెలుగు':'te','हिन्दी':'hi'}[lang])

# --- CROP ---
with tabs[2]:
    st.subheader("Crop Recommendation")
    col1, col2 = st.columns(2)
    n,p,k = col1.number_input("N",0,200),col1.number_input("P",0,200),col1.number_input("K",0,200)
    temp,hum,ph,rain = col2.number_input("Temp"),col2.number_input("Hum"),col2.number_input("pH"),col2.number_input("Rain")
    if st.button("Recommend"):
        pred = crop_mdl.predict([[n,p,k,temp,hum,ph,rain]])[0]
        crop = crop_le.inverse_transform([pred])[0]
        msg = {"English":f"Recommended crop: {crop}","తెలుగు":f"సిఫారసు: {crop}","हिन्दी":f"अनुशंसित: {crop}"}[lang]
        st.success(msg); tts(msg, {'English':'en','తెలుగు':'te','हिन्दी':'hi'}[lang])

# --- SCHEMES ---
with tabs[3]:
    st.subheader("Government Schemes")
    q = st.text_input("Ask about schemes")
    if st.button("Get Scheme Info"):
        qe = GoogleTranslator(source='auto',target='en').translate(q)
        emb = sch_mdl.encode(qe, convert_to_tensor=True)
        idx = torch.argmax(util.pytorch_cos_sim(emb, sch_emb))
        ans = df_sch.loc[idx,'Description']
        tr = GoogleTranslator(source='en', target={'English':'en','తెలుగు':'te','हिन्दी':'hi'}[lang]).translate(ans)
        st.markdown(f"<div class='result-card'>{tr}</div>", unsafe_allow_html=True)
        tts(tr, {'English':'en','తెలుగు':'te','हिन्दी':'hi'}[lang])

# --- PEST ---
with tabs[4]:
    st.subheader("Pest Management")
    crop_i = st.text_input("Crop name")
    area = st.number_input("Area (ha)",0.1)
    if st.button("Get Pest Plan"):
        dfp = pd.read_csv("pest_db.csv")
        sel = dfp[dfp.Crop.str.lower()==crop_i.lower()]
        if sel.empty: st.error("No data")
        else:
            for _,r in sel.iterrows():
                msg = (f"For {r.Crop} affected by {r.Pest_Disease}, use {r.Pesticide}. "
                       f"Dose: {r.Dose_per_ha*area} {r.Unit}. Note: {r.Notes}")
                tr = GoogleTranslator(source='en',target={'English':'en','తెలుగు':'te','हिन्दी':'hi'}[lang]).translate(msg)
                st.success(tr); tts(tr, {'English':'en','తెలుగు':'te','हिन्दी':'hi'}[lang])

