import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
import os
import io

# 1. Hugging Face'ten modeli güvenli bir şekilde indirip yükle
@st.cache_resource
def get_model():
    # Buradaki repo_id senin Hugging Face kullanıcı adın ve model repo isminle aynı olmalı
    # Örn: 'kayasakir/Lift-Echo-Model'
    model_path = hf_hub_download(repo_id="kayasakir/Lift-Echo-Model", filename="model.keras")
    return tf.keras.models.load_model(model_path, compile=False)

# Modeli belleğe yükle
model = get_model()

# 2. Veri İşleme Fonksiyonu
def read_bearing_data(file_content):
    df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), sep='\t', header=None)
    return df.values.flatten()

# 3. Streamlit Arayüzü
st.set_page_config(page_title="Lift-Echo AI", page_icon="🛗")
st.title("🛗 Lift-Echo AI: Endüstriyel Denetim Paneli")
st.markdown("---")

uploaded_file = st.file_uploader("Vibrasyon verisi (.csv) yükle", type=['csv'])

if uploaded_file:
    file_bytes = uploaded_file.read()
    data = read_bearing_data(file_bytes).reshape(1, -1)
    
    # 4. Tahmin ve Anomali Analizi
    prediction = model.predict(data, verbose=0)
    error = np.mean(np.square(data - prediction))
    
    st.write(f"### Analiz Sonucu")
    st.write(f"Anomali Skoru (MSE): **{error:.6f}**")
    
    threshold = 0.005 
    
    if error > threshold:
        st.error("KRİTİK: Anomali Tespit Edildi! Derhal bakım planlayın.")
    else:
        st.success("Sistem Sağlıklı: Asansör normal çalışma parametrelerinde.")