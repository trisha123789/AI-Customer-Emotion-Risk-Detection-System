from tensorflow.keras.models import load_model
import pickle
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences



with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl","rb") as f:
    le = pickle.load(f)

model = load_model('emotion_model.h5')
st.set_page_config("AI Emotion Detector", "🧠")
st.title("🧠 AI Customer Emotion & Risk Detector")
st.write("LSTM-based NLP system for real-world support analysis")
user_text = st.text_area("Enter custom message:")
if st.button("analyze"):
    seq = tokenizer.texts_to_sequences([user_text])
    padded = pad_sequences(seq,maxlen=30)
    pred = model.predict(padded)
    ind = np.argmax(pred,axis =1)
    emotion = le.inverse_transform(ind)[0]
    risk = "High 🚨" if emotion=="angry" else "Medium ⚠️" if emotion=="frustrated" else "Low ✅"
    st.success(f"Emotion: {emotion}")
    st.warning(f"Risk Level: {risk}")

