import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load save model
model_fraud = pickle.load(open('/content/drive/MyDrive/NLP-SMS/dataset/model_fraud.sav','rb'))

tfidf = TfidfVectorizer
loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("/content/drive/MyDrive/NLP-SMS/dataset/new_selected_features_tf-idf.sav", "rb"))))

# Judul halaman web
st.title("Prediksi SMS Penipuan")

clean_teks = st.text_input("Masukkan Text SMS")

fraud_detection = ' '

if st.button("Hasil Prediksi"):
  predict_fraud = model_fraud.predict(loaded_vec.fit_transform([clean_teks]))

  if (predict_fraud == 0):
    fraud_detection = "SMS Normal"
  elif (predict_fraud == 1):
    fraud_detection = "SMS Penipuan"
  else:
    fraud_detection = "SMS Promo"
st.success(fraud_detection)