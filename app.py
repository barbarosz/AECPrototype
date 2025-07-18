 
import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("InboxAI: Email Categorizer")
text = st.text_area("Paste your email text:")

if st.button("Categorize"):
    vec = vectorizer.transform([text])
    label = model.predict(vec)[0]
    st.write("Predicted Category:", label)
