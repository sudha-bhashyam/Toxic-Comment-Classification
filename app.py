import streamlit as st
import joblib
import sys
import os

# Fix path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import clean_text

# Load model and vectorizer
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# Page config
st.set_page_config(page_title="Toxic Comment Classifier")

st.title(" Toxic Comment Classifier")
st.write("Enter a comment to check if it's toxic or not.")

# Input box
user_input = st.text_area("Enter your comment:")

# Button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("⚠️ Toxic Comment")
        else:
            st.success("✅ Not Toxic")