import numpy as np
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.metrics import classification_report

import nltk
from nltk.corpus import stopwords
from collections import Counter

# Load vectorizer and model from disk
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit app UI
st.title("📧 Email/SMS Spam Classifier")

user_input = st.text_area("Enter Message", "")

if st.button("Check"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        # Convert user input to feature vector
        features = vectorizer.transform([user_input])

        # Predict using the loaded model
        prediction = model.predict(features)

        # Display result
        if prediction[0] == 1:
            st.success("✅ This is a **Ham Mail** (Not Spam).")
        else:
            st.error("🚫 This is a **Spam Mail**.")

