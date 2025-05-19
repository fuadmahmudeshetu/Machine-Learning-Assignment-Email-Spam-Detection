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

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📧 Email/SMS Spam Classifier")

user_input = st.text_area("Enter Message", "")

if st.button("Check"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        features = vectorizer.transform([user_input])

        prediction = model.predict(features)

        if prediction[0] == 1:
            st.success("✅ This is a **Ham Mail** (Not Spam).")
        else:
            st.error("🚫 This is a **Spam Mail**.")
