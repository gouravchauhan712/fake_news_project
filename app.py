# =========================
# 1. IMPORT LIBRARIES
# =========================
import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# =========================
# 2. LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return pickle.load(open("model1.pkl", "rb"))

model = load_model()

# =========================
# 3. NLTK SETUP
# =========================
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# =========================
# 4. CLEAN FUNCTION (SAME AS TRAINING)
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# =========================
# 5. PREDICTION FUNCTION (FIXED)
# =========================
def predict_news(text):
    text = clean_text(text)
    
    prob = model.predict_proba([text])[0]
    classes = model.classes_
    
    prob_dict = dict(zip(classes, prob))
    
    prob_real = prob_dict[1]
    prob_fake = prob_dict[0]

    if prob_real >= 0.6:
        return f"✅ Real News (Confidence: {prob_real:.2f})"
    else:
        return f"🚨 Fake News (Confidence: {prob_fake:.2f})"

# =========================
# 6. UI DESIGN
# =========================
st.set_page_config(page_title="Fake News Detector 🇮🇳", layout="centered")

st.title("📰 Fake News Detection System")

st.write("Enter a news headline or statement to check if it's real or fake.")

# Input box
user_input = st.text_area("✍️ Enter News Text")

# Button
if st.button("🔍 Check News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        result = predict_news(user_input)
        
        if "Fake" in result:
            st.error(result)
        else:
            st.success(result)

# Footer
st.markdown("---")
st.caption("Developed using Machine Learning (TF-IDF + Logistic Regression)")