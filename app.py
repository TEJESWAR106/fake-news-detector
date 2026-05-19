import streamlit as st
import joblib, re, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

model = joblib.load("model.pkl")
tfidf  = joblib.load("tfidf.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [ps.stem(w) for w in text.split()
              if w not in stop_words]
    return " ".join(tokens)

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detector")
st.write("Paste any news article and find out if it is Real or Fake.")
st.markdown("---")

article = st.text_area("Paste news article here", height=250,
    placeholder="Copy and paste a full news article...")

if st.button("Analyze"):
    if not article.strip():
        st.warning("Please paste some article text first.")
    else:
        vec  = tfidf.transform([clean_text(article)])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        st.markdown("---")
        if pred == 1:
            conf = round(prob[1] * 100, 1)
            st.success(f"REAL NEWS  —  Confidence: {conf}%")
            st.progress(int(conf))
        else:
            conf = round(prob[0] * 100, 1)
            st.error(f"FAKE NEWS  —  Confidence: {conf}%")
            st.progress(int(conf))

        st.caption(f"Model: Logistic Regression  |  "
                   f"Cleaned words: {len(clean_text(article).split())}")