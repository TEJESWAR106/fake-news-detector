# 📰 Fake News Detector

An NLP-based machine learning web application that classifies news articles as **Real** or **Fake** with confidence scores — built with Python, Scikit-learn, and Streamlit.

---

## 🚀 Live Demo

👉 [Click here to try the app](https://fake-news-detector-iwpyqqjgzjzbsmap2mzvay.streamlit.app/)


---

## 📌 Project Overview

This project was built as part of the **Elevate Labs AI/ML Internship** (2-week project phase).

The app takes any news article as input, preprocesses the text using NLP techniques, and predicts whether the article is real or fake using a trained Logistic Regression model.

---

## 🧠 How It Works

```
Raw Article Text
      ↓
Text Cleaning (lowercase, remove URLs, punctuation)
      ↓
Stopword Removal + Stemming (NLTK)
      ↓
TF-IDF Vectorization (5000 features, bigrams)
      ↓
Logistic Regression Model
      ↓
Prediction → REAL or FAKE + Confidence %
```

---

## 📂 Project Structure

```
fake-news-detector/
│
├── dataset/
│   ├── Fake.csv          # Fake news articles (Kaggle - ISOT dataset)
│   └── True.csv          # Real news articles (Kaggle - ISOT dataset)
│
├── app.py                # Streamlit web application
├── train.py              # Model training script
├── model.pkl             # Trained Logistic Regression model
├── tfidf.pkl             # Fitted TF-IDF vectorizer
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| NLP | NLTK, TF-IDF Vectorizer |
| ML Model | Scikit-learn (Logistic Regression) |
| Web App | Streamlit |
| Data | Pandas, NumPy |
| Deployment | Streamlit Community Cloud |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~99% |
| Precision | 0.99 |
| Recall | 0.99 |
| F1-Score | 0.99 |

Trained and evaluated on the **ISOT Fake News Dataset** (~44,000 articles).

---

## ⚙️ Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/TEJESWAR106/fake-news-detector.git
cd fake-news-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (generates model.pkl and tfidf.pkl)
```bash
python train.py
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Dataset

- **Name:** ISOT Fake News Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Size:** ~44,000 articles (23,502 fake + 21,417 real)
- **Features used:** Article `text` column + `label` (0 = Fake, 1 = Real)

---

## 🔍 Steps Involved

1. **Data Collection** — Downloaded ISOT dataset from Kaggle
2. **Preprocessing** — Lowercasing, URL removal, punctuation removal, stopword filtering, stemming
3. **Feature Extraction** — TF-IDF vectorization with unigrams and bigrams
4. **Model Training** — Logistic Regression with 80/20 train-test split
5. **Evaluation** — Classification report, confusion matrix
6. **Deployment** — Streamlit web app hosted on Streamlit Cloud


---

## 👨‍💻 Author

**Tejeswar A**
AI/ML Intern — Elevate Labs
- GitHub: [@TEJESWAR106](https://github.com/TEJESWAR106)

---

## 📄 License

This project is for educational purposes as part of an internship program.
