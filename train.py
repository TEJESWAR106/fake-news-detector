import pandas as pd
import re, joblib, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [ps.stem(w) for w in text.split()
              if w not in stop_words]
    return " ".join(tokens)

# --- Load your dataset ---
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

fake["label"] = 0   # 0 = Fake
real["label"] = 1   # 1 = Real

df = pd.concat([fake, real], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset loaded:", df.shape)
print(df["label"].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Clean text
df["clean"] = df["text"].apply(clean_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)

# Vectorize
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save both files
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
print("model.pkl and tfidf.pkl saved!")