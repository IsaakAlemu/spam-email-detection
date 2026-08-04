"""
Standalone script: rebuilds the TF-IDF features from data/spam.csv and
trains a CALIBRATED Linear SVM (so predict_proba works for the Streamlit
confidence UI), then saves both to model/.

Run from the project root:
    python calibrate_model.py

Requires: data/spam.csv to exist (the SMS Spam Collection dataset).
"""

import os
import re
import string

import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# ----------------------------------------------------------------------
# 1. Load and clean the dataset (same pipeline as the notebook)
# ----------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv("data/spam.csv", encoding="latin-1")
df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
df = df.drop_duplicates()

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    words = [w for w in text.split() if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


print("Cleaning text...")
df["clean_message"] = df["message"].apply(clean_text)

# ----------------------------------------------------------------------
# 2. TF-IDF features
# ----------------------------------------------------------------------
print("Building TF-IDF features...")
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["clean_message"])
y = LabelEncoder().fit_transform(df["label"])  # ham=0, spam=1

# ----------------------------------------------------------------------
# 3. Train the calibrated Linear SVM (this is what gives predict_proba)
# ----------------------------------------------------------------------
print("Training calibrated Linear SVM (this may take a moment)...")
base_model = LinearSVC(class_weight="balanced")
calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
calibrated_model.fit(X, y)

# ----------------------------------------------------------------------
# 4. Save the model + vectorizer for the Streamlit app
# ----------------------------------------------------------------------
os.makedirs("model", exist_ok=True)
joblib.dump(calibrated_model, "model/spam_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

print("\nDone.")
print("Saved: model/spam_model.pkl")
print("Saved: model/tfidf_vectorizer.pkl")
print("This model now supports predict_proba() for the confidence UI.")