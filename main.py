
import pandas as pd
import re
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent

# Preprocess text (must remain identical in app.py)
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Load first dataset
data1 = pd.read_csv(BASE_DIR / "spam.csv", encoding="latin-1")
data1 = data1[["v1", "v2"]]
data1.columns = ["label", "message"]

# Load second dataset
data2 = pd.read_csv(BASE_DIR / "sample_sms_messages.csv")

# Combine datasets
data = pd.concat([data1, data2], ignore_index=True)

# Normalize labels
label_mapping = {
    "ham": 0,
    "spam": 1,
    "0": 0,
    "1": 1
}

data["label"] = (
    data["label"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(label_mapping)
)

# Remove invalid or missing rows
data = data.dropna(subset=["label", "message"]).copy()
data["label"] = data["label"].astype(int)
data["message"] = data["message"].astype(str)

# Remove duplicate messages
data["clean_message"] = data["message"].apply(preprocess_text)
data = data.drop_duplicates(subset=["clean_message"])

# Check dataset
print("Dataset distribution:")
print(data["label"].value_counts())

if data["label"].nunique() != 2:
    raise ValueError("Dataset must contain both spam and ham messages.")

# Preprocess messages
X = data["clean_message"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=2
)
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, BASE_DIR / "spam_classifier_model.pkl")
joblib.dump(tfidf, BASE_DIR / "tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")
