"""
train.py
--------
Trains the fake-news detector from scratch and saves the model + vectorizer.
Run: python train.py
"""

import pickle
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from nlp_utils import clean_text

DATASET_DIR = "../dataset"
MODEL_PATH = "best_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

print("Loading dataset...")
true_df = pd.read_csv(f"{DATASET_DIR}/True.csv")
fake_df = pd.read_csv(f"{DATASET_DIR}/Fake.csv")

true_df["label"] = 0  # Real
fake_df["label"] = 1  # Fake

df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Total articles: {len(df)}  (Real: {(df.label==0).sum()}, Fake: {(df.label==1).sum()})")

print("Cleaning text (a few minutes)...")
t0 = time.time()
df["clean_text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)
print(f"Done in {time.time()-t0:.1f}s")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

print("Vectorizing (TF-IDF)...")
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 1), min_df=8, max_df=0.6)
X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)

print("\nTraining models...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=0.05),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
}

results = {}
for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = (model, acc)
    print(f"  {name}: test accuracy = {acc:.4f}  (trained in {time.time()-t0:.1f}s)")

best_name = max(results, key=lambda k: results[k][1])
best_model, best_acc = results[best_name]

lr_model, lr_acc = results["Logistic Regression"]
if best_name != "Logistic Regression" and (best_acc - lr_acc) < 0.01:
    best_name, best_model, best_acc = "Logistic Regression", lr_model, lr_acc
    print("(Switched to Logistic Regression: near-equal accuracy, better explainability)")

print(f"\nBest model: {best_name} ({best_acc:.4f} test accuracy)")
print(f"Train accuracy: {best_model.score(X_train, y_train):.4f}")

print("\nClassification report:")
print(classification_report(y_test, best_model.predict(X_test), target_names=["Real", "Fake"]))

with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)
with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(tfidf, f)

print(f"\nSaved {MODEL_PATH} and {VECTORIZER_PATH}")

sample = "The government announced a new education policy to improve higher education access for students across the country this year."
sv = tfidf.transform([clean_text(sample)])
pred = best_model.predict(sv)[0]
proba = best_model.predict_proba(sv)[0]
print(f"\nSanity check: {'Fake' if pred == 1 else 'Real'} ({proba.max()*100:.1f}% confidence)")
print("(Close to 50% is healthy — this sentence has no strong signal either way)")