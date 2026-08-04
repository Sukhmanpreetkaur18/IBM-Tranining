# 📰 The Verification Desk — Fake News Detector & Generator

An NLP + Machine Learning + Generative AI project that detects whether a news
article is real or fake, explains *why* it made that call, and can generate
sample news-style text using a local AI model.

## Features

- **Detect** — classifies text as Real or Fake using TF-IDF + Logistic
  Regression (~94% test accuracy).
- **Explain** — highlights which words in the input pushed the verdict
  toward Real or Fake, so the result isn't a black box.
- **Generate** — uses a small local Generative AI model (`distilgpt2`) to
  draft sample news-style text from a headline. Free, no API key, runs
  entirely on your machine.
- A case-file/evidence-desk themed UI — verdicts are stamped onto the
  result like a rubber stamp on a document (VERIFIED / FLAGGED).

## Tech stack

| Layer          | Tech |
|-----------------|------|
| NLP / cleaning   | Python, `re`, scikit-learn stopwords |
| Model            | TF-IDF vectorizer + Logistic Regression (scikit-learn) |
| Generative AI    | Hugging Face `transformers`, `distilgpt2` |
| Backend API      | FastAPI, Uvicorn |
| Frontend         | HTML, CSS, vanilla JavaScript |

## Project structure

fake-news-project/
├── dataset/
│   ├── Fake.csv
│   └── True.csv
├── notebook/
│   └── Fake_News_Detection.ipynb   ← full training walkthrough with EDA
├── backend/
│   ├── nlp_utils.py                ← shared text-cleaning logic
│   ├── explain_utils.py            ← "why" behind each prediction
│   ├── generate_utils.py           ← local GenAI text generation
│   ├── train.py                    ← trains and saves the model
│   ├── app.py                      ← FastAPI server
│   ├── requirements.txt
│   ├── best_model.pkl              ← trained model (ready to use)
│   └── tfidf_vectorizer.pkl        ← trained vectorizer (ready to use)
└── frontend/
    ├── index.html
    ├── styles.css
    └── script.js

## Setup & run

### 1. Install dependencies

    cd backend
    pip install -r requirements.txt

### 2. Train the model (optional — pre-trained files are already included)

    python train.py

### 3. Start the API

    uvicorn app:app --reload

Leave this running — it serves on `http://localhost:8000`.

### 4. Open the app

Open `frontend/index.html` in your browser.

> The first time you use the **Generate** tab, it downloads the `distilgpt2`
> model (~350MB) — needs internet once, then works offline.

## Why the accuracy is ~94%, not 99%+

The raw Kaggle/ISOT Fake/Real news dataset has a well-known flaw: almost all
REAL articles carry a `(Reuters)` wire-service tag and curly apostrophes,
while FAKE articles carry site-branding text (`21st Century Wire`,
`Featured Image`) and have contractions split by plain spaces. A model that
doesn't account for this hits 99%+ accuracy by detecting **formatting**, not
content — and then fails on real-world typed text.

`nlp_utils.py` strips these dataset-specific artifacts out before training,
so the model is forced to learn from actual content. `train.py` includes a
sanity check confirming the model responds close to 50/50 on a neutral,
hand-typed sentence rather than confidently (and wrongly) calling everything
fake.

## Label convention

`0 = Real`, `1 = Fake` — used consistently across the dataset, the saved
model, and the API.

## Disclaimer

Built for educational purposes. Predictions reflect patterns in a limited
training dataset and should not be used as a substitute for real fact-checking.