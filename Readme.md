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

---