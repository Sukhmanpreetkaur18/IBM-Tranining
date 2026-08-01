"""
app.py
------
FastAPI backend. Run: uvicorn app:app --reload
"""

import pickle

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nlp_utils import clean_text
from explain_utils import explain_prediction

MODEL_PATH = "best_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl" 

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        tfidf = pickle.load(f)
except FileNotFoundError:
    print("Model files not found. Run `python train.py` first.")
    model, tfidf = None, None

app = FastAPI(title="Fake News Detector + Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewsItem(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    message: str


class ExplainResponse(PredictionResponse):
    top_words: list


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 80


class GenerateResponse(BaseModel):
    generated_text: str


def _predict(text: str):
    if model is None or tfidf is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    pred = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    confidence = float(proba.max() * 100)
    label = "Fake" if pred == 1 else "Real"
    return label, confidence, vector


@app.get("/")
async def root():
    return {"message": "Fake News Detector + Generator API is running."}


@app.post("/predict", response_model=PredictionResponse)
async def predict_news(item: NewsItem):
    label, confidence, _ = _predict(item.text)
    return PredictionResponse(
        prediction=label,
        confidence=round(confidence, 2),
        message=f"The model is {confidence:.2f}% confident this is {label.lower()} news.",
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain_news(item: NewsItem):
    label, confidence, vector = _predict(item.text)
    words = explain_prediction(vector, tfidf, model)
    return ExplainResponse(
        prediction=label,
        confidence=round(confidence, 2),
        message=f"The model is {confidence:.2f}% confident this is {label.lower()} news.",
        top_words=words,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(req: GenerateRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    try:
        from generate_utils import generate_news
        text = generate_news(req.prompt, req.max_new_tokens)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Generator not installed. Run: pip install transformers torch",
        )
    return GenerateResponse(generated_text=text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)