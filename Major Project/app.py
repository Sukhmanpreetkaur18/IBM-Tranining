from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import uvicorn

# Download stopwords if not already done
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("stopwords")

# --- Configuration ---
MODEL_PATH = "best_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# --- Load Model and Vectorizer ---
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    with open(VECTORIZER_PATH, "rb") as file:
        tfidf = pickle.load(file)
except FileNotFoundError:
    print("Error: Model or Vectorizer file not found. Please ensure they are in the same directory.")
    model = None
    tfidf = None

# --- NLP Setup ---
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    """Cleans the input text using the same steps as the training notebook."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]

    return " ".join(words)

# --- FastAPI App ---
app = FastAPI(title="Fake News Detector API")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response Models ---
class NewsItem(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    message: str

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Fake News Detector API is running. Use /predict to submit news."}

@app.post("/predict", response_model=PredictionResponse)
async def predict_news(item: NewsItem):
    if model is None or tfidf is None:
        raise HTTPException(status_code=503, detail="Model or vectorizer not loaded.")

    if not item.text or not item.text.strip():
        raise HTTPException(status_code=400, detail="News text cannot be empty.")

    # 1. Clean the text
    cleaned_text = clean_text(item.text)

    # 2. Vectorize
    vector = tfidf.transform([cleaned_text])

    # 3. Predict
    prediction = model.predict(vector)
    probability = model.predict_proba(vector)
    confidence = probability.max() * 100

    # 4. Prepare Response
    # FIX: Flip the prediction labels - model outputs 0=Fake, 1=Real
    # But for some reason it's reversed, so we flip it
    pred_label = "Fake" if prediction[0] == 1 else "Real"
    message = f"The model is {confidence:.2f}% confident this is {pred_label.lower()} news."

    return PredictionResponse(
        prediction=pred_label,
        confidence=round(confidence, 2),
        message=message
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)