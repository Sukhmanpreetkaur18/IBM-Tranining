"""
nlp_utils.py
------------
Single shared source of truth for text cleaning.

IMPORTANT: This exact function is used both when TRAINING the model
(notebook) and when SERVING predictions (app.py). Never duplicate this
logic in two places — if training and serving ever clean text differently,
predictions will be wrong even if the model itself is fine.
"""

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    text = str(text).lower()

    # --- Remove dataset-specific "shortcut" signals ---
    text = re.sub(r"^[a-z\s,\.\-]+\(reuters\)\s*-\s*", "", text)
    text = re.sub(r"\breuters\b", "", text)
    text = re.sub(r"21st century wire", "", text)
    text = re.sub(r"21stcenturywire\.com", "", text)
    text = re.sub(r"featured image", "", text)
    text = re.sub(r"\bgetty\b", "", text)
    text = re.sub(r"image via", "", text)
    text = re.sub(r"via twitter", "", text)
    text = re.sub(r"read more .*? at:.*", "", text)
    text = re.sub(r"pic\.twitter\.com\S*", "", text)

    # --- Standard cleaning ---
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)

    # Normalize ALL punctuation (ascii + unicode smart quotes/dashes/etc.)
    # to a space rather than deleting it, so contractions split consistently
    # regardless of which apostrophe style the source text used.
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]

    return " ".join(words)