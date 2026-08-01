"""
explain_utils.py
-----------------
Shows which words in the input pushed the prediction toward "Real" or "Fake".
"""

import numpy as np


def explain_prediction(text_vector, tfidf, model, top_n=8):
    feature_names = np.array(tfidf.get_feature_names_out())
    nonzero_idx = text_vector.nonzero()[1]

    if not len(nonzero_idx):
        return []

    if hasattr(model, "coef_"):
        weights = model.coef_[0][nonzero_idx] * text_vector.toarray()[0][nonzero_idx]
        words = feature_names[nonzero_idx]
        ranked = sorted(zip(words, weights), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        return [
            {"word": w, "leans": "fake" if wt > 0 else "real", "weight": round(float(abs(wt)), 4)}
            for w, wt in ranked
        ]
    else:
        weights = model.feature_importances_[nonzero_idx] * text_vector.toarray()[0][nonzero_idx]
        words = feature_names[nonzero_idx]
        ranked = sorted(zip(words, weights), key=lambda x: x[1], reverse=True)[:top_n]
        return [
            {"word": w, "leans": "influential", "weight": round(float(wt), 4)}
            for w, wt in ranked
        ]