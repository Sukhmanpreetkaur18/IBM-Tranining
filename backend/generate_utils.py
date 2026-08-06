"""
generate_utils.py
------------------
Generates sample news-style text using distilgpt2 from Hugging Face,
with a fast template fallback if memory or model loading fails on free cloud hosting.
"""

from functools import lru_cache
import random

FALLBACK_TEMPLATES = [
    "Further details released by officials indicate ongoing investigations into the matter. Authorities emphasized that public safety and transparency remain top priorities.",
    "According to preliminary reports, experts have advised caution while independent data verification is underway.",
    "Sources close to the development confirmed that additional updates will be published as official findings are finalized.",
    "Industry analysts note that this development could have widespread implications across multiple key sectors over the coming months.",
    "Official spokespersons stated that comprehensive measures are being actively implemented to address all primary concerns."
]

@lru_cache(maxsize=1)
def _get_generator():
    try:
        from transformers import pipeline
        return pipeline("text-generation", model="distilgpt2")
    except Exception as e:
        print(f"Could not load distilgpt2 model: {e}")
        return None


def generate_news(prompt: str, max_new_tokens: int = 50) -> str:
    try:
        generator = _get_generator()
        if generator is not None:
            output = generator(
                prompt,
                max_new_tokens=min(max_new_tokens, 50),
                num_return_sequences=1,
                do_sample=True,
                temperature=0.85,
                top_p=0.9,
                pad_token_id=generator.tokenizer.eos_token_id,
            )
            return output[0]["generated_text"]
    except Exception as err:
        print(f"GenAI model failed or timed out on cloud environment, using fallback: {err}")

    # Fast fallback generator for lightweight cloud environments
    suffix = random.choice(FALLBACK_TEMPLATES)
    return f"{prompt} — {suffix}"