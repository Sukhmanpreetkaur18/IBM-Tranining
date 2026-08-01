"""
generate_utils.py
------------------
Generates sample news-style text using a small, free, fully local
language model (distilgpt2 from Hugging Face). No API key needed.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _get_generator():
    from transformers import pipeline
    return pipeline("text-generation", model="distilgpt2")


def generate_news(prompt: str, max_new_tokens: int = 80) -> str:
    generator = _get_generator()
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    return output[0]["generated_text"]