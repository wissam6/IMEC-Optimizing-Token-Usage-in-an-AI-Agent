"""All experiment settings in one place.

Each config differs from the baseline ONLY in the two techniques under test:
  - top_k:             technique 1, top-k filtering (input tokens)
  - max_tokens +
    length_control:    technique 2, response length/structure control (output tokens)
Everything else (model, temperature, embedding model, history, retrieval query) is shared.
"""
import os

from dotenv import load_dotenv

load_dotenv()

MODEL = os.environ.get("OPENAI_MODEL")
TEMPERATURE = 0

BASE_SYSTEM_PROMPT = (
    "You are an internal operational knowledge assistant. "
    "Answer the user's question using only the provided context passages. "
    "If the context does not contain the answer, say so."
)

LENGTH_CONSTRAINTS = """
You must follow these output constraints:
1. Length limit: your entire response must not exceed 150 words.
2. Structure: give the direct answer first, with the most important information first. \
Then end with exactly one line of the form "Sources: [n], [m]" listing the context passages you used.
3. No filler: no introductions (e.g. "Here is the answer"), no pleasantries, no repetition of the question."""

JUDGE_MODEL = "gpt-4.1-mini"

PRICES = {"input": 0.20, "cached_input": 0.02, "output": 1.25}

CONFIGS = {
    "baseline":    {"top_k": 10, "max_tokens": 1024, "length_control": False},
    "combined":    {"top_k": 4,  "max_tokens": 200,  "length_control": True},
    "combined_rerank": {"top_k": 4, "max_tokens": 200, "length_control": True, "rerank_candidates": 20},
    "topk_only":        {"top_k": 4,  "max_tokens": 1024, "length_control": False},
    "length_only":      {"top_k": 10, "max_tokens": 200,  "length_control": True},
    "topk_rerank_only": {"top_k": 4,  "max_tokens": 1024, "length_control": False, "rerank_candidates": 20},
}


def system_prompt(config):
    if config["length_control"]:
        return BASE_SYSTEM_PROMPT + "\n" + LENGTH_CONSTRAINTS
    return BASE_SYSTEM_PROMPT
