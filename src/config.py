"""All experiment settings in one place.

Each config differs from the baseline ONLY in the two techniques under test:
  - top_k:             technique 1, top-k filtering (input tokens)
  - max_tokens +
    length_control:    technique 2, response length/structure control (output tokens)
Everything else (model, temperature, embedding model, history, retrieval query) is shared.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY and OPENAI_MODEL from .env (git-ignored)

# --- Shared by every config -------------------------------------------------
MODEL = os.environ.get("OPENAI_MODEL")  # set in .env, e.g. OPENAI_MODEL=<model name>
TEMPERATURE = 0

BASE_SYSTEM_PROMPT = (
    "You are an internal operational knowledge assistant. "
    "Answer the user's question using only the provided context passages. "
    "If the context does not contain the answer, say so."
)

# Technique 2: appended to the base prompt only when length_control is on.
LENGTH_CONSTRAINTS = """
You must follow these output constraints:
1. Length limit: your entire response must not exceed 150 words.
2. Structure: give the direct answer first, with the most important information first. \
Then end with exactly one line of the form "Sources: [n], [m]" listing the context passages you used.
3. No filler: no introductions (e.g. "Here is the answer"), no pleasantries, no repetition of the question."""

# Judge model for RAGAS (non-reasoning, temperature 0, different from the agent model).
JUDGE_MODEL = "gpt-4.1-mini"

# USD per 1M tokens for the agent model (gpt-5.4-nano, OpenAI pricing page).
PRICES = {"input": 0.20, "cached_input": 0.02, "output": 1.25}

# --- The configs ------------------------------------------------------------
CONFIGS = {
    "baseline":    {"top_k": 10, "max_tokens": 1024, "length_control": False},
    "combined":    {"top_k": 4,  "max_tokens": 200,  "length_control": True},
    # Follow-up (added after the main run): same as combined, but the top 4 are chosen by a cross-encoder
    # reranker from the embedding search's top 20, to recover gold passages that k=4 lost.
    "combined_rerank": {"top_k": 4, "max_tokens": 200, "length_control": True, "rerank_candidates": 20},
    # Ablations: each applies ONE technique to the baseline, to measure its effect in isolation.
    "topk_only":        {"top_k": 4,  "max_tokens": 1024, "length_control": False},
    "length_only":      {"top_k": 10, "max_tokens": 200,  "length_control": True},
    "topk_rerank_only": {"top_k": 4,  "max_tokens": 1024, "length_control": False, "rerank_candidates": 20},
}


def system_prompt(config):
    """The system prompt for a config: base prompt, plus constraints if length_control is on."""
    if config["length_control"]:
        return BASE_SYSTEM_PROMPT + "\n" + LENGTH_CONSTRAINTS
    return BASE_SYSTEM_PROMPT
