"""RAG agent: run(turn, config) -> answer + token usage, with one JSON log line per LLM call.

Usage:
  python src/agent.py preview <config> [turn_index]   # print the exact messages, NO API call, no cost
  python src/agent.py run <config> [turn_index]       # one real API call (costs money)

  <config> is a key of CONFIGS in src/config.py (baseline, combined, ...).
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import CONFIGS, MODEL, TEMPERATURE, system_prompt
from retriever import Retriever

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample_govt.jsonl"
LOG_DIR = ROOT / "results" / "logs"

_retriever = None
_client = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever.load()
    return _retriever


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI  # imported here so `preview` works without an API key

        _client = OpenAI()  # reads OPENAI_API_KEY from the environment / .env
    return _client


def load_turns():
    with open(SAMPLE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_messages(turn, passages, config):
    """System prompt + full conversation history + (context passages and the current question)."""
    messages = [{"role": "system", "content": system_prompt(config)}]
    for msg in turn["history"]:
        role = "user" if msg["speaker"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["text"]})

    context = "\n\n".join(f"[{i}] {p['text'].strip()}" for i, p in enumerate(passages, start=1))
    messages.append({"role": "user", "content": f"Context passages:\n\n{context}\n\nQuestion: {turn['question'].strip()}"})
    return messages


def run(turn, config_name, run_name="dev", call_llm=None):
    """Answer one turn with one config. Returns a result dict and appends it to results/logs/<run_name>.jsonl.

    call_llm can be replaced in tests with a fake function(messages, max_tokens) -> OpenAI-style response.
    """
    config = CONFIGS[config_name]
    if call_llm is None and not MODEL:
        raise RuntimeError("Set OPENAI_MODEL in .env before making real API calls.")

    # Retrieval query: MTRAG's standalone rewrite (same for every config).
    t_retrieval = time.perf_counter()
    passages = get_retriever().search(
        turn["rewritten_question"], config["top_k"], rerank_candidates=config.get("rerank_candidates"))
    retrieval_latency = time.perf_counter() - t_retrieval
    messages = build_messages(turn, passages, config)

    call_llm = call_llm or _openai_call
    t0 = time.perf_counter()
    response = call_llm(messages, config["max_tokens"])
    latency = time.perf_counter() - t0

    choice, usage = response.choices[0], response.usage
    details = usage.completion_tokens_details
    cached = usage.prompt_tokens_details
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "config": config_name,
        "task_id": turn["task_id"],
        "model": response.model,
        "top_k": config["top_k"],
        "max_tokens": config["max_tokens"],
        "length_control": config["length_control"],
        "rerank_candidates": config.get("rerank_candidates"),
        "retrieved_ids": [p["id"] for p in passages],
        "gold_ids": turn["gold_passage_ids"],
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
        "cached_input_tokens": getattr(cached, "cached_tokens", None) if cached else None,
        "finish_reason": choice.finish_reason,
        "latency_s": round(latency, 3),  # LLM call only
        "retrieval_latency_s": round(retrieval_latency, 3),  # embedding search (+ reranking, if any)
        "answer": choice.message.content or "",
    }
    log(result, run_name)
    return result


def _openai_call(messages, max_tokens):
    return get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_completion_tokens=max_tokens,  # the current name of max_tokens in the OpenAI API
    )


def log(result, run_name):
    """Append one JSON line per LLM call."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / f"{run_name}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def preview(config_name, i=0):
    """Print exactly what would be sent to the LLM, without calling it."""
    turn = load_turns()[i]
    config = CONFIGS[config_name]
    passages = get_retriever().search(
        turn["rewritten_question"], config["top_k"], rerank_candidates=config.get("rerank_candidates"))
    messages = build_messages(turn, passages, config)
    print(f"config={config_name} {config}  model={MODEL}  task={turn['task_id']}")
    print(f"retrieval query: {turn['rewritten_question']}")
    print(f"retrieved: {[p['id'] for p in passages]}\n")
    for m in messages:
        text = m["content"] if len(m["content"]) < 1500 else m["content"][:1500] + f"\n... [{len(m['content'])} chars total]"
        print(f"--- {m['role']} ---\n{text}\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] in ("preview", "run") and args[1] in CONFIGS:
        idx = int(args[2]) if len(args) > 2 else 0
        if args[0] == "preview":
            preview(args[1], idx)
        else:
            print(json.dumps(run(load_turns()[idx], args[1]), indent=2, ensure_ascii=False))
    else:
        print(__doc__)
