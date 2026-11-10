"""Evaluation: RAGAS Faithfulness and FactualCorrectness, Recall@k, token/cost/latency summaries.

Usage:
  python src/eval.py score <run_name> [n]        # judge one run's answers -> results/scores_<run_name>.csv
  python src/eval.py summary <run_name> ...      # per-config averages     -> results/summary.csv

FactualCorrectness: one judge pass per answer, using RAGAS's own two steps and formulas:
  - answer claims checked against the reference     -> TP (supported) / FP (not supported)
  - reference claims checked against the answer     -> FN (not covered)
  precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = ragas fbeta_score(TP, FP, FN)
"""
import asyncio
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "results" / "logs"
RESULTS = ROOT / "results"
SAMPLE = ROOT / "data" / "sample_govt.jsonl"
PASSAGES = ROOT / "data" / "index" / "govt_passages.jsonl"
CONCURRENCY = 5

_SOURCES_LINE = re.compile(r"\s*\**sources?\**\s*:", re.IGNORECASE)


def answer_for_scoring(answer):
    if not answer or not answer.strip():
        return ""
    lines = answer.strip().splitlines()
    for i in range(len(lines) - 1, 0, -1):
        if _SOURCES_LINE.match(lines[i]):
            return "\n".join(lines[:i]).strip()
    return answer.strip()


def is_empty(answer):
    return answer_for_scoring(answer) == ""


def recall_at_k(retrieved_ids, gold_ids):
    gold = set(gold_ids)
    return len(set(retrieved_ids) & gold) / len(gold) if gold else None


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_metrics():
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import FactualCorrectness, Faithfulness

    from config import JUDGE_MODEL

    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    return FactualCorrectness(llm=llm), Faithfulness(llm=llm)


async def judge_one(record, turn, contexts, fc, ff, sem):
    from ragas import SingleTurnSample
    from ragas.metrics.utils import fbeta_score

    answer = answer_for_scoring(record["answer"])
    reference = turn["reference_answer"]
    row = {"empty": answer == "", "error": ""}
    detail = {"task_id": record["task_id"], "config": record["config"], "question": turn["question"],
              "answer_scored": answer, "reference": reference}

    if row["empty"]:
        row.update(fc_precision=0.0, fc_recall=0.0, fc_f1=0.0, faithfulness=0.0, tp=0, fp=0, fn=0)
        return row, detail

    async with sem:
        for attempt in (1, 2):
            try:
                answer_claims = await fc.decompose_claims(answer, None)
                ref_claims = await fc.decompose_claims(reference, None)
                a_ok, r_ok = await asyncio.gather(
                    fc.verify_claims(premise=reference, hypothesis_list=answer_claims, callbacks=None),
                    fc.verify_claims(premise=answer, hypothesis_list=ref_claims, callbacks=None),
                )
                faith = await ff.single_turn_ascore(SingleTurnSample(
                    user_input=turn["rewritten_question"], response=answer, retrieved_contexts=contexts))
                break
            except Exception as e:
                if attempt == 2:
                    row.update(error=f"{type(e).__name__}: {e}"[:300])
                    return row, detail

    tp, fp, fn = int(a_ok.sum()), int((~a_ok).sum()), int((~r_ok).sum())
    row.update(
        fc_precision=tp / (tp + fp) if tp + fp else 0.0,
        fc_recall=tp / (tp + fn) if tp + fn else 0.0,
        fc_f1=float(fbeta_score(tp, fp, fn, 1.0)),
        faithfulness=float(faith) if faith is not None else math.nan,
        tp=tp, fp=fp, fn=fn,
    )
    detail.update(answer_claims=list(zip(answer_claims, a_ok.tolist())),
                  reference_claims=list(zip(ref_claims, r_ok.tolist())))
    return row, detail


async def score_run(run_name, n=None):
    import pandas as pd

    records = load_jsonl(LOG_DIR / f"{run_name}.jsonl")[:n]
    turns = {t["task_id"]: t for t in load_jsonl(SAMPLE)}
    needed = {pid for r in records for pid in r["retrieved_ids"]}
    texts = {p["id"]: p["text"] for p in load_jsonl(PASSAGES) if p["id"] in needed}

    fc, ff = make_metrics()
    sem = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(*[
        judge_one(r, turns[r["task_id"]], [texts[i] for i in r["retrieved_ids"]], fc, ff, sem) for r in records
    ])

    rows = []
    for r, (scores, _) in zip(records, results):
        t = turns[r["task_id"]]
        rows.append({
            "run_name": run_name, "config": r["config"], "task_id": r["task_id"], "turn": t["turn"],
            "answerability": t["answerability"], "top_k": r["top_k"], "max_tokens": r["max_tokens"],
            "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
            "cached_input_tokens": r["cached_input_tokens"], "latency_s": r["latency_s"],
            "retrieval_latency_s": r.get("retrieval_latency_s"),
            "finish_reason": r["finish_reason"], "truncated": r["finish_reason"] == "length",
            "recall_at_k": recall_at_k(r["retrieved_ids"], r["gold_ids"]),
            "answer_words": len(answer_for_scoring(r["answer"]).split()),
            **scores,
        })
    df = pd.DataFrame(rows)
    suffix = f"_first{n}" if n else ""
    df.to_csv(RESULTS / f"scores_{run_name}{suffix}.csv", index=False)
    with open(RESULTS / f"judgments_{run_name}{suffix}.jsonl", "w", encoding="utf-8") as f:
        for _, detail in results:
            f.write(json.dumps(detail, ensure_ascii=False) + "\n")

    print(f"Scored {len(df)} answers -> results/scores_{run_name}{suffix}.csv")
    print(f"errors: {(df['error'] != '').sum()}  empty: {df['empty'].sum()}  faithfulness NaN: {df['faithfulness'].isna().sum()}")
    print(df[["fc_precision", "fc_recall", "fc_f1", "faithfulness", "recall_at_k"]].mean().round(3).to_string())


def cost_usd(row, prices):
    if any(v is None for v in prices.values()):
        return None
    cached = row["cached_input_tokens"] or 0
    return ((row["input_tokens"] - cached) * prices["input"] + cached * prices["cached_input"]
            + row["output_tokens"] * prices["output"]) / 1e6


def summary(run_names):
    import pandas as pd

    from config import PRICES

    df = pd.concat([pd.read_csv(RESULTS / f"scores_{r}.csv") for r in run_names])
    df["cost_usd"] = df.apply(cost_usd, axis=1, prices=PRICES)
    df["cost_usd_per_1k_questions"] = pd.to_numeric(df["cost_usd"]) * 1000
    s = df.groupby(["run_name", "config"]).agg(
        n=("task_id", "count"),
        input_tokens=("input_tokens", "mean"), output_tokens=("output_tokens", "mean"),
        cost_usd_per_1k_questions=("cost_usd_per_1k_questions", "mean"), latency_s=("latency_s", "mean"),
        retrieval_latency_s=("retrieval_latency_s", "mean"),
        truncation_rate=("truncated", "mean"), recall_at_k=("recall_at_k", "mean"),
        fc_f1=("fc_f1", "mean"), fc_precision=("fc_precision", "mean"), fc_recall=("fc_recall", "mean"),
        faithfulness=("faithfulness", "mean"), empty=("empty", "sum"),
    ).round(3)
    s.to_csv(RESULTS / "summary.csv")
    print(s.T.to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) in (2, 3) and args[0] == "score":
        asyncio.run(score_run(args[1], int(args[2]) if len(args) == 3 else None))
    elif len(args) >= 2 and args[0] == "summary":
        summary(args[1:])
    else:
        print(__doc__)
