"""Inspect the MTRAG download and sample the evaluation turns.

Usage:
  python src/prepare_data.py inspect   # per-domain corpus size and task statistics
  python src/prepare_data.py sample    # write data/sample_govt.jsonl
"""
import json
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MTRAG = ROOT / "data" / "raw" / "mt-rag-benchmark"
CORPORA = MTRAG / "corpora" / "passage_level"
TASKS = MTRAG / "mtrag-human" / "generation_tasks" / "reference.jsonl"
DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]

# Sampling choices (see README): smallest corpus by passage count, gold passages required.
DOMAIN = "govt"
KEEP = ("ANSWERABLE", "PARTIAL")  # UNANSWERABLE / CONVERSATIONAL have no gold passages
N_SAMPLE = 30
SEED = 42


def count_passages(domain):
    """Count passages and total characters in a zipped passage-level corpus."""
    n, chars = 0, 0
    with zipfile.ZipFile(CORPORA / f"{domain}.jsonl.zip") as zf:
        name = [x for x in zf.namelist() if x.endswith(".jsonl")][0]
        with zf.open(name) as f:
            for line in f:
                n += 1
                chars += len(json.loads(line)["text"])
    return n, chars


def load_tasks():
    with open(TASKS, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def inspect():
    tasks = load_tasks()
    # Note: the real files use capitalised keys ("Collection", "Answerability"), unlike the README.
    by_domain = {d: [t for t in tasks if d in t["Collection"]] for d in DOMAINS}
    print(f"Total generation tasks: {len(tasks)}")
    print(f"Collection names seen: {sorted(Counter(t['Collection'] for t in tasks))}\n")
    for d in DOMAINS:
        n, chars = count_passages(d)
        ts = by_domain[d]
        ans = Counter(t["Answerability"][0] for t in ts)
        convs = len({t["conversation_id"] for t in ts})
        n_ctx = Counter(len(t["contexts"]) for t in ts)
        hist = [sum(len(x["text"]) for x in t["input"][:-1]) for t in ts]
        print(f"== {d} ==")
        print(f"  passages: {n:,}   avg chars/passage: {chars / n:.0f}")
        print(f"  tasks: {len(ts)}   conversations: {convs}")
        print(f"  answerability: {dict(ans)}")
        print(f"  gold passages per task: {dict(sorted(n_ctx.items()))}")
        print(f"  avg history chars before the question: {sum(hist) / len(hist):.0f}\n")


def load_rewrites(domain):
    """Standalone (rewritten) question per task, e.g. '|user|: What items ...' -> 'What items ...'."""
    path = MTRAG / "mtrag-human" / "retrieval_tasks" / domain / f"{domain}_rewrite.jsonl"
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {r["_id"]: r["text"].removeprefix("|user|:").strip() for r in rows}


def sample(domain=DOMAIN, n=N_SAMPLE, seed=SEED):
    """Sample n ANSWERABLE/PARTIAL tasks from one domain and save them to data/sample_<domain>.jsonl."""
    tasks = [t for t in load_tasks() if domain in t["Collection"] and t["Answerability"][0] in KEEP]
    tasks.sort(key=lambda t: t["task_id"])  # fixed order so the seed is reproducible
    picked = random.Random(seed).sample(tasks, n)
    rewrites = load_rewrites(domain)

    out = ROOT / "data" / f"sample_{domain}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for t in picked:
            row = {
                "task_id": t["task_id"],
                "conversation_id": t["conversation_id"],
                "turn": int(t["turn"]),
                "answerability": t["Answerability"][0],
                "question": t["input"][-1]["text"],
                "rewritten_question": rewrites.get(t["task_id"]),
                "history": [{"speaker": x["speaker"], "text": x["text"]} for x in t["input"][:-1]],
                "reference_answer": t["targets"][0]["text"],
                "gold_passage_ids": [c["document_id"] for c in t["contexts"]],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Eligible {domain} tasks ({'/'.join(KEEP)}): {len(tasks)}  ->  sampled {n} with seed {seed}")
    print(f"Saved to {out.relative_to(ROOT)}")
    print(f"answerability: {dict(Counter(t['Answerability'][0] for t in picked))}")
    print(f"turn position: {dict(sorted(Counter(int(t['turn']) for t in picked).items()))}")
    print(f"gold passages per task: {dict(sorted(Counter(len(t['contexts']) for t in picked).items()))}")
    print(f"conversations covered: {len({t['conversation_id'] for t in picked})}")
    print(f"missing rewritten question: {sum(rewrites.get(t['task_id']) is None for t in picked)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["inspect"]:
        inspect()
    elif sys.argv[1:] == ["sample"]:
        sample()
    else:
        print(__doc__)
