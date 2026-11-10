"""Dense retriever: bge-small embeddings + exact FAISS inner-product search over one MTRAG corpus.

Usage:
  python src/retriever.py build    # embed the whole Govt corpus, save to data/index/ (~10-15 min on CPU)
  python src/retriever.py check    # Recall@4 / Recall@10 on the sampled turns

In code:
  r = Retriever.load()
  r.search("How do I know a government website is secure?", k=4)
"""
import json
import sys
import time
import zipfile
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CORPUS_ZIP = ROOT / "data" / "raw" / "mt-rag-benchmark" / "corpora" / "passage_level" / "govt.jsonl.zip"
INDEX_DIR = ROOT / "data" / "index"
SAMPLE = ROOT / "data" / "sample_govt.jsonl"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# bge v1.5 recommends this instruction on queries (not on passages) for retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_corpus():
    """Return a list of {'id', 'text'} dicts from the zipped passage-level corpus."""
    with zipfile.ZipFile(CORPUS_ZIP) as zf:
        name = [x for x in zf.namelist() if x.endswith(".jsonl")][0]
        with zf.open(name) as f:
            return [{"id": d["_id"], "text": d["text"]} for d in map(json.loads, f)]


class Retriever:
    def __init__(self, model, index, passages):
        self.model = model
        self.index = index
        self.passages = passages  # position i in the index == passages[i]

    @classmethod
    def build(cls):
        passages = load_corpus()
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        print(f"Embedding {len(passages):,} passages with {MODEL_NAME} ...")
        t0 = time.time()
        emb = model.encode(
            [p["text"] for p in passages],
            batch_size=64,
            normalize_embeddings=True,  # normalized + inner product == cosine similarity
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")
        print(f"Done in {(time.time() - t0) / 60:.1f} min")

        index = faiss.IndexFlatIP(emb.shape[1])  # exact search; 50k x 384 is small enough
        index.add(emb)

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(INDEX_DIR / "govt.faiss"))
        with open(INDEX_DIR / "govt_passages.jsonl", "w", encoding="utf-8") as f:
            for p in passages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        return cls(model, index, passages)

    @classmethod
    def load(cls):
        index = faiss.read_index(str(INDEX_DIR / "govt.faiss"))
        with open(INDEX_DIR / "govt_passages.jsonl", encoding="utf-8") as f:
            passages = [json.loads(line) for line in f]
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        return cls(model, index, passages)

    def search(self, query, k):
        """Return exactly k passages as dicts {'id', 'text', 'score'}, best first."""
        q = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True)
        scores, idx = self.index.search(q.astype("float32"), k)
        return [{**self.passages[i], "score": float(s)} for s, i in zip(scores[0], idx[0])]


def recall_at_k(retrieved_ids, gold_ids):
    """Share of gold passages that appear in the retrieved list."""
    if not gold_ids:
        return None
    return len(set(retrieved_ids) & set(gold_ids)) / len(set(gold_ids))


def check():
    """Sanity check: Recall@k on the sampled turns, querying with MTRAG's rewritten question."""
    r = Retriever.load()
    with open(SAMPLE, encoding="utf-8") as f:
        turns = [json.loads(line) for line in f]
    for k in (4, 10):
        recalls, hits = [], 0
        for t in turns:
            ids = [p["id"] for p in r.search(t["rewritten_question"], k)]
            rec = recall_at_k(ids, t["gold_passage_ids"])
            recalls.append(rec)
            hits += rec > 0
        print(f"Recall@{k:<2}: {np.mean(recalls):.3f}   turns with >=1 gold passage retrieved: {hits}/{len(turns)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        Retriever.build()
    elif sys.argv[1:] == ["check"]:
        check()
    else:
        print(__doc__)
