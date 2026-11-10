"""Dense retriever: bge-small embeddings + exact FAISS inner-product search over one MTRAG corpus.

Usage:
  python src/retriever.py build    # embed the whole Govt corpus, save to data/index/ (74 min on CPU)
  python src/retriever.py check    # Recall@k on the sampled turns, with and without the reranker

In code:
  r = Retriever.load()
  r.search("How do I know a government website is secure?", k=4)
  r.search("...", k=4, rerank_candidates=20)   # top 20 by embedding, reranked by a cross-encoder, keep 4
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
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANKER_NAME = "BAAI/bge-reranker-base"


def load_corpus():
    with zipfile.ZipFile(CORPUS_ZIP) as zf:
        name = [x for x in zf.namelist() if x.endswith(".jsonl")][0]
        with zf.open(name) as f:
            return [{"id": d["_id"], "text": d["text"]} for d in map(json.loads, f)]


class Retriever:
    def __init__(self, model, index, passages):
        self.model = model
        self.index = index
        self.passages = passages

    @classmethod
    def build(cls):
        passages = load_corpus()
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        print(f"Embedding {len(passages):,} passages with {MODEL_NAME} ...")
        t0 = time.time()
        emb = model.encode(
            [p["text"] for p in passages],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")
        print(f"Done in {(time.time() - t0) / 60:.1f} min")

        index = faiss.IndexFlatIP(emb.shape[1])
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

    def search(self, query, k, rerank_candidates=None):
        n = max(k, rerank_candidates or 0)
        q = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True)
        scores, idx = self.index.search(q.astype("float32"), n)
        hits = [{**self.passages[i], "score": float(s)} for s, i in zip(scores[0], idx[0])]
        if not rerank_candidates:
            return hits[:k]
        rerank_scores = self.reranker().predict([(query, h["text"]) for h in hits])
        order = np.argsort(-rerank_scores, kind="stable")[:k]
        return [{**hits[i], "score": float(rerank_scores[i])} for i in order]

    def reranker(self):
        if not hasattr(self, "_reranker"):
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(RERANKER_NAME, device="cpu", max_length=512)
        return self._reranker


def recall_at_k(retrieved_ids, gold_ids):
    if not gold_ids:
        return None
    return len(set(retrieved_ids) & set(gold_ids)) / len(set(gold_ids))


def check():
    r = Retriever.load()
    with open(SAMPLE, encoding="utf-8") as f:
        turns = [json.loads(line) for line in f]
    for k, rerank in ((4, None), (10, None), (20, None), (4, 20)):
        recalls, hits = [], 0
        t0 = time.time()
        for t in turns:
            ids = [p["id"] for p in r.search(t["rewritten_question"], k, rerank_candidates=rerank)]
            rec = recall_at_k(ids, t["gold_passage_ids"])
            recalls.append(rec)
            hits += rec > 0
        label = f"Recall@{k}" + (f" (rerank top {rerank})" if rerank else "")
        print(f"{label:<26}: {np.mean(recalls):.3f}   turns with >=1 gold: {hits}/{len(turns)}"
              f"   {(time.time() - t0) / len(turns):.2f} s/query")


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        Retriever.build()
    elif sys.argv[1:] == ["check"]:
        check()
    else:
        print(__doc__)
