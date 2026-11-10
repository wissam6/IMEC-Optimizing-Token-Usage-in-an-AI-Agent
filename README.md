# Optimizing Token Usage in a RAG Knowledge Assistant (imec exercise, Part C)

A small proof of concept that compares a **baseline RAG agent** with a **token-optimized RAG agent**. The goal is to see whether token usage can drop without a significant loss in answer quality.

> Status: experiment design fixed (this section was written **before any runs**). Results will be added at the end.

## Techniques under test

1. **Top-k filtering** (reduces input tokens): retrieve fewer passages per question. Baseline uses `k = 10` and the optimized agent uses `k = 4`.
2. **Response length and structure control** (reduces output tokens): set `max_tokens = 200` and use a system prompt that limits the answer to about 150 words, gives the direct answer first and the cited sources after it, and allows no filler.

## Experiment design

| | Baseline | Combined (optimized) |
|---|---|---|
| Top-k | 10 | 4 |
| `max_tokens` | 1024 (a ceiling answers should never reach) | 200 |
| System prompt | answer only from the context | same + constraints: at most 150 words, answer first, then one `Sources: [n]` line, no filler |

All configs are defined in [src/config.py](src/config.py).

**Optional ablations (only if time allows):** Top-k only (k=4, no length control) and Length only (k=10 + length control).

**Held constant across configs:** the same OpenAI model (`gpt-5.4-nano-2026-03-17`; `gpt-5.6-luna` was rejected because it does not support temperature 0), temperature 0, the same embedding model and index, the same conversation history for each turn, and the same retrieval query (MTRAG's standalone rewritten question).

**Prompt design note:** Part B proposed "exactly two sentences". It was loosened to "at most 150 words, answer first" before any runs, because MTRAG reference answers have a median of 89 words; a one-sentence answer would lose FactualCorrectness recall by construction rather than because of the technique.

**Dataset:** [MTRAG](https://github.com/IBM/mt-rag-benchmark), a human-written multi-turn RAG benchmark (`mtrag-human`, passage-level corpora).

- **Domain: Govt.** It is the smallest corpus by passage count (49,607 passages; MTRAG's own README swaps the Cloud/FiQA/Govt sizes, so these are counted from the files), and its policy-style pages are closest to an internal knowledge base.
- **Turns: 30**, sampled with `seed = 42` from the 201 Govt tasks labelled ANSWERABLE or PARTIAL. UNANSWERABLE and CONVERSATIONAL turns are excluded because they have no gold passages, so Recall@k and FactualCorrectness are undefined or meaningless for them.
- **Resulting sample** ([data/sample_govt.jsonl](data/sample_govt.jsonl)): 28 answerable and 2 partial turns from 15 conversations, at turn positions 1 to 8, with 1 to 9 gold passages each (78 in total, all present in the corpus).
- Each turn stores the question, the prior conversation history, MTRAG's standalone rewritten question, the reference answer and the gold passage IDs. Recreate it with `python src/prepare_data.py sample`.

**Retriever:** `BAAI/bge-small-en-v1.5` embeddings (normalized, bge query prefix) with an exact FAISS inner-product index over the **full** Govt corpus, so no distractor subset is needed. Building it took 74 min on CPU (`python src/retriever.py build`); the index is git-ignored.

**Index sanity check** (`python src/retriever.py check`, retrieval only, no LLM, querying with the rewritten question):

| | Recall@k | Turns with ≥1 gold passage retrieved |
|---|---|---|
| k = 10 (Baseline) | 0.425 | 20 / 30 |
| k = 4 (Combined) | 0.283 | 15 / 30 |

### Metrics

| Metric | What it tells us |
|---|---|
| Input / output tokens per turn | the main target (taken from the API `usage` field) |
| Cost per turn | tokens × provider price |
| Latency per turn | wall-clock time of the LLM call |
| Recall@k | whether the gold passage(s) are still retrieved at the lower k |
| Truncation rate | share of answers with `finish_reason = length` |
| RAGAS Faithfulness | whether the answer is supported by the retrieved context |
| RAGAS FactualCorrectness (F1) | whether the answer agrees with the MTRAG reference answer; **the pass/fail metric** |
| FactualCorrectness precision | share of the answer's claims that the reference supports (*diagnostic only*: a drop means the agent says wrong things) |
| FactualCorrectness recall | share of the reference's claims that the answer covers (*diagnostic only*: a drop means the agent says less) |

RAGAS scores only the **answer part** of each response, not the cited-sources line. Precision, recall and F1 are computed from a **single judge pass** per answer (one claim decomposition and verification, three numbers derived from it), not from three separate RAGAS runs, so the three are consistent with each other.

### Pass/fail rules (fixed before running)

Let Δ = mean(Combined) − mean(Baseline), and let *noise* be the absolute difference between two identical Baseline runs on the same metric.

The optimized agent **passes** if both of these hold:

1. **FactualCorrectness (F1):** Δ ≥ −0.03, **or** |Δ| ≤ noise.
2. **Faithfulness:** Δ ≥ 0, **or** |Δ| ≤ noise.

FactualCorrectness precision and recall are reported next to F1 to explain *why* F1 moved, but they do not enter the verdict.

In other words, a drop only counts as real degradation if it is larger than the run-to-run noise. Turns where Combined scores lower than Baseline are reviewed by hand and labelled with a cause: *gold passage lost* (Recall@k), *truncated* (`finish_reason = length`), or *detail dropped* (shorter answer).

### Validation

- **Smoke test:** the first 5 turns of the sample (indices 0–4) for both configs (10 calls), read by hand before the full run.
- **Noise run:** Baseline is run twice to measure judge and model variance.
- **Judge hand-check:** 20 RAGAS judgments are checked by hand against my own assessment.
- **Unit tests** (`pytest`): the retriever returns exactly k passages, token counts are logged, each config changes only what it should, the system prompt contains the constraints, and the evaluation handles empty answers.

## Repository layout

```
src/
  retriever.py   embedding + top-k retrieval
  config.py      all configs and prompts in one place
  agent.py       run(turn, config), JSON-line logging of every LLM call (results/logs/)
  eval.py        RAGAS, Recall@k, token/cost/latency summaries
tests/           pytest unit tests
data/            sampled turns (raw MTRAG download and index are git-ignored)
results/         per-run CSVs, logs and charts
```

## How to run

_To be completed once the pipeline is built._

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Results

_To be added after the experiment._

## Limitations

_To be added._
