# Optimizing Token Usage in a RAG Knowledge Assistant - Documentation

This is a small proof of concept on optimizing token usage for a RAG knowledge retrieval application.

## Results

The optimized agent (top-k=4 + response-length control) cut total tokens by 52% and cost by 49%, but failed the quality rules (FactualCorrectness F1 −0.048, Faithfulness −0.005). In exploratory follow-ups, **top-k filtering with a cross-encoder reranker, without length control**, cut tokens by 53% and passed both rules (F1 and Faithfulness).

| Config | Tokens | Cost / 1k q. | Δ F1 | Δ Faithfulness | Rules |
|---|---|---|---|---|---|
| Baseline | 4,892 | $1.12 | (0.426) | (0.945) | — |
| **Combined** | −52% | $0.57 | −0.048 [−0.15, +0.05] | −0.005 [−0.08, +0.07] | fail |
| Combined + reranker | −52% | $0.56 | −0.032 [−0.10, +0.04] | +0.029 [−0.03, +0.10] | fail (F1) |
| Top-k only | −53% | $0.59 | −0.035 [−0.12, +0.05] | −0.023 [−0.08, +0.04] | fail |
| Top-k + reranker only | −53% | $0.60 | −0.008 [−0.07, +0.06] | +0.020 [−0.01, +0.06] | **pass** |
| Length only | +1% | $1.09 | −0.039 [−0.10, +0.02] | +0.002 [−0.06, +0.07] | fail (F1) |

## Techniques used

The methods used in this application are *top-k filtering* with *re-reranking*, as well as response control using a system prompt and setting the *max_tokens* variable. GPT 5.4 is used inside Ragas and functions as an *LLM as a judge* to evaluate the results.

## Tools used

The tools used are:
- mtrag benchmark dataset
- FactualCorrectness F1 and Faithfulness (mtrag tools for evaluation)
- ragas
- openai api ([gpt 5.4 nano](https://developers.openai.com/api/docs/models/gpt-5.4-nano))
- [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) (cross-encoder for re-ranking)


## Experiment design

The experiment is a 5 way ablation:

- Baseline (k=10)
- Top-k filtering (k=4) + re-ranking 
- Top-k filtering (k=4) + response control (max_tokens=200)
- Top-k (k=4) + response control (max_tokens=200) + re-reranking
- Top-k filtering only (k=4)
- Response control only (max_tokens=200)

The base prompt is the following:

```
BASE_SYSTEM_PROMPT = (
    "You are an internal operational knowledge assistant. "
    "Answer the user's question using only the provided context passages. "
    "If the context does not contain the answer, say so."
)
```

When limiting the output to **max_token = 200 (150 words)* as a technique to reduce tokens, this is also added:

```
You must follow these output constraints:
1. Length limit: your entire response must not exceed 150 words.
2. Structure: give the direct answer first, with the most important information first. \
Then end with exactly one line of the form "Sources: [n], [m]" listing the context passages you used.
3. No filler: no introductions (e.g. "Here is the answer"), no pleasantries, no repetition of the question."""
```

Temperature is set to 0, in order to have a fair comparison between different ablations

For the top-k filtering, for the baseline, k is set to 10, while in the experiment it is set to 4.

From MTRAG, we are using the GOVT domain, and we are sampling 30 questions that are conversation turns from the human written part of MTRAG (mtrag-human inside their repo)

GPT 5.4 nano is used are the RAG agent, while gpt 5.4 mini is used as a RAGAS judge. So first we run the agent to get the answers, then the RAGAS judge evaluates them. They are FactualCorrectness and Faithfullness (built-in).

## Validation

- **Smoke test:** 5 turns per config read by hand before the full run.
- **Unit tests:** 32 pytest tests (`python -m pytest`, no API calls): retrieval returns exactly k passages, tokens are logged per call, each config changes only its own technique.


## Repository layout

```
src/
  prepare_data.py   inspect MTRAG and sample the 30 turns
  retriever.py      embeddings + FAISS top-k retrieval, optional cross-encoder reranking
  config.py         all configs, prompts, judge model and prices in one place
  agent.py          run(turn, config)
  run_experiment.py run configs over the sampled turns
  eval.py           RAGAS judging (FactualCorrectness P/R/F1 + Faithfulness), Recall@k, cost summary
  analysis.py       results table, chart, per-turn causes, hand-check export
tests/              pytest unit tests
data/               sampled turns (raw MTRAG download and index)
results/
  scores_*.csv      per-turn metrics per run; judgments_*.jsonl: every judged claim
  summary.csv, results_table.md, comparison_vs_baseline.csv, token_vs_quality.png,
  worse_turns.csv, hand_check.csv
```

## How to run

**Change k (number of retrieved passages):** edit `top_k` for a config in `CONFIGS` in `src/config.py`
(e.g. `"baseline": {"top_k": 10, ...}`).

**Change the prompt:** edit `BASE_SYSTEM_PROMPT` in `src/config.py`. The length rules added by the
optimized configs are in `LENGTH_CONSTRAINTS` in the same file.

**Run the RAG agent** (calls the OpenAI API; pick a new run name each time, e.g. `my_run`):

```powershell
python src/run_experiment.py my_run baseline
```

**Evaluate it with RAGAS** (calls the judge model):

```powershell
python src/eval.py score my_run
python src/eval.py summary my_run
```

`score` judges each answer with the RAGAS judge (paid) and writes per-question scores to
`results/scores_my_run.csv`. `summary` averages those scores per config (free) and writes `results/summary.csv`;
pass several run names to compare them side by side.

## Lessons

## Main Sources

- [Ragas](https://arxiv.org/abs/2309.15217)
- [MTRAG](https://arxiv.org/abs/2501.03468)
- Articles:
  - https://knowledged.to/notes/ml/top-k-in-rag-search/
  - https://neuraltrust.ai/blog/ai-token-optimization-guide
  - https://apxml.com/courses/optimizing-rag-for-production/chapter-5-cost-optimization-production-rag/minimize-llm-token-usage-rag


## AI Usage

AI was NOT used to generate any text. However, I used Claude Opus 5.5 Medium as a code assistant. It was very useful, for instance, it recommended me using a re-ranker along with the top-k filtering when that alone failed to pass the tests.

The table under the results is also generated with my coding assistant.

## Time Split (24 hours total)

- 2h research and planning (5/11/2026)
- 2h Answering A and B (6/11/2026)
- 1h Answering A and B (7/11/2026)
- 1h B (8/11)
- 2h B (9/11)
- 15 hours code and documentation (9/11 and 10/11)
- 1 hour powerpoint

