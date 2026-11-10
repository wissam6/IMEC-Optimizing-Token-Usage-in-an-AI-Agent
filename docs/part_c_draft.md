## C. Small experimental proposal and proof of concept

**Hypothesis.** Retrieving fewer passages (top-k 10 → 4) and constraining the answer (max 200 tokens, ≤150 words, answer first, no filler) roughly halves token usage without a meaningful loss in answer quality.

**Design.** Baseline vs Combined (both techniques), held equal on everything else: `gpt-5.4-nano` at temperature 0, the same bge-small FAISS index over the full MTRAG Govt corpus (49,607 passages), the same conversation history and retrieval query. Test set: 30 answerable/partial turns from MTRAG (human-written multi-turn RAG benchmark), sampled with a fixed seed. Configs, prompts and pass/fail rules were committed to the repository before any run.

**Metrics.** Input/output tokens (API usage field), cost, latency, Recall@k against gold passages, truncation rate, and RAGAS FactualCorrectness (F1, with precision/recall as diagnostics, from one judge pass) and Faithfulness, judged by `gpt-4.1-mini`.

**Pass rule (fixed in advance).** The optimized agent passes if F1 drops by at most 0.03 and Faithfulness does not drop, unless the drop is smaller than the run-to-run noise of two Baseline runs.

**Results.** One run per config, 30 turns; Δ vs Baseline with paired bootstrap 95% CI. Only Combined was pre-registered; the rest are exploratory follow-ups.

| Config | Tokens | Cost / 1k q. | Δ F1 | Δ Faithfulness | Rules |
|---|---|---|---|---|---|
| Baseline | 4,892 | $1.12 | (0.426) | (0.945) | — |
| **Combined** | −52% | $0.57 | −0.048 [−0.15, +0.05] | −0.005 [−0.08, +0.07] | fail |
| Combined + reranker | −52% | $0.56 | −0.032 [−0.10, +0.04] | +0.029 [−0.03, +0.10] | fail (F1) |
| Top-k only | −53% | $0.59 | −0.035 [−0.12, +0.05] | −0.023 [−0.08, +0.04] | fail |
| Top-k + reranker only | −53% | $0.60 | −0.008 [−0.07, +0.06] | +0.020 [−0.01, +0.06] | **pass** |
| Length only | +1% | $1.09 | −0.039 [−0.10, +0.02] | +0.002 [−0.06, +0.07] | fail (F1) |

The pre-registered Combined agent halved tokens but **failed** the quality rule. Ablations show that top-k filtering produces almost all savings (input is ~97% of tokens) and that its quality cost comes from lost evidence: in the worst case, k=4 dropped the only relevant passage and the agent answered with another city's rules. A cross-encoder reranker choosing the best 4 of 20 candidates recovered most of the loss at no token cost (but ~5.7 s of CPU time per question), and was the only configuration to pass. Output control saved little: the model was already concise (median ~97 output tokens), the extra prompt instructions cost ~90 input tokens per call, and the 200-token cap truncated some answers despite the 150-word limit.

**Validation.** *Smoke test:* 5 turns per config read by hand before the full run; it exposed inconsistent citation formats, which the evaluation now strips. *Unit tests:* 32 tests verify retrieval returns exactly k passages, tokens are logged per call, and each config changes only its own technique. *Noise:* the planned second Baseline run was not done; instead, repeated smoke-test calls showed that 0/10 answers were identical at temperature 0, so the verdict uses fixed thresholds plus bootstrap CIs, all of which include 0. *Judge hand-check:* [X of 20 judgments agreed with my own assessment; main disagreement: …].

**Lessons.** (1) Save tokens on the input side, where they are; (2) fix ranking rather than k; (3) output limits only pay off for verbose models and need a token margin above the word limit; (4) with temperature-0 non-determinism and ±0.3 per-turn swings, a 0.03 threshold needs far more than 30 turns and repeated runs. Code, logs and all per-turn scores: [repository link].
