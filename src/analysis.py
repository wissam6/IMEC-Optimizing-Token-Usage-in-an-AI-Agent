"""Step 9 analysis: results table, token-vs-quality chart, per-turn causes, hand-check export.

Usage:  python src/analysis.py
Reads results/scores_<run>.csv (from eval.py) and results/judgments_<run>.jsonl; writes to results/.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BASELINE = "baseline_run1"
# Order = how they appear in tables. 'pre-registered' marks the comparison fixed before any runs.
RUNS = {
    "combined_run1": "Combined (pre-registered)",
    "combined_rerank_run1": "Combined + reranker",
    "topk_only_run1": "Top-k only",
    "topk_rerank_only_run1": "Top-k + reranker only",
    "length_only_run1": "Length only",
}
F1_THRESHOLD, FAITH_THRESHOLD = -0.03, 0.0
N_BOOT, SEED = 5000, 0
HAND_CHECK_TASKS, HAND_CHECK_SEED = 10, 7

# Chart tokens (light surface; static PNG for the PDF).
SERIES, INK, INK_2, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


def scores(run):
    return pd.read_csv(RESULTS / f"scores_{run}.csv").set_index("task_id")


def paired_ci(diff):
    """Mean of per-turn differences with a paired bootstrap 95% CI."""
    rng = np.random.default_rng(SEED)
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(N_BOOT)]
    return diff.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


# --- 1. Results table ---------------------------------------------------------

def comparison_table():
    b = scores(BASELINE)
    b_tokens = (b.input_tokens + b.output_tokens).mean()
    rows = []
    for run, label in {BASELINE: "Baseline", **RUNS}.items():
        x = scores(run)
        row = {
            "config": label,
            "input_tokens": x.input_tokens.mean(),
            "output_tokens": x.output_tokens.mean(),
            "token_reduction_pct": 100 * (1 - (x.input_tokens + x.output_tokens).mean() / b_tokens),
            "truncated": f"{int(x.truncated.sum())}/{len(x)}",
            "recall_at_k": x.recall_at_k.mean(),
            "fc_f1": x.fc_f1.mean(), "fc_precision": x.fc_precision.mean(), "fc_recall": x.fc_recall.mean(),
            "faithfulness": x.faithfulness.mean(),
            "llm_latency_s": x.latency_s.mean(),
            # Not logged for the first runs (baseline, combined), which predate the reranker follow-up.
            "retrieval_latency_s": x["retrieval_latency_s"].mean() if "retrieval_latency_s" in x else np.nan,
        }
        if run != BASELINE:
            for m, thr in (("fc_f1", F1_THRESHOLD), ("faithfulness", FAITH_THRESHOLD)):
                d, lo, hi = paired_ci((x[m] - b[m]).values)
                row[f"d_{m}"], row[f"d_{m}_ci_low"], row[f"d_{m}_ci_high"] = d, lo, hi
                row[f"{m}_rule"] = "pass" if d >= thr else "fail"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "comparison_vs_baseline.csv", index=False, float_format="%.4f")
    return df


def write_markdown_table(df):
    lines = [
        "| Config | Input tok | Output tok | Token reduction | Truncated | Recall@k | FC F1 (Δ, 95% CI) | Faithfulness (Δ, 95% CI) | Rules |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        if r.config == "Baseline":
            f1, fa, rules, red = f"{r.fc_f1:.3f}", f"{r.faithfulness:.3f}", "—", "—"
        else:
            f1 = f"{r.fc_f1:.3f} ({r.d_fc_f1:+.3f}, [{r.d_fc_f1_ci_low:+.3f}, {r.d_fc_f1_ci_high:+.3f}])"
            fa = (f"{r.faithfulness:.3f} ({r.d_faithfulness:+.3f}, "
                  f"[{r.d_faithfulness_ci_low:+.3f}, {r.d_faithfulness_ci_high:+.3f}])")
            both = r.fc_f1_rule == "pass" and r.faithfulness_rule == "pass"
            rules = f"F1 {r.fc_f1_rule}, Faith {r.faithfulness_rule} → **{'PASS' if both else 'FAIL'}**"
            red = f"{r.token_reduction_pct:.1f}%"
        lines.append(f"| {r.config} | {r.input_tokens:,.0f} | {r.output_tokens:.1f} | {red} | {r.truncated} | "
                     f"{r.recall_at_k:.3f} | {f1} | {fa} | {rules} |")
    (RESULTS / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- 2. Chart: token reduction vs quality change -------------------------------

def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)


def chart(df):
    """Three panels sharing one row per config: token reduction | ΔF1 | ΔFaithfulness (dot = mean, bar = 95% CI)."""
    d = df[df.config != "Baseline"].iloc[::-1]  # first config at the top
    y = np.arange(len(d))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True, facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1.25, 1.25]})

    ax = axes[0]
    _style(ax)
    ax.barh(y, d.token_reduction_pct, height=0.45, color=SERIES, zorder=2)
    for yi, v in zip(y, d.token_reduction_pct):
        ax.text(max(v, 0) + 1.5, yi, f"{v:.0f}%", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_xlim(-5, 70)
    ax.set_title("Token reduction vs Baseline\n(input + output, higher = cheaper)", fontsize=9, color=INK, loc="left")
    ax.set_yticks(y, d.config, fontsize=9, color=INK)

    panels = [("d_fc_f1", "Δ FactualCorrectness F1 vs Baseline\n(pass if ≥ −0.03, dashed)", F1_THRESHOLD),
              ("d_faithfulness", "Δ Faithfulness vs Baseline\n(pass if ≥ 0)", None)]
    for ax, (m, title, thr) in zip(axes[1:], panels):
        _style(ax)
        ax.axvline(0, color=INK_2, lw=1, zorder=1)
        if thr is not None:
            ax.axvline(thr, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
        lo, hi = d[f"{m}_ci_low"], d[f"{m}_ci_high"]
        ax.hlines(y, lo, hi, color=SERIES, lw=2, alpha=0.35, zorder=2)
        ax.scatter(d[m], y, s=64, color=SERIES, edgecolor=SURFACE, linewidth=2, zorder=3)
        for yi, v in zip(y, d[m]):
            ax.annotate(f"{v:+.3f}", (v, yi), xytext=(0, 8), textcoords="offset points", ha="center",
                        fontsize=8, color=INK, zorder=4,
                        bbox={"boxstyle": "square,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"})
        ax.set_xlim(-0.17, 0.12)
        ax.set_title(title, fontsize=9, color=INK, loc="left")

    fig.text(0.01, 0.01, "30 turns, one run per config. Dot = mean per-turn change; bar = paired bootstrap 95% CI. "
             "All configs except Combined were added after the main run (exploratory).", fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(RESULTS / "token_vs_quality.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


# --- 3. Per-turn causes where an optimized config scored worse -----------------

def worse_turns():
    b = scores(BASELINE)
    turns = {t["task_id"]: t for t in map(json.loads, open(ROOT / "data" / "sample_govt.jsonl", encoding="utf-8"))}
    rows = []
    for run, label in RUNS.items():
        x = scores(run)
        for tid in x.index[(x.fc_f1 - b.fc_f1) < -0.01]:
            o, base = x.loc[tid], b.loc[tid]
            causes = []
            if o.recall_at_k < base.recall_at_k:
                causes.append("gold passage lost")
            if o.truncated:
                causes.append("truncated")
            if not o.truncated and o.fc_recall < base.fc_recall and o.answer_words < base.answer_words:
                causes.append("detail dropped")
            if not causes:
                causes.append("other (run-to-run variation / judge)")
            rows.append({
                "config": label, "task_id": tid, "question": turns[tid]["question"].strip(),
                "answerability": o.answerability, "baseline_f1": base.fc_f1, "config_f1": o.fc_f1,
                "delta_f1": o.fc_f1 - base.fc_f1, "baseline_recall_at_k": base.recall_at_k,
                "config_recall_at_k": o.recall_at_k, "truncated": o.truncated,
                "baseline_words": base.answer_words, "config_words": o.answer_words, "cause": " + ".join(causes),
            })
    df = pd.DataFrame(rows).sort_values(["config", "delta_f1"])
    df.to_csv(RESULTS / "worse_turns.csv", index=False, float_format="%.3f")
    return df


# --- 4. Hand-check export --------------------------------------------------------

def hand_check():
    """20 judgments: the same 10 random turns, judged for Baseline and for Combined."""
    rng = np.random.default_rng(HAND_CHECK_SEED)
    task_ids = sorted(scores(BASELINE).index)
    picked = set(rng.choice(task_ids, HAND_CHECK_TASKS, replace=False))
    rows = []
    for run, label in ((BASELINE, "Baseline"), ("combined_run1", "Combined")):
        s = scores(run)
        for j in map(json.loads, open(RESULTS / f"judgments_{run}.jsonl", encoding="utf-8")):
            if j["task_id"] not in picked:
                continue
            fmt = lambda claims: "\n".join(f"[{'OK' if ok else 'NO'}] {c}" for c, ok in claims)
            r = s.loc[j["task_id"]]
            rows.append({
                "task_id": j["task_id"], "config": label, "question": j["question"].strip(),
                "reference": j["reference"].strip(), "answer_scored": j["answer_scored"],
                "answer_claims (OK = supported by reference)": fmt(j.get("answer_claims", [])),
                "reference_claims (OK = covered by answer)": fmt(j.get("reference_claims", [])),
                "fc_precision": round(r.fc_precision, 2), "fc_recall": round(r.fc_recall, 2),
                "fc_f1": round(r.fc_f1, 2), "faithfulness": round(r.faithfulness, 2),
                "my_verdicts_agree (yes/partly/no)": "", "my_f1_estimate": "", "notes": "",
            })
    df = pd.DataFrame(rows).sort_values(["task_id", "config"])
    df.to_csv(RESULTS / "hand_check.csv", index=False, encoding="utf-8-sig")  # utf-8-sig opens cleanly in Excel
    return df


if __name__ == "__main__":
    table = comparison_table()
    write_markdown_table(table)
    chart(table)
    worse = worse_turns()
    hc = hand_check()
    print((RESULTS / "results_table.md").read_text(encoding="utf-8"))
    print("Causes of lower F1, per config (turns):")
    print(worse.assign(cause=worse.cause.str.split(" \\+ ")).explode("cause")
          .groupby(["config", "cause"]).size().unstack(fill_value=0).to_string())
    print(f"\nWrote: comparison_vs_baseline.csv, results_table.md, token_vs_quality.png, "
          f"worse_turns.csv ({len(worse)} rows), hand_check.csv ({len(hc)} rows)")
