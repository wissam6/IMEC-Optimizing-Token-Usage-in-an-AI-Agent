"""Run configs over sampled turns; every call is logged to results/logs/<run_name>.jsonl by agent.run.

Usage:
  python src/run_experiment.py <run_name> <config>[,<config>...] [n_turns]

Examples:
  python src/run_experiment.py smoke baseline,combined 5      # smoke test: first 5 turns
  python src/run_experiment.py baseline_run1 baseline         # all 30 turns
"""
import sys

from agent import LOG_DIR, load_turns, run
from config import CONFIGS


def main(run_name, config_names, n_turns=None):
    log_file = LOG_DIR / f"{run_name}.jsonl"
    if log_file.exists():
        sys.exit(f"{log_file} already exists; pick a new run name so logs are never mixed.")
    turns = load_turns()[:n_turns]
    for i, turn in enumerate(turns):
        for name in config_names:
            r = run(turn, name, run_name=run_name)
            print(f"[{i + 1}/{len(turns)}] {name:<9} in={r['input_tokens']:>6} out={r['output_tokens']:>4} "
                  f"reasoning={r['reasoning_tokens']} finish={r['finish_reason']} {r['latency_s']:.1f}s")
    print(f"Log: {log_file}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    names = sys.argv[2].split(",")
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        sys.exit(f"Unknown config(s): {unknown}. Choose from {list(CONFIGS)}")
    main(sys.argv[1], names, int(sys.argv[3]) if len(sys.argv) == 4 else None)
