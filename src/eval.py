"""Evaluation: RAGAS Faithfulness and FactualCorrectness, Recall@k, token/cost/latency summaries.

The RAGAS part is implemented in step 8. The helpers below are used (and unit-tested) already.
"""
import re

# A "Sources: ..." line in any of the formats seen in the smoke test:
# "Sources: [1], [3]", "Sources: 1, 2, 3, 4", "Sources: [1], [1]", also "**Sources:**".
_SOURCES_LINE = re.compile(r"\s*\**sources?\**\s*:", re.IGNORECASE)


def answer_for_scoring(answer):
    """The part of a response that RAGAS should judge: the answer without the Sources line.

    Returns "" for a missing or blank answer.
    """
    if not answer or not answer.strip():
        return ""
    lines = answer.strip().splitlines()
    # Cut at the last Sources line, but only if answer text comes before it.
    for i in range(len(lines) - 1, 0, -1):
        if _SOURCES_LINE.match(lines[i]):
            return "\n".join(lines[:i]).strip()
    return answer.strip()


def is_empty(answer):
    """Empty answers are not sent to the judge; they are scored 0 on every quality metric (a failure)."""
    return answer_for_scoring(answer) == ""
