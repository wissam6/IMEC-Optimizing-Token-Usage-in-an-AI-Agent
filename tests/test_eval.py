"""Eval helper tests: stripping the Sources line and handling empty answers."""
import pytest

from eval import answer_for_scoring, is_empty


# Formats actually produced in the smoke test (results/logs/smoke.jsonl).
@pytest.mark.parametrize("answer", [
    "Federal sites use .gov.  \nSources: 1, 2, 3, 4",
    "Call (800) 320-8077 before 12 noon.  \nSources: [1], [1]",
    "The context does not say.  \n\nSources: [1], [3]",
    "Use HTTPS.\n**Sources:** [2]",
    "Use HTTPS.\nsource: [2]",
])
def test_sources_line_is_removed(answer):
    out = answer_for_scoring(answer)
    assert "ource" not in out
    assert out == out.strip() and out


def test_answer_without_sources_is_unchanged():
    text = "Climate change can contribute to floods and droughts.\n\nSo, it is not the sole cause."
    assert answer_for_scoring(text) == text


def test_word_sources_inside_answer_is_kept():
    text = "Sources: of funding include taxes."  # a Sources-like first line is answer text, not a citation line
    assert answer_for_scoring(text) == text


@pytest.mark.parametrize("answer", [None, "", "   ", "\n\n"])
def test_empty_answers(answer):
    assert answer_for_scoring(answer) == ""
    assert is_empty(answer)


def test_non_empty_answer_is_not_empty():
    assert not is_empty("Yes.\nSources: [1]")
