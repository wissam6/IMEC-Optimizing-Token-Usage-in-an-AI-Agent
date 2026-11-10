"""Agent and config tests. No API calls: a fake retriever and a fake LLM replace the real ones."""
import json
from types import SimpleNamespace as NS

import pytest

import agent
from config import BASE_SYSTEM_PROMPT, CONFIGS, system_prompt

TURN = {
    "task_id": "conv1<::>2",
    "question": "What about earthquakes?",
    "rewritten_question": "What should I keep in a safe room for earthquakes?",
    "history": [
        {"speaker": "user", "text": "What items should I keep in a safe room?"},
        {"speaker": "agent", "text": "Water, food and a first aid kit."},
    ],
    "gold_passage_ids": ["p0", "p7"],
}


class FakeRetriever:
    def search(self, query, k, rerank_candidates=None):
        self.last_query, self.last_rerank = query, rerank_candidates
        return [{"id": f"p{i}", "text": f"passage {i}", "score": 1.0 - i / 100} for i in range(k)]


def fake_llm(content="An answer.\nSources: [1]", finish="stop", prompt_tokens=1000, completion_tokens=20):
    def call(messages, max_tokens):
        call.messages, call.max_tokens = messages, max_tokens
        return NS(
            model="fake-model",
            choices=[NS(finish_reason=finish, message=NS(content=content))],
            usage=NS(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                     completion_tokens_details=NS(reasoning_tokens=0),
                     prompt_tokens_details=NS(cached_tokens=0)),
        )
    return call


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Fake retriever + logs written to a temp folder, never to results/."""
    monkeypatch.setattr(agent, "_retriever", FakeRetriever())
    monkeypatch.setattr(agent, "LOG_DIR", tmp_path)
    return tmp_path


# --- Logging ----------------------------------------------------------------

def test_token_counts_are_logged_one_line_per_call(isolated):
    agent.run(TURN, "baseline", run_name="t", call_llm=fake_llm(prompt_tokens=1234, completion_tokens=56))
    agent.run(TURN, "combined", run_name="t", call_llm=fake_llm())
    lines = (isolated / "t.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["input_tokens"] == 1234 and first["output_tokens"] == 56
    for key in ("task_id", "config", "retrieved_ids", "gold_ids", "finish_reason", "latency_s", "answer", "reasoning_tokens"):
        assert key in first


def test_truncation_is_recorded(isolated):
    r = agent.run(TURN, "combined", run_name="t", call_llm=fake_llm(finish="length"))
    assert r["finish_reason"] == "length"


def test_none_content_becomes_empty_string(isolated):
    r = agent.run(TURN, "combined", run_name="t", call_llm=fake_llm(content=None))
    assert r["answer"] == ""


# --- Each config changes only what it should ---------------------------------

def test_baseline_and_combined_values():
    assert CONFIGS["baseline"] == {"top_k": 10, "max_tokens": 1024, "length_control": False}
    assert CONFIGS["combined"] == {"top_k": 4, "max_tokens": 200, "length_control": True}


def test_config_controls_k_and_max_tokens(isolated):
    for name, cfg in CONFIGS.items():
        llm = fake_llm()
        r = agent.run(TURN, name, run_name="t", call_llm=llm)
        assert len(r["retrieved_ids"]) == cfg["top_k"]
        assert llm.max_tokens == cfg["max_tokens"]


def test_only_system_prompt_and_context_differ_between_configs(isolated):
    """Same retrieval query and same history in every config; the system prompt differs only by the constraints."""
    msgs = {}
    for name in ("baseline", "combined"):
        llm = fake_llm()
        agent.run(TURN, name, run_name="t", call_llm=llm)
        msgs[name] = llm.messages
        assert agent._retriever.last_query == TURN["rewritten_question"]

    base, comb = msgs["baseline"], msgs["combined"]
    assert len(base) == len(comb) == 1 + len(TURN["history"]) + 1
    assert base[1:-1] == comb[1:-1]                              # identical history
    assert comb[0]["content"].startswith(base[0]["content"])    # constraints are only appended
    assert base[-1]["content"].count("\n[") == 10 and comb[-1]["content"].count("\n[") == 4
    assert base[-1]["content"].endswith("Question: What about earthquakes?")


def test_rerank_config_differs_from_combined_only_by_reranking(isolated):
    extra = {k: v for k, v in CONFIGS["combined_rerank"].items() if CONFIGS["combined"].get(k) != v}
    assert extra == {"rerank_candidates": 20}
    r = agent.run(TURN, "combined_rerank", run_name="t", call_llm=fake_llm())
    assert agent._retriever.last_rerank == 20 and r["rerank_candidates"] == 20
    assert "retrieval_latency_s" in r
    agent.run(TURN, "combined", run_name="t", call_llm=fake_llm())
    assert agent._retriever.last_rerank is None  # no reranking in the original configs


@pytest.mark.parametrize("name, expected_change", [
    ("topk_only", {"top_k": 4}),
    ("length_only", {"max_tokens": 200, "length_control": True}),
    ("topk_rerank_only", {"top_k": 4, "rerank_candidates": 20}),
])
def test_each_ablation_changes_only_its_technique(name, expected_change):
    base = CONFIGS["baseline"]
    changed = {k: v for k, v in CONFIGS[name].items() if base.get(k) != v}
    assert changed == expected_change


def test_history_roles_are_mapped():
    msgs = agent.build_messages(TURN, [], CONFIGS["baseline"])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]


# --- System prompt ------------------------------------------------------------

def test_combined_prompt_contains_constraints():
    p = system_prompt(CONFIGS["combined"])
    assert "150 words" in p
    assert "Sources:" in p
    assert "direct answer first" in p
    assert "filler" in p.lower()


def test_baseline_prompt_has_no_constraints():
    p = system_prompt(CONFIGS["baseline"])
    assert p == BASE_SYSTEM_PROMPT
    assert "150 words" not in p and "Sources:" not in p
