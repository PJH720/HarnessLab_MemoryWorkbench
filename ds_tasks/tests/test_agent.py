"""Agent loop and memory contracts, with a scripted LLM (no server)."""

import json

from agent.loop import parse_answer, run_episode
from agent.memories import CueVerified, KnnSuccess
from agent.sandbox import make_workdir, run_python
from scoring import score

CARD = {"id": "c1", "question": "What is 6*7?", "answer": 42, "answer_type": "number",
        "tolerance": {"abs": 0}, "tags": ["t"], "inputs": []}


def scripted(*replies):
    it = iter(replies)
    return lambda messages, seed: (next(it), {"in": 1, "out": 1})


def test_episode_runs_code_then_answers():
    ep = run_episode(CARD, scripted("```python\nprint(6*7)\n```", "FINAL ANSWER: 42"), seed=0)
    assert ep.finished and ep.answer == 42 and ep.steps[0]["stdout"].strip() == "42"
    assert score(CARD, ep.answer)


def test_parse_answer_handles_lists_and_strings():
    assert parse_answer('["a", "b"]') == ["a", "b"]
    assert parse_answer("95250") == 95250
    assert parse_answer('"Low"') == "Low"


def test_sandbox_timeout_and_isolated_cwd():
    wd = make_workdir([])
    assert run_python("import time; time.sleep(5)", wd, timeout=1).timed_out
    assert run_python("import os; print(os.getcwd())", wd).stdout.strip().endswith(wd.name)


def test_knn_stores_only_successes():
    mem = KnnSuccess()
    good = run_episode(CARD, scripted("```python\nprint(42)\n```", "FINAL ANSWER: 42"), 0)
    bad = run_episode(CARD, scripted("FINAL ANSWER: 41"), 0)
    mem.learn(CARD, good, True, None, 0, score)
    mem.learn(CARD, bad, False, None, 0, score)
    assert mem.state()["n_items"] == 1 and mem.items[0]["code"].strip() == "print(42)"


def test_cue_stored_only_when_retry_succeeds():
    cue = json.dumps({"title": "t", "when": "w", "cue": "multiply carefully"})
    bad = run_episode(CARD, scripted("FINAL ANSWER: 41"), 0)
    passing = CueVerified()
    passing.learn(CARD, bad, False, scripted(cue, "FINAL ANSWER: 42"), 0, score)
    failing = CueVerified()
    failing.learn(CARD, bad, False, scripted(cue, "FINAL ANSWER: 40"), 0, score)
    unparsable = CueVerified()
    unparsable.learn(CARD, bad, False, scripted("no json"), 0, score)
    assert passing.state() | {"items": None} == {"items": None, "n_failures_seen": 1, "n_items": 1, "n_rejected": 0, "n_parse_failures": 0}
    assert (failing.state()["n_items"], failing.state()["n_rejected"]) == (0, 1)
    assert (unparsable.state()["n_items"], unparsable.state()["n_parse_failures"]) == (0, 1)
    for m in (passing, failing, unparsable):
        s = m.state()
        assert s["n_failures_seen"] == s["n_items"] + s["n_rejected"] + s["n_parse_failures"]
