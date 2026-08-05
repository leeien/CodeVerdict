"""Reviewer reachback: QA and the jurors querying the index before they decide.

The failure this addresses is not an ignorant reviewer, it is a confident one.
A diff shows what changed, never what depends on it, so "this breaks callers",
"this doesn't match the pattern" and "this case isn't covered" were all being
asserted about code the reviewer could not see — and a wrong blocking finding
costs a full paid Dev+QA+Review round.

What is asserted here is mostly the BOUNDS, because an unbounded evidence loop
turns a reviewer into a second Dev agent:

  * a reviewer that asks gets exactly one follow-up, never a loop;
  * a reviewer that already voted is not asked again (paying twice for the same
    model's opinion on the same evidence buys nothing);
  * a reviewer that does not ask costs exactly what it cost before;
  * usage from both calls is billed, so the panel's real cost stays visible.
"""

from __future__ import annotations

import json

import pytest
from app.config import settings
from app.models import Judge
from app.services.jury import panel
from app.services.knowledge import tools


@pytest.fixture(autouse=True)
def stub_tools(monkeypatch):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(tools, "call",
                        lambda repo, cwd, name, arg: calls.append((name, arg))
                        or f"ANSWER({name} {arg})")
    monkeypatch.setattr(settings, "jury_tool_calls", 3)
    return calls


def judge():
    return Judge(id=1, name="Reliability", persona="reliability", enabled=True,
                 position=0, provider="groq", model="m")


def sequence(*texts):
    """Stub the juror's model with a fixed sequence of replies."""
    queue = list(texts)
    seen: list[str] = []

    def _call(system, user, provider, model, workdir="", cache_prefix=""):
        # Record what the model actually saw: the shared case file is passed
        # separately so it can be cached, but it is still part of the prompt.
        seen.append(cache_prefix + user)
        return {"text": queue.pop(0) if queue else "", "tokens_in": 7, "tokens_out": 3,
                "cost": 0.002, "error": None}
    return _call, seen


VERDICT = json.dumps({"verdict": "APPROVE", "summary": "checked it", "findings": []})


class TestJurorReachback:
    def test_a_juror_that_asks_is_answered_and_called_again(self, monkeypatch, stub_tools):
        call, seen = sequence("<<<CALLERS cell_len>>>", VERDICT)
        monkeypatch.setattr(panel, "_call", call)
        op = panel.poll_judge(judge(), {"diff": "d", "task_key": "t", "title": "T",
                                        "criteria": []}, workdir="/tmp/repo")
        assert stub_tools == [("callers", "cell_len")]
        assert "ANSWER(callers cell_len)" in seen[1]
        assert op.verdict == "APPROVE" and not op.error

    def test_the_follow_up_forbids_asking_again(self, monkeypatch, stub_tools):
        # One follow-up, never a loop — the bound is the whole safety property.
        call, seen = sequence("<<<CALLERS x>>>", "<<<CALLERS y>>>")
        monkeypatch.setattr(panel, "_call", call)
        op = panel.poll_judge(judge(), {"diff": "d", "task_key": "t", "title": "T",
                                        "criteria": []}, workdir="/tmp/repo")
        assert len(seen) == 2, "exactly two model calls"
        assert "Do not request more lookups" in seen[1]
        assert stub_tools == [("callers", "x")], "the second request is not run"
        assert op.error, "a juror that never voted abstains loudly"

    def test_usage_from_both_calls_is_billed(self, monkeypatch, stub_tools):
        call, _ = sequence("<<<LOOKUP x>>>", VERDICT)
        monkeypatch.setattr(panel, "_call", call)
        op = panel.poll_judge(judge(), {"diff": "d", "task_key": "t", "title": "T",
                                        "criteria": []}, workdir="/tmp/repo")
        # A panel whose real cost is hidden is a panel the operator can't judge.
        assert op.tokens_in == 14 and op.cost == pytest.approx(0.004)

    def test_a_juror_that_already_voted_is_not_re_asked(self, monkeypatch, stub_tools):
        """Findings mean it decided. Re-asking buys the same opinion twice."""
        voted = json.dumps({"verdict": "REQUEST_CHANGES", "summary": "s",
                            "findings": [{"title": "bug", "evidence": "line",
                                          "severity": "high", "confidence": 0.9}]})
        call, seen = sequence(voted + "\n<<<CALLERS x>>>", VERDICT)
        monkeypatch.setattr(panel, "_call", call)
        op = panel.poll_judge(judge(), {"diff": "d", "task_key": "t", "title": "T",
                                        "criteria": []}, workdir="/tmp/repo")
        assert len(seen) == 1 and stub_tools == []
        assert op.verdict == "REQUEST_CHANGES"

    def test_a_juror_that_does_not_ask_costs_one_call(self, monkeypatch, stub_tools):
        call, seen = sequence(VERDICT)
        monkeypatch.setattr(panel, "_call", call)
        op = panel.poll_judge(judge(), {"diff": "d", "task_key": "t", "title": "T",
                                        "criteria": []}, workdir="/tmp/repo")
        assert len(seen) == 1 and op.verdict == "APPROVE" and stub_tools == []

    def test_reachback_can_be_switched_off(self, monkeypatch, stub_tools):
        monkeypatch.setattr(settings, "jury_tool_calls", 0)
        call, seen = sequence("<<<CALLERS x>>>", VERDICT)
        monkeypatch.setattr(panel, "_call", call)
        panel.poll_judge(judge(), {"diff": "d", "task_key": "t", "title": "T",
                                   "criteria": []}, workdir="/tmp/repo")
        assert len(seen) == 1 and stub_tools == []


class TestRequestProtocol:
    def test_every_tool_is_parseable_from_a_reply(self):
        text = " ".join(f"<<<{t.upper()} arg{i}>>>" for i, t in enumerate(tools.TOOL_NAMES))
        assert [n for n, _ in tools.parse_requests(text)] == list(tools.TOOL_NAMES)

    def test_an_empty_argument_is_not_a_request(self):
        assert tools.parse_requests("<<<LOOKUP >>>") == []

    def test_prose_is_not_a_request(self):
        assert tools.parse_requests("I would look up the callers of cell_len.") == []

    def test_requests_are_capped(self, stub_tools):
        many = [("lookup", f"s{i}") for i in range(10)]
        tools.run_requests("repo", "/tmp/x", many, limit=2)
        assert len(stub_tools) == 2

    def test_no_requests_render_to_nothing(self):
        assert tools.run_requests("repo", "/tmp/x", []) == ""

    def test_the_offer_names_the_configured_budget(self, monkeypatch):
        monkeypatch.setattr(settings, "jury_tool_calls", 5)
        assert "5 request block(s)" in tools.evidence_block()
