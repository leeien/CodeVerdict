"""Token accounting and prompt-cache plumbing.

Two things the cost meter has to get right, because both of them fail *quietly*:
a retry whose first attempt vanishes from the bill, and a cached prompt priced
as though it were fresh. Either one makes the reported cost of a delivery a
number nobody should act on.
"""

from __future__ import annotations

import pytest
from app.services import anthropic_api, llm


class TestCarryUsage:
    def test_a_failed_attempt_is_still_billed(self):
        """The provider read the whole prompt before it timed out. Those tokens
        are on the invoice whether or not the answer was usable."""
        failed = {"tokens_in": 5000, "tokens_out": 0, "cost": 0.015, "error": "timeout"}
        kept = {"tokens_in": 5000, "tokens_out": 400, "cost": 0.021, "error": None}
        out = llm.carry_usage(kept, failed)
        assert out["tokens_in"] == 10000
        assert out["tokens_out"] == 400
        assert out["cost"] == pytest.approx(0.036)

    def test_it_returns_the_kept_result_not_the_failed_one(self):
        kept = {"text": "good", "tokens_in": 1, "tokens_out": 1, "cost": 0.0}
        out = llm.carry_usage(kept, {"text": "bad", "tokens_in": 9, "tokens_out": 0, "cost": 0.0})
        assert out is kept and out["text"] == "good"

    def test_none_and_missing_keys_are_tolerated(self):
        """Backends that report tokens but not dollars return None costs; a
        preflight failure may return no usage keys at all."""
        out = llm.carry_usage({"tokens_in": 3}, None, {}, {"cost": None, "tokens_out": 2})
        assert out["tokens_in"] == 3 and out["tokens_out"] == 2 and out["cost"] == 0

    def test_several_attempts_accumulate(self):
        out = llm.carry_usage({"tokens_in": 1, "cost": 0.1},
                              {"tokens_in": 2, "cost": 0.2},
                              {"tokens_in": 4, "cost": 0.4})
        assert out["tokens_in"] == 7 and out["cost"] == pytest.approx(0.7)


class TestCachePricing:
    """`cache_read_input_tokens` used to be added straight into `tokens_in` and
    charged at the full input rate, so a working cache and a broken one produced
    the same bill — which would have hidden the saving entirely."""

    def _usage(self, monkeypatch, *, fresh, read, write, out=100):
        class _Usage:
            input_tokens = fresh
            cache_read_input_tokens = read
            cache_creation_input_tokens = write
            output_tokens = out

        class _Block:
            type = "text"
            text = "ok"

        class _Resp:
            content = [_Block()]
            usage = _Usage()

        class _Messages:
            def create(self, **kw):
                return _Resp()

        class _Client:
            messages = _Messages()

        import sys
        import types
        fake = types.ModuleType("anthropic")
        fake.Anthropic = lambda **kw: _Client()
        fake.APIStatusError = type("APIStatusError", (Exception,), {})
        monkeypatch.setitem(sys.modules, "anthropic", fake)
        monkeypatch.setattr(anthropic_api.settings, "anthropic_api_key", "k")
        return anthropic_api.chat("sys", "user", model="claude-sonnet-5")

    def test_a_cache_read_costs_a_tenth_of_fresh_input(self, monkeypatch):
        fresh_only = self._usage(monkeypatch, fresh=10_000, read=0, write=0)
        mostly_cached = self._usage(monkeypatch, fresh=1_000, read=9_000, write=0)
        # Same 10k tokens of input either way…
        assert fresh_only["tokens_in"] == mostly_cached["tokens_in"] == 10_000
        # …but sonnet-tier input is $3/M, so 10k fresh is $0.030 while
        # 1k fresh + 9k cache-read (at 10%) is 1.9k effective = $0.0057.
        # Output is 100 tokens at $15/M = $0.0015 in both.
        assert fresh_only["cost"] == pytest.approx(0.0315)
        assert mostly_cached["cost"] == pytest.approx(0.0072)
        assert mostly_cached["cost"] < fresh_only["cost"] / 4

    def test_a_cache_write_costs_more_than_fresh(self, monkeypatch):
        """Writing the entry carries a premium — worth it only because the next
        juror reads it. Pricing it as free would make the first call look wrong."""
        fresh = self._usage(monkeypatch, fresh=10_000, read=0, write=0)
        writing = self._usage(monkeypatch, fresh=0, read=0, write=10_000)
        assert writing["cost"] > fresh["cost"]

    def test_usage_is_reported_so_a_dead_cache_is_visible(self, monkeypatch):
        res = self._usage(monkeypatch, fresh=1, read=500, write=200)
        assert res["cached_read"] == 500 and res["cached_write"] == 200


class TestCachePrefixPlumbing:
    def test_a_short_prefix_is_not_split_into_a_cache_block(self, monkeypatch):
        """Anthropic will not cache below its minimum, so a breakpoint there is
        pure write-premium for nothing. Short prefixes just concatenate."""
        captured = {}

        class _Messages:
            def create(self, **kw):
                captured.update(kw)
                return type("R", (), {"content": [], "usage": type("U", (), {
                    "input_tokens": 1, "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0, "output_tokens": 1})()})()

        import sys
        import types
        fake = types.ModuleType("anthropic")
        fake.Anthropic = lambda **kw: type("C", (), {"messages": _Messages()})()
        fake.APIStatusError = type("APIStatusError", (Exception,), {})
        monkeypatch.setitem(sys.modules, "anthropic", fake)
        monkeypatch.setattr(anthropic_api.settings, "anthropic_api_key", "k")

        anthropic_api.chat("sys", "tail", model="claude-sonnet-5", cache_prefix="short")
        assert captured["messages"][0]["content"] == "shorttail"
        assert isinstance(captured["system"], str)

    def test_a_long_prefix_becomes_its_own_cached_block_before_the_tail(self, monkeypatch):
        captured = {}

        class _Messages:
            def create(self, **kw):
                captured.update(kw)
                return type("R", (), {"content": [], "usage": type("U", (), {
                    "input_tokens": 1, "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0, "output_tokens": 1})()})()

        import sys
        import types
        fake = types.ModuleType("anthropic")
        fake.Anthropic = lambda **kw: type("C", (), {"messages": _Messages()})()
        fake.APIStatusError = type("APIStatusError", (Exception,), {})
        monkeypatch.setitem(sys.modules, "anthropic", fake)
        monkeypatch.setattr(anthropic_api.settings, "anthropic_api_key", "k")

        prefix = "CASE FILE " * 2000
        anthropic_api.chat("sys", "the charge", model="claude-sonnet-5", cache_prefix=prefix)
        blocks = captured["messages"][0]["content"]
        assert isinstance(blocks, list) and len(blocks) == 2
        assert blocks[0]["text"] == prefix
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        # The variable tail must come AFTER the breakpoint, or it invalidates it.
        assert blocks[1]["text"] == "the charge"
        assert "cache_control" not in blocks[1]


class TestNonAnthropicProvidersKeepPrefixFirst:
    def test_openai_style_providers_get_prefix_then_tail(self, monkeypatch):
        """OpenAI/Groq/Gemini/DeepSeek cache automatically on the longest matching
        prefix, so the only thing that matters is that stable text stays first."""
        captured = {}

        def _fake(system, user, *, provider, model, timeout=180, json_mode=False,
                  messages=None):
            captured["user"] = user
            return {"text": "", "tokens_in": 0, "tokens_out": 0, "cost": 0.0, "error": None}

        monkeypatch.setattr(llm.openai_agent, "chat", _fake)
        monkeypatch.setattr(llm.providers, "kind", lambda p: "openai")
        llm.chat("sys", "TAIL", provider="groq", model="m", cache_prefix="PREFIX")
        assert captured["user"] == "PREFIXTAIL"


class TestDevTelemetry:
    """Dev is ~84% of a delivery's tokens and that spend is turns, not prompt
    size. These assert the numbers that explain it actually get captured."""

    def test_tool_calls_and_read_paths_are_tallied(self):
        from app.services import claude_agent

        res = {"tools": {}, "read_paths": []}
        claude_agent._record_tool(res, "Read", {"file_path": "rich/cells.py"})
        claude_agent._record_tool(res, "Read", {"file_path": "rich/text.py"})
        claude_agent._record_tool(res, "Bash", {"command": "pytest"})
        assert res["tools"] == {"Read": 2, "Bash": 1}
        assert res["read_paths"] == ["rich/cells.py", "rich/text.py"]

    def test_a_nameless_or_odd_tool_call_does_not_crash_the_run(self):
        from app.services import claude_agent

        res = {"tools": {}, "read_paths": []}
        claude_agent._record_tool(res, None, {})
        claude_agent._record_tool(res, "Read", "not-a-dict")
        assert res["tools"] == {"Read": 1} and res["read_paths"] == []

    def test_read_paths_are_bounded(self):
        """A pathological agent must not turn telemetry into a memory leak."""
        from app.services import claude_agent

        res = {"tools": {}, "read_paths": []}
        for i in range(500):
            claude_agent._record_tool(res, "Read", {"file_path": f"f{i}.py"})
        assert len(res["read_paths"]) == 200

    def test_a_re_read_of_an_injected_file_is_reported_as_waste(self):
        from app.services import orchestrator

        events = []
        verified = "pins…\n\n--- rich/cells.py ---\ncode here\n\n--- rich/text.py ---\nmore"
        orchestrator._report_dev_efficiency(
            {"turns": 12, "tools": {"Read": 3, "Edit": 1},
             "read_paths": ["rich/cells.py", "rich/other.py"]},
            verified, lambda lvl, msg: events.append((lvl, msg)))
        assert any("12 turn(s)" in m for _, m in events)
        warned = [m for lvl, m in events if lvl == "warn"]
        assert warned and "re-read 1 of 2" in warned[0]

    def test_injection_that_held_is_reported_too(self):
        from app.services import orchestrator

        events = []
        orchestrator._report_dev_efficiency(
            {"turns": 4, "tools": {"Edit": 2}, "read_paths": []},
            "--- rich/cells.py ---\ncode", lambda lvl, msg: events.append((lvl, msg)))
        assert any("no wasted reads" in m for _, m in events)
        assert not [m for lvl, m in events if lvl == "warn"]

    def test_a_backend_that_reports_nothing_stays_silent(self):
        """Never invent a number for a backend that doesn't supply one."""
        from app.services import orchestrator

        events = []
        orchestrator._report_dev_efficiency({}, "--- a.py ---", lambda l, m: events.append((l, m)))
        assert events == []


class TestQuotaRepliesAreNotAnswers:
    """The CLI reports an exhausted plan by *answering* with it: exit 0, no
    is_error, quota notice as the result text. It was stored as the agent's own
    words — observed live, where a session-limit notice became a PM turn in the
    scope history and 34,466 tokens were billed for it."""

    def test_a_session_limit_notice_is_detected(self):
        from app.services.claude_agent import quota_error

        assert quota_error("You've hit your session limit · resets 2:50am (Asia/Kolkata)")
        assert quota_error("Claude usage limit reached. Upgrade to increase your usage limit.")

    def test_a_real_answer_mentioning_limits_is_not_flagged(self):
        """A long reply that discusses rate limiting is a real reply."""
        from app.services.claude_agent import quota_error

        essay = ("The handler retries on 429s. Note the session limit reached path "
                 "is distinct from the per-minute cap, and we surface it to callers. ") * 4
        assert not quota_error(essay)
        assert not quota_error("The diff looks correct; approving.")
        assert not quota_error("")


class TestScopingCostIsVisible:
    """PM clarify turns are billed before any ticket exists, so they have no
    AgentRun to be found under and `/costs` omitted them entirely — including
    whole scopes that never reached tickets."""

    def test_pm_scoping_tokens_appear_in_the_breakdown(self, db):
        from app.core import costs
        from app.models import ScopeSession

        s = ScopeSession(repo_id=1, title="clarify-heavy scope", pm_tokens_input=100_000,
                         pm_tokens_output=4_000, pm_cost_usd=0.42)
        db.add(s)
        db.commit()
        out = costs.breakdown(db)
        mine = [x for x in out["scopes"] if x["session_id"] == s.id]
        assert mine, "a scope with only PM spend must still be reported"
        assert mine[0]["scoping"]["tokens_in"] == 100_000
        assert mine[0]["tokens_in"] >= 100_000
        assert out["totals"]["cost"] >= 0.42

    def test_a_scope_with_no_spend_is_not_invented(self, db):
        from app.core import costs
        from app.models import ScopeSession

        s = ScopeSession(repo_id=1, title="untouched")
        db.add(s)
        db.commit()
        assert not [x for x in costs.breakdown(db)["scopes"] if x["session_id"] == s.id]


class TestPlanRendering:
    def test_a_structured_test_plan_is_not_dumped_as_a_raw_dict(self):
        from app.cli.render import _plan_item_lines

        lines = _plan_item_lines({"files": ["tests/test_text.py"],
                                  "new_cases": ["test_a", "test_b"]})
        assert lines == ["files: tests/test_text.py", "new cases: test_a, test_b"]
        assert not any("{" in ln or "'" in ln for ln in lines)

    def test_plain_strings_and_lists_still_render(self):
        from app.cli.render import _plan_item_lines

        assert _plan_item_lines("rich/style.py") == ["rich/style.py"]
        assert _plan_item_lines(["a", "b"]) == ["a, b"]
        assert _plan_item_lines({}) == ["{}"]
