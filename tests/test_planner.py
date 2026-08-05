"""The Planner stage (services/planner.py).

The Planner is the stage the research says carries the difference between a
multi-agent pipeline and a single-agent one, so what is asserted here is not
"it produces JSON" but the three properties that make it worth its cost:

  * it is BOUNDED — a planner that retrieves forever never plans, so the last
    round is forced to commit;
  * its output is VERIFIED — every symbol it names is resolved against the real
    repository, and one that resolves nowhere is marked as new rather than
    silently pointed at whatever matched first;
  * it FAILS OPEN — no plan is a pipeline that runs without one (the research's
    no-planner ablation), never a pipeline that stops.

The dotted-symbol resolution rules it inherited from ticket grounding have their
own regression tests in test_grounding_and_jury_context.py.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from app.config import settings
from app.services import planner


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "cells.py").write_text(
        "def cell_len(text):\n    return len(text)\n\n\nclass Table:\n    pass\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_cells.py").write_text(
        "from cells import cell_len\n\n\ndef test_cell_len():\n    assert cell_len('a') == 1\n")
    for cmd in (["init", "-q"], ["add", "-A"],
                ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"]):
        subprocess.run(["git", *cmd], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


@pytest.fixture(autouse=True)
def no_graph(monkeypatch):
    """Default to the no-binary tier: ripgrep + symbol map. It is what a plain
    install runs on, so it is what the defaults should be tested against."""
    monkeypatch.setattr(planner.graph, "available", lambda: False)
    monkeypatch.setattr(planner.graph, "bootstrap_text", lambda url: "")
    monkeypatch.setattr(planner.symbol_map, "load", lambda url: None)


def replies(*payloads):
    """Stub the Planner's model with a fixed sequence of decisions."""
    queue = list(payloads)

    def _ask(system, user, cache_prefix=""):
        body = queue.pop(0) if queue else {"action": "plan", "plan": None}
        return {"text": json.dumps(body), "tokens_in": 10, "tokens_out": 5,
                "cost": 0.001, "error": None}
    return _ask


PLAN_STEP = {"intent": "Count printable width without ansi codes",
             "edit_kind": "modify", "files": ["cells.py"],
             "symbols": ["cells.py::cell_len"], "why": "it is the width function",
             "verify": ["a colored string measures the same as a plain one"]}


class TestTheLoop:
    def test_it_plans_without_retrieving_when_it_already_knows_enough(self, repo, monkeypatch):
        monkeypatch.setattr(planner, "_ask", replies(
            {"action": "plan", "plan": {"summary": "fix the width", "steps": [PLAN_STEP]}}))
        out = planner.plan("repo", str(repo), {"summary": "borders", "acceptance_criteria": []})
        assert out["rounds"] == 0
        assert out["plan"]["steps"][0]["intent"].startswith("Count printable")

    def test_retrieval_rounds_feed_the_next_decision(self, repo, monkeypatch):
        seen: list[str] = []
        monkeypatch.setattr(planner.knowledge_retriever, "retrieve_context",
                            lambda url, q, **k: f"HITS FOR {q}")
        monkeypatch.setattr(planner, "_ask", replies(
            {"action": "retrieve", "queries": ["measure cell width"]},
            {"action": "plan", "plan": {"summary": "s", "steps": [PLAN_STEP]}}))
        real_ask = planner._ask

        def spy(system, user, cache_prefix=""):
            seen.append(cache_prefix + user)
            return real_ask(system, user, cache_prefix)
        monkeypatch.setattr(planner, "_ask", spy)
        out = planner.plan("repo", str(repo), {"summary": "borders"})
        assert out["retrieved"] == ["measure cell width"]
        assert "HITS FOR measure cell width" in seen[1], "round 2 must see round 1's results"

    def test_the_last_round_is_forced_to_commit(self, repo, monkeypatch):
        """A planner that keeps retrieving never plans. The cap is the contract."""
        monkeypatch.setattr(settings, "planner_max_rounds", 2)
        monkeypatch.setattr(planner.knowledge_retriever, "retrieve_context",
                            lambda *a, **k: "hits")
        prompts: list[str] = []
        calls = {"n": 0}

        def _ask(system, user, cache_prefix=""):
            prompts.append(cache_prefix + user)
            calls["n"] += 1
            # Always asks to retrieve; only the forced final round yields a plan.
            body = ({"action": "plan", "plan": {"summary": "s", "steps": [PLAN_STEP]}}
                    if "LAST round" in user else {"action": "retrieve", "queries": ["q"]})
            return {"text": json.dumps(body), "tokens_in": 1, "tokens_out": 1,
                    "cost": 0.0, "error": None}
        monkeypatch.setattr(planner, "_ask", _ask)
        out = planner.plan("repo", str(repo), {"summary": "x"})
        assert calls["n"] == 3, "two retrieval rounds, then one forced decision"
        assert out["plan"]["steps"], "the forced round must still produce a plan"

    def test_a_retrieval_that_finds_nothing_says_so(self, repo, monkeypatch):
        monkeypatch.setattr(planner.knowledge_retriever, "retrieve_context", lambda *a, **k: "")
        prompts: list[str] = []

        def _ask(system, user, cache_prefix=""):
            prompts.append(cache_prefix + user)
            body = ({"action": "retrieve", "queries": ["nothing"]} if len(prompts) == 1
                    else {"action": "plan", "plan": {"summary": "s", "steps": [PLAN_STEP]}})
            return {"text": json.dumps(body), "tokens_in": 0, "tokens_out": 0,
                    "cost": 0.0, "error": None}
        monkeypatch.setattr(planner, "_ask", _ask)
        planner.plan("repo", str(repo), {"summary": "x"})
        # Silence would read as "nothing to find"; the model must be told the
        # lookup came back empty so it changes tack rather than repeating it.
        assert "returned nothing" in prompts[1]

    def test_tool_calls_go_through_the_shared_dispatcher(self, repo, monkeypatch):
        called: list[tuple] = []
        monkeypatch.setattr(planner.kb_tools, "call",
                            lambda repo_, cwd, name, arg: called.append((name, arg)) or "RESULT")
        monkeypatch.setattr(planner.knowledge_retriever, "retrieve_context", lambda *a, **k: "")
        monkeypatch.setattr(planner, "_ask", replies(
            {"action": "retrieve", "queries": [], "tools": [["lookup", "cell_len"],
                                                           ["expand", "Table"]]},
            {"action": "plan", "plan": {"summary": "s", "steps": [PLAN_STEP]}}))
        planner.plan("repo", str(repo), {"summary": "x"})
        assert called == [("lookup", "cell_len"), ("expand", "Table")]

    def test_usage_is_accumulated_across_rounds(self, repo, monkeypatch):
        monkeypatch.setattr(planner.knowledge_retriever, "retrieve_context", lambda *a, **k: "h")
        monkeypatch.setattr(planner, "_ask", replies(
            {"action": "retrieve", "queries": ["q"]},
            {"action": "plan", "plan": {"summary": "s", "steps": [PLAN_STEP]}}))
        out = planner.plan("repo", str(repo), {"summary": "x"})
        assert out["tokens_in"] == 20 and out["cost"] == pytest.approx(0.002)


class TestFailingOpen:
    def test_a_model_that_returns_no_plan_does_not_stop_the_pipeline(self, repo, monkeypatch):
        monkeypatch.setattr(planner, "_ask", replies({"action": "plan", "plan": None}))
        out = planner.plan("repo", str(repo), {"summary": "x"})
        assert out["plan"] == {} and out["error"]

    def test_unparseable_output_is_an_empty_plan_not_an_exception(self, repo, monkeypatch):
        monkeypatch.setattr(planner, "_ask", lambda s, u, cache_prefix="": {
            "text": "I think we should probably look at cells.py", "tokens_in": 0,
            "tokens_out": 0, "cost": 0.0, "error": None})
        assert planner.plan("repo", str(repo), {"summary": "x"})["plan"] == {}

    def test_a_provider_error_is_reported_not_raised(self, repo, monkeypatch):
        monkeypatch.setattr(planner, "_ask", lambda s, u, cache_prefix="": {
            "text": "", "tokens_in": 0, "tokens_out": 0, "cost": 0.0,
            "error": "429 rate limited"})
        out = planner.plan("repo", str(repo), {"summary": "x"})
        assert out["plan"] == {} and "429" in out["error"]


class TestNormalization:
    def test_steps_are_numbered_and_capped(self):
        raw = {"steps": [{"intent": f"step {i}"} for i in range(10)]}
        steps = planner._normalize(raw)["steps"]
        assert len(steps) == 6 and [s["id"] for s in steps] == [1, 2, 3, 4, 5, 6]

    def test_an_unknown_edit_kind_falls_back_to_modify(self):
        out = planner._normalize({"steps": [{"intent": "x", "edit_kind": "refactor-everything"}]})
        assert out["steps"][0]["edit_kind"] == "modify"

    def test_a_step_with_no_intent_is_dropped(self):
        out = planner._normalize({"steps": [{"files": ["a.py"]}, {"intent": "real"}]})
        assert [s["intent"] for s in out["steps"]] == ["real"]

    def test_malformed_fields_do_not_raise(self):
        out = planner._normalize({"summary": None, "steps": "not a list", "risks": {"a": 1},
                                  "tests": "nope"})
        assert out["steps"] == [] and out["risks"] == [] and out["tests"]["files"] == []


class TestVerification:
    def test_a_real_symbol_is_pinned_to_its_definition_file(self, repo):
        plan = planner.verify_plan("repo", str(repo), planner._normalize(
            {"steps": [{"intent": "x", "files": [], "symbols": ["cell_len"]}]}))
        step = plan["steps"][0]
        assert step["symbols"] == ["cells.py::cell_len"]
        assert step["files"] == ["cells.py"], "the verified file is added, not guessed"

    def test_an_invented_symbol_is_marked_new_rather_than_mislocated(self, repo):
        plan = planner.verify_plan("repo", str(repo), planner._normalize(
            {"steps": [{"intent": "x", "symbols": ["compute_width_v2"]}]}))
        assert "(new — not in repo yet" in plan["steps"][0]["symbols"][0]
        assert plan["unresolved_symbols"] == 1

    def test_a_symbol_that_is_used_but_not_defined_is_left_alone(self, repo):
        # It appears in the repo but nothing defines it — a usage, a string, a
        # config key. Inventing a definition site for it would be a fabrication.
        (repo / "cells.py").write_text(
            (repo / "cells.py").read_text(encoding="utf-8") + "\nprint(SOME_EXTERNAL_FLAG)\n")
        plan = planner.verify_plan("repo", str(repo), planner._normalize(
            {"steps": [{"intent": "x", "symbols": ["SOME_EXTERNAL_FLAG"]}]}))
        assert plan["steps"][0]["symbols"] == ["SOME_EXTERNAL_FLAG"]
        assert plan["unresolved_symbols"] == 0

    def test_existing_tests_are_attached_to_the_step(self, repo):
        plan = planner.verify_plan("repo", str(repo), planner._normalize(
            {"steps": [{"intent": "x", "symbols": ["cell_len"]}]}))
        assert plan["steps"][0]["existing_tests"] == ["tests/test_cells.py"]

    def test_the_call_graph_blast_radius_is_attached(self, repo, monkeypatch):
        monkeypatch.setattr(planner.graph, "available", lambda: True)
        monkeypatch.setattr(planner.graph, "lookup", lambda *a, **k: [
            {"file_path": "cells.py", "qualified_name": "cells.cell_len"}])
        monkeypatch.setattr(planner.graph, "callers", lambda *a, **k: [
            {"name": "render", "file_path": "console.py", "start_line": 40}])
        plan = planner.verify_plan("repo", str(repo), planner._normalize(
            {"steps": [{"intent": "x", "symbols": ["cell_len"]}]}))
        assert plan["steps"][0]["blast_radius"] == ["console.py:40 (render)"]

    def test_verification_without_a_working_copy_still_returns_the_plan(self, tmp_path):
        plan = planner.verify_plan("repo", str(tmp_path / "nope"), planner._normalize(
            {"steps": [{"intent": "x", "symbols": ["cell_len"]}]}))
        assert plan["steps"][0]["intent"] == "x"

    def test_an_empty_plan_passes_through(self):
        assert planner.verify_plan("repo", "", {}) == {}


class TestRendering:
    def test_the_dev_prompt_carries_pins_radius_and_tests(self):
        plan = {"summary": "fix width", "steps": [{
            "id": 1, "intent": "strip ansi", "edit_kind": "modify",
            "files": ["cells.py"], "symbols": ["cells.py::cell_len"], "why": "it measures",
            "verify": ["colored == plain"], "blast_radius": ["console.py:40 (render)"],
            "existing_tests": ["tests/test_cells.py"]}],
            "risks": ["double-counting wide glyphs"], "open_questions": [],
            "tests": {"files": [], "new_cases": []}}
        out = planner.as_prompt(plan)
        for expected in ("cells.py::cell_len", "console.py:40 (render)",
                         "tests/test_cells.py", "double-counting wide glyphs",
                         "colored == plain"):
            assert expected in out

    def test_open_questions_are_flagged_as_unconfirmed(self):
        out = planner.as_prompt({"steps": [{"id": 1, "intent": "x", "edit_kind": "modify"}],
                                 "open_questions": ["whether the cache is shared"]})
        assert "could NOT confirm" in out and "whether the cache is shared" in out

    def test_an_empty_plan_renders_to_nothing(self):
        assert planner.as_prompt({}) == "" and planner.as_prompt({"steps": []}) == ""

    def test_targets_dedupe_across_steps(self):
        plan = {"steps": [{"files": ["a.py"], "symbols": ["a.py::x"]},
                          {"files": ["a.py", "b.py"], "symbols": ["a.py::x", "b.py::y"]}]}
        assert planner.targets(plan) == (["a.py", "b.py"], ["a.py::x", "b.py::y"])

    def test_describe_counts_what_matters(self):
        line = planner.describe({"steps": [{"files": ["a.py"]}, {"files": ["b.py"]}],
                                 "risks": ["r"], "unresolved_symbols": 2,
                                 "open_questions": ["q"]})
        assert "2 step(s)" in line and "2 file(s)" in line
        assert "2 symbol(s) not found" in line and "1 open question" in line


class TestPromptIsCacheable:
    """Measured on a live gitea run: the Planner paid ~95% of full retail on
    322,893 input tokens, while Dev — whose backend caches across turns — paid
    17% on 1.88M. Every round is a fresh process re-sending the whole
    accumulated context, and that context is APPEND-ONLY, which is the ideal
    shape for prompt caching. It just has to be handed over as a prefix."""

    def test_the_stable_part_is_the_prefix_and_the_instruction_is_the_tail(
            self, repo, monkeypatch):
        seen: list[tuple[str, str]] = []

        def _ask(system, user, cache_prefix=""):
            seen.append((cache_prefix, user))
            return {"text": json.dumps({"action": "plan",
                                        "plan": {"summary": "s", "steps": [PLAN_STEP]}}),
                    "tokens_in": 1, "tokens_out": 1, "cost": 0.0, "error": None}

        monkeypatch.setattr(planner, "_ask", _ask)
        planner.plan("repo", str(repo), {"summary": "borders"})

        prefix, tail = seen[0]
        assert "borders" in prefix, "the scope belongs in the cacheable prefix"
        assert "Decide your next action" in tail
        # The tail must stay small; a long varying suffix defeats the point.
        assert len(tail) < 400

    def test_the_prefix_only_ever_grows(self, repo, monkeypatch):
        """Caching pays only while each round's prefix CONTAINS the previous
        one verbatim. Any reordering or trimming silently breaks it."""
        prefixes: list[str] = []
        queue = [{"action": "retrieve", "queries": ["a"]},
                 {"action": "retrieve", "queries": ["b"]},
                 {"action": "plan", "plan": {"summary": "s", "steps": [PLAN_STEP]}}]

        monkeypatch.setattr(planner.knowledge_retriever, "retrieve_context",
                            lambda url, q, **k: f"HITS {q}")

        def _ask(system, user, cache_prefix=""):
            prefixes.append(cache_prefix)
            return {"text": json.dumps(queue.pop(0)), "tokens_in": 1, "tokens_out": 1,
                    "cost": 0.0, "error": None}

        monkeypatch.setattr(planner, "_ask", _ask)
        planner.plan("repo", str(repo), {"summary": "x"})

        assert len(prefixes) == 3
        for earlier, later in zip(prefixes, prefixes[1:], strict=False):
            assert later.startswith(earlier), "a round's prefix must extend the last"


class TestRetrievalIsNotRepeated:
    """The rounds ask near-identical questions by design — observed live:
    "dependency picker dropdown", then "dependency dropdown template", then
    "dependency sidebar dropdown list". Their result sets overlap heavily, and
    because the context is re-sent whole every round, each duplicated snippet
    is paid for again on every later round."""

    def test_the_same_set_is_carried_across_rounds(self, repo, monkeypatch):
        excludes: list[set] = []

        def fake_retrieve(url, q, **kw):
            ex = kw.get("exclude")
            excludes.append(ex)
            if ex is not None:
                ex.add(f"hit-for-{q}")
            return f"HITS {q}"

        monkeypatch.setattr(planner.knowledge_retriever, "retrieve_context", fake_retrieve)
        monkeypatch.setattr(planner, "_ask", replies(
            {"action": "retrieve", "queries": ["one"]},
            {"action": "retrieve", "queries": ["two"]},
            {"action": "plan", "plan": {"summary": "s", "steps": [PLAN_STEP]}}))
        planner.plan("repo", str(repo), {"summary": "x"})

        assert len(excludes) == 2
        assert excludes[0] is excludes[1], "one set must span the rounds"
        assert "hit-for-one" in excludes[1], "round 2 must know what round 1 showed"


class TestCriterionCoverage:
    """A plan can be right about everything it covers and still deliver half the
    scope.

    From a live gitea run: a five-criterion scope whose three plan steps
    addressed criteria 1-2 and never mentioned 3-5 (grouping, sort order and
    filter-exclusion of pinned issues). The Planner had noticed them — it listed
    the missing behaviour verbatim under `tests.new_cases` — but a test case is
    not an implementation step, so nothing was built. Dev implemented the plan
    faithfully, and QA plus four jurors approved two fifths of the ask.
    """

    SCOPE = {"acceptance_criteria": [
        "Users with manage-issues permission can pin and unpin issues from both the "
        "issue detail page and inline from the issue list",
        "Each repository can have a maximum of 3 pinned issues; attempting to pin a "
        "4th issue displays an error",
        "When viewing an issue list, pinned issues that match the active filters "
        "appear grouped at the top, above non-pinned matching issues",
        "Within the pinned group and non-pinned group separately, issues are ordered "
        "according to the active sort order",
        "Pinned issues that do not match the active filters are excluded from the "
        "list view",
    ]}

    # The three steps that plan actually produced, trimmed to their intents.
    STEPS = [
        {"intent": "Confirm the permission check on pin/unpin API and web handlers is "
                   "exactly manage-issues (CanWrite(unit.TypeIssues)) and add it if "
                   "PinIssue/UnpinIssue/IssuePinOrUnpin lack an explicit check",
         "why": "the criteria require a non-manage-issues user to get an authorization "
                "error on pin/unpin from the detail page and the list"},
        {"intent": "Confirm MaxPinned is set to 3 and that PinIssue's limit check returns "
                   "a distinguishable ErrIssueMaxPinReached surfaced as a user-readable "
                   "400 message",
         "why": "pinning a 4th issue must error with a readable message"},
        {"intent": "Confirm pin/unpin controls in templates are gated on manage-issues "
                   "permission rather than IsRepoAdmin",
         "why": "manage-issues users who aren't repo admins should see pin controls"},
    ]

    def test_it_finds_the_criteria_that_plan_really_missed(self):
        from app.services import planner

        gaps = planner.coverage_gaps(self.SCOPE, {"steps": self.STEPS})
        assert gaps == self.SCOPE["acceptance_criteria"][2:], (
            "expected exactly the grouping/sort/filter criteria to be flagged")

    def test_it_does_not_flag_a_criterion_a_step_covers(self):
        """The 3-pin criterion is covered by step 2 — in code vocabulary. Nothing
        in it repeats the criterion's words: `maximum` has to reach `MaxPinned`
        and `error` has to reach `ErrIssueMaxPinReached`, or this is a false
        positive on a plan that is doing its job."""
        from app.services import planner

        gaps = planner.coverage_gaps(self.SCOPE, {"steps": self.STEPS})
        assert self.SCOPE["acceptance_criteria"][1] not in gaps

    def test_a_plan_that_covers_everything_is_silent(self):
        from app.services import planner

        steps = [*self.STEPS, {
            "intent": "Filter and sort the pinned group by the active filters and sort "
                      "order, excluding pinned issues that do not match",
            "why": "pinned issues grouped at the top above non-pinned matching issues",
            "files": ["routers/web/repo/issue_list.go"],
            "symbols": ["GetPinnedIssues"],
            "verify": ["a filtered list view excludes non-matching pinned issues"]}]
        assert planner.coverage_gaps(self.SCOPE, {"steps": steps}) == []

    def test_the_tests_block_does_not_count_as_coverage(self):
        """The exact shape of the miss: the behaviour named under `tests.new_cases`
        and nowhere in the steps. Counting it would score that plan as complete."""
        from app.services import planner

        plan = {"steps": self.STEPS, "tests": {"new_cases": [
            "issue list groups pinned issues matching active filters above non-pinned, "
            "each sub-group ordered by the active sort, and excludes a pinned issue "
            "that fails the active filter"]}}
        assert len(planner.coverage_gaps(self.SCOPE, plan)) == 3

    def test_it_stays_quiet_when_it_cannot_know(self):
        from app.services import planner

        assert planner.coverage_gaps({}, {"steps": self.STEPS}) == []
        assert planner.coverage_gaps(self.SCOPE, {"steps": []}) == []
        assert planner.coverage_gaps({}, {}) == []

    def test_dev_is_told_which_criteria_have_no_step(self):
        from app.services import planner

        prompt = planner.as_prompt({
            "steps": [{"id": 1, "edit_kind": "modify", "intent": "do a thing"}],
            "uncovered_criteria": ["pinned issues are excluded when they fail the filter"]})
        assert "NO step above" in prompt
        assert "pinned issues are excluded when they fail the filter" in prompt


class TestTruncationIsLabelled:
    """QA reported a complete test file as "cut off mid-assertion — MakeRequest(t,
    with no closing paren" and raised it as a finding. The paren was in the file,
    9,000 characters in: the prompt was truncated, not the code. Every block a
    judge reads has to say when it was cut."""

    def test_a_cut_names_itself(self):
        from app.services.prompts import clip

        long = "".join(f"line {i} of the diff\n" for i in range(2000))
        out = clip(long, 500)
        assert "DISPLAY CUT" in out
        assert "not the end of the code" in out

    def test_it_cuts_on_a_line_boundary(self):
        """Half a line of Go reads exactly like the truncated assertion QA
        invented a finding about."""
        from app.services.prompts import clip

        long = "".join(f"line {i} of the diff\n" for i in range(2000))
        assert clip(long, 507).split("[...")[0].endswith("\n")

    def test_text_within_budget_is_untouched(self):
        from app.services.prompts import clip

        assert clip("a\nb\n", 4000) == "a\nb\n"
        assert clip("", 10) == ""

    def test_every_judge_facing_prompt_labels_its_cut(self):
        import inspect

        from app.services.jury.prompts import judge_case
        from app.services.prompts import qa_user, review

        diff = "func TestX(t *testing.T) {\n\tMakeRequest(t, req, 400)\n}\n" * 400
        assert "DISPLAY CUT" in qa_user("G-1", "t", ["c"], diff, "out")
        assert "DISPLAY CUT" in review("G-1", ["c"], diff)
        kwargs = dict.fromkeys(inspect.signature(judge_case).parameters, "")
        kwargs["diff"] = diff
        assert "DISPLAY CUT" in judge_case(**kwargs)

    def test_truncated_test_output_is_labelled_as_test_output(self):
        from app.services.prompts import qa_user

        assert "test output omitted" in qa_user("G-1", "t", ["c"], "d", "x" * 5000)
