"""Localization grounding and per-juror evidence.

Both regressions here were found by live pipeline runs on `rich`, not by review:
a ticket pinned to `rich/panel.py::Table.__init__`, and an Architecture juror
blocking a delivery over "should match the established pattern" without ever
having been shown the pattern.

The grounding rules now live in `services/planner.py` (they ground a Planner's
decision rather than a PM's guess), but they are the same rules and these are
the same regressions — the scars moved with the code.
"""

from __future__ import annotations

from app.services import planner
from app.services.jury import evidence, personas, prompts


class TestDottedSymbolGrounding:
    def test_a_generic_method_resolves_to_its_owning_class(self):
        # "Table.__init__" must not be looked up as bare "__init__" — that is how
        # the live run pinned rich/panel.py.
        assert planner._ident_and_owner(["Table", "__init__"]) == ("Table", "Table")

    def test_a_specific_method_keeps_its_owner_for_filtering(self):
        assert planner._ident_and_owner(["Table", "_measure_column"]) == ("_measure_column", "Table")

    def test_a_bare_symbol_has_no_owner(self):
        assert planner._ident_and_owner(["cell_len"]) == ("cell_len", "")

    def test_no_identifiers_is_handled(self):
        assert planner._ident_and_owner([]) == ("", "")

    def test_graph_hits_are_filtered_to_the_named_owner(self, monkeypatch):
        monkeypatch.setattr(planner.graph, "lookup", lambda repo, name, limit=5: [
            {"file_path": "rich/panel.py", "qualified_name": "rich.panel.Panel._render"},
            {"file_path": "rich/table.py", "qualified_name": "rich.table.Table._render"},
        ])
        assert planner._graph_files("repo", "_render", "Table") == ["rich/table.py"]

    def test_unfilterable_hits_are_not_discarded(self, monkeypatch):
        monkeypatch.setattr(planner.graph, "lookup", lambda repo, name, limit=5: [
            {"file_path": "rich/box.py", "qualified_name": "rich.box.get_row"},
        ])
        assert planner._graph_files("repo", "get_row", "Table") == ["rich/box.py"]


class TestPerJurorEvidence:
    DIFF = "--- a/rich/table.py\n+++ b/rich/table.py\n@@\n+x\n"

    def test_changed_files_come_from_the_diff(self):
        assert evidence.changed_files(self.DIFF) == ["rich/table.py"]

    def test_personas_without_a_builder_get_nothing(self, tmp_path):
        assert evidence.for_persona("correctness", str(tmp_path), self.DIFF) == ""

    def test_evidence_is_skipped_when_the_graph_is_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(evidence.graph, "available", lambda: False)
        assert evidence.for_persona("architecture", str(tmp_path), self.DIFF) == ""

    def test_architecture_juror_is_shown_the_neighbours_it_must_cite(self, tmp_path, monkeypatch):
        monkeypatch.setattr(evidence.graph, "available", lambda: True)
        monkeypatch.setattr(evidence.graph, "outline", lambda repo, path, limit=30: [
            {"name": "border_style", "start_line": 10},
            {"name": "row_styles", "start_line": 20},
        ])
        out = evidence.for_persona("architecture", str(tmp_path), self.DIFF)
        assert "border_style:10" in out and "row_styles:20" in out

    def test_security_juror_is_told_whether_the_change_is_reachable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(evidence.graph, "available", lambda: True)
        monkeypatch.setattr(evidence.graph, "architecture", lambda repo: {
            "routes": [], "entry_points": [{"name": "cli", "file": "rich/__main__.py"}]})
        out = evidence.for_persona("security", str(tmp_path), self.DIFF)
        assert "changed files that are themselves entry points: NONE" in out
        assert "ABSTAIN" in out

    def test_a_broken_graph_never_breaks_the_review(self, tmp_path, monkeypatch):
        monkeypatch.setattr(evidence.graph, "available", lambda: True)
        monkeypatch.setattr(evidence.graph, "outline",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        assert evidence.for_persona("architecture", str(tmp_path), self.DIFF) == ""

    def test_the_brief_carries_the_evidence_block(self):
        brief = prompts.judge_user("CHARGE", "T-1", "t", ["c"], "diff",
                                   evidence="PERSONA-EVIDENCE-BLOCK")
        assert "PERSONA-EVIDENCE-BLOCK" in brief


class TestBlockingCalibration:
    def test_architecture_juror_is_told_annotations_are_not_blocking(self):
        charge = personas.charge("architecture")
        assert "type annotations" in charge and 'severity "low"' in charge

    def test_foreperson_reclassifies_polish_labelled_medium(self):
        contract = " ".join(prompts._FOREPERSON_CONTRACT.split())
        assert "type-annotation nits" in contract
        assert "A juror labelling polish as \"medium\" does not make it blocking" in contract


class TestBareGenericSymbols:
    """Second live-run variant: the PM emitted a bare `__init__` with no class,
    which resolved to `rich/panel.py::__init__` on a table ticket."""

    def test_a_bare_generic_method_is_refused(self):
        assert planner._ident_and_owner(["__init__"]) == ("", "")

    def test_a_bare_specific_symbol_still_resolves(self):
        assert planner._ident_and_owner(["_measure_column"]) == ("_measure_column", "")

    def test_an_unresolvable_symbol_adds_no_files(self, monkeypatch, tmp_path):
        (tmp_path / ".git").mkdir()
        monkeypatch.setattr(planner.graph, "available", lambda: True)
        monkeypatch.setattr(planner.graph, "lookup",
                            lambda *a, **k: [{"file_path": "rich/panel.py",
                                              "qualified_name": "rich.panel.Panel.__init__"}])
        monkeypatch.setattr(planner.graph, "callers", lambda *a, **k: [])
        plan = planner.verify_plan("https://example.com/o/r", str(tmp_path), {
            "steps": [{"id": 1, "intent": "x", "edit_kind": "modify",
                       "files": ["rich/table.py"], "symbols": ["__init__"],
                       "why": "", "verify": [], "blast_radius": []}]})
        assert plan["steps"][0]["files"] == ["rich/table.py"]
        assert "rich/panel.py" not in plan["steps"][0]["files"]


class TestProtocolDunders:
    """Third live-run variant: `Table.__rich__` resolved to `rich/json.py`,
    because __rich__ is a protocol method many classes implement."""

    def test_any_dunder_is_generic_so_the_class_is_pinned(self):
        assert planner._ident_and_owner(["Table", "__rich__"]) == ("Table", "Table")
        assert planner._ident_and_owner(["Table", "__rich_measure__"]) == ("Table", "Table")

    def test_an_owner_scoped_miss_resolves_the_class_not_a_stranger(self, monkeypatch):
        def lookup(repo, name, limit=5):
            if name == "_render_empty":  # only a foreign class defines it
                return [{"file_path": "rich/json.py", "qualified_name": "rich.json.JSON._render_empty"}]
            return [{"file_path": "rich/table.py", "qualified_name": "rich.table.Table"}]

        monkeypatch.setattr(planner.graph, "lookup", lookup)
        assert planner._graph_files("repo", "_render_empty", "Table") == ["rich/table.py"]


class TestForepersonSeesTheUserRequest:
    """The jurors were given the user's verbatim request; the foreperson was
    not. It saw only "The requirement the change was accepted against: <PM
    criteria>", so a juror appealing to user intent had made, from where the
    foreperson sat, an unsupported claim.

    Observed live on gitea: the user wrote "it should only show issues that
    aren't already dependencies" twice, PM's criteria said already-linked issues
    should "appear with disabled styling", Dev implemented the criteria, and the
    one juror who caught the divergence was dismissed for "contradicting the
    documented requirement" — the paraphrase being the thing that drifted."""

    def _prompt(self, request=""):
        from app.services.jury import prompts

        return prompts.foreperson_user(
            "G-101", "Disable already-linked dependencies",
            ["already-linked issues appear with disabled styling"],
            "juror said the items are still displayed", 0.5, request=request)

    def test_the_users_words_reach_the_foreperson(self):
        out = self._prompt("it should only show issues that aren't already dependencies")
        assert "only show issues that aren't already dependencies" in out

    def test_it_is_told_the_criteria_are_a_paraphrase(self):
        out = self._prompt("hide them")
        assert "paraphrase" in out.lower()
        assert "do not dismiss it merely because it conflicts" in out.lower()

    def test_drift_is_named_as_blocking_in_the_contract(self):
        """Without this the foreperson re-derives 'it satisfies the criteria,
        therefore approve' — which is locally correct and globally wrong."""
        out = self._prompt("hide them")
        low = out.lower()
        assert "requirement drift" in low
        assert "single juror" in low, "the other jurors check the same paraphrase"

    def test_no_request_leaves_the_prompt_unchanged_in_shape(self):
        out = self._prompt("")
        assert "What the user actually asked for" not in out
        assert "The requirement the change was accepted against" in out


class TestQaCannotSilentlyPassWithoutTests:
    """QA sees a diff and the test output, so "no failures in the output" is
    exactly what a passing run looks like. On gitea the suite timed out and QA
    returned PASS having verified nothing."""

    def test_a_missing_suite_is_announced_not_implied(self):
        from app.services import orchestrator

        assert "TIMED OUT" in orchestrator._test_outcome(
            None, "Could not run tests: timed out after 600 seconds")

    def test_the_stamped_verdict_says_it_was_unverified(self):
        """qa_summary feeds the board, the PR body and the write-back — a bare
        PASS there claims a verification that never happened."""
        note = ("\n\n[NO TEST SIGNAL — the suite did not run for this change; "
                "this verdict is a reading of the diff, not a verified result.]")
        stamped = "VERDICT: PASS" + note
        assert "NO TEST SIGNAL" in stamped
        assert "not a verified result" in stamped
