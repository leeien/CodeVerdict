"""The verdict-parsing helpers that drive the revise loop.

These decide whether the pipeline loops back to Dev, so their exact semantics
matter: CONCERNS must not trigger a revision, an errored-with-no-text agent call
is INCONCLUSIVE (not a pass), and a lost Claude CLI must be detected so the
pipeline can fall back.
"""

from app.services import orchestrator as orch


class TestQaFailed:
    def test_explicit_fail_triggers_revision(self):
        assert orch._qa_failed("VERDICT: FAIL — tests do not pass")

    def test_pass_does_not(self):
        assert not orch._qa_failed("VERDICT: PASS")

    def test_concerns_does_not_trigger(self):
        # CONCERNS is deliberately weaker than FAIL — it must not loop.
        assert not orch._qa_failed("VERDICT: CONCERNS — minor style nits")

    def test_fail_must_be_near_the_verdict(self):
        # The word "fail" elsewhere in prose must not be read as a FAIL verdict.
        assert not orch._qa_failed(
            "The feature prevents the login from failing. VERDICT: PASS"
        )

    def test_empty(self):
        assert not orch._qa_failed("")
        assert not orch._qa_failed(None)


class TestReviewChangesRequested:
    def test_detected_case_insensitive(self):
        assert orch._review_changes_requested("changes requested: fix the null check")
        assert orch._review_changes_requested("CHANGES REQUESTED")

    def test_approved_is_not_a_change_request(self):
        assert not orch._review_changes_requested("APPROVED — ships as-is")

    def test_empty(self):
        assert not orch._review_changes_requested("")

    def test_inconclusive_is_not_approval(self):
        assert orch._inconclusive_verdict("VERDICT: INCONCLUSIVE — provider unavailable")
        assert not orch._inconclusive_verdict("VERDICT: APPROVED")


class TestInconclusive:
    def test_error_with_no_text_is_inconclusive(self):
        assert orch._inconclusive({"error": "timeout", "text": ""})

    def test_error_but_has_verdict_is_not_inconclusive(self):
        assert not orch._inconclusive({"error": "timeout", "text": "VERDICT: PASS"})

    def test_clean_result_is_not_inconclusive(self):
        assert not orch._inconclusive({"text": "VERDICT: APPROVED"})


class TestBackendUnavailable:
    def test_missing_backend_is_unavailable(self, monkeypatch):
        # Backend can't run on this machine → unavailable regardless of error text.
        monkeypatch.setattr(orch.agent_backends, "is_available", lambda _b: False)
        assert orch._backend_unavailable("claude-code", "anything")

    def test_auth_error_with_backend_present_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(orch.agent_backends, "is_available", lambda _b: True)
        assert orch._backend_unavailable("claude-code", "Invalid API key — please run login")

    def test_ordinary_error_with_backend_present_is_not_unavailable(self, monkeypatch):
        # Backend runs and the error is a real code problem → keep the strong model.
        monkeypatch.setattr(orch.agent_backends, "is_available", lambda _b: True)
        assert not orch._backend_unavailable("claude-code", "syntax error in generated patch")


class TestIsTransient:
    def test_rate_limit_and_overload_are_transient(self):
        assert orch._is_transient("HTTP 429 rate limit exceeded")
        assert orch._is_transient("Error 529: overloaded")

    def test_code_error_is_not_transient(self):
        assert not orch._is_transient("patch does not apply")


def _sub(**over):
    base = {"affected_files": ["rich/ansi.py"], "criteria": ["given X, Y happens"],
            "target_symbols": ["rich/ansi.py::decode_line"]}
    base.update(over)
    return base


class TestFastPathEligible:
    """Pre-Dev triage is deterministic: it must only fire on one fully
    grep-pinned ticket, and any unverified localization must fall through."""

    def test_pinned_single_ticket_is_eligible(self, monkeypatch):
        monkeypatch.setattr(orch.settings, "fast_path_enabled", True)
        assert orch._fast_path_eligible([_sub()])

    def test_disabled_setting_wins(self, monkeypatch):
        monkeypatch.setattr(orch.settings, "fast_path_enabled", False)
        assert not orch._fast_path_eligible([_sub()])

    def test_multiple_tickets_are_not_trivial(self, monkeypatch):
        monkeypatch.setattr(orch.settings, "fast_path_enabled", True)
        assert not orch._fast_path_eligible([_sub(), _sub()])

    def test_unpinned_symbol_falls_through(self, monkeypatch):
        # A bare name (mention-only) or a "(new — not in repo yet)" tag means
        # ground_tickets could NOT verify the definition site.
        monkeypatch.setattr(orch.settings, "fast_path_enabled", True)
        assert not orch._fast_path_eligible([_sub(target_symbols=["decode_line"])])
        assert not orch._fast_path_eligible(
            [_sub(target_symbols=["decode_line (new — not in repo yet)"])])
        assert not orch._fast_path_eligible([_sub(target_symbols=[])])

    def test_wide_localization_falls_through(self, monkeypatch):
        monkeypatch.setattr(orch.settings, "fast_path_enabled", True)
        assert not orch._fast_path_eligible([_sub(affected_files=["a.py", "b.py", "c.py"])])
        assert not orch._fast_path_eligible([_sub(affected_files=[])])
        assert not orch._fast_path_eligible([_sub(criteria=["a", "b", "c", "d", "e"])])


class TestDiffStats:
    DIFF = (
        "diff --git a/rich/ansi.py b/rich/ansi.py\n"
        "--- a/rich/ansi.py\n"
        "+++ b/rich/ansi.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-old = 1\n"
        "+new = 1\n"
        " context\n"
        "diff --git a/tests/test_ansi.py b/tests/test_ansi.py\n"
        "--- a/tests/test_ansi.py\n"
        "+++ b/tests/test_ansi.py\n"
        "@@ -0,0 +1,3 @@\n"
        "+def test_new():\n"
        "+    assert new == 1\n"
        "+    assert True\n"
    )

    def test_counts_source_only(self):
        # test files don't count against the fast-path budget — a one-line fix
        # plus real regression tests is exactly the shape the fast path is for
        files, lines = orch._diff_stats(self.DIFF)
        assert files == 1
        assert lines == 2

    def test_empty_diff(self):
        assert orch._diff_stats("") == (0, 0)
