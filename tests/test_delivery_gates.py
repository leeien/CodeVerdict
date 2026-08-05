"""Delivery-state safety gates.

These tests cover the boundaries that must remain true even when the model
responses and pipeline are otherwise mocked: an unverified or incomplete
delivery cannot become merged, and an interrupted run remains visible as blocked.
"""

import pytest
from app.core import tasks
from app.core.errors import CoreError
from app.models import Repo, ScopeSession, Task, TaskStatus
from app.services import orchestrator
from sqlmodel import Session


def _task(db: Session, *, status: str = TaskStatus.pr.value,
          qa: str = "VERDICT: PASS", review: str = "VERDICT: APPROVED",
          findings: dict | None = None) -> Task:
    repo = Repo(name="repo", org="org", git_url="https://example.com/org/repo.git")
    db.add(repo)
    db.commit()
    task = Task(key="T-1", repo_id=repo.id, title="change", status=status,
                branch="agent/scope-1", qa_summary=qa, review_summary=review,
                review_findings=findings or {})
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def test_merge_requires_pr_lane_and_explicit_approval(db, monkeypatch):
    monkeypatch.setattr(tasks.settings, "demo_mode", True)
    task = _task(db, status=TaskStatus.scoped.value)
    with pytest.raises(CoreError):
        tasks.merge(db, task.id)

    task.status = TaskStatus.pr.value
    task.review_summary = "VERDICT: INCONCLUSIVE — reviewer unavailable"
    db.add(task)
    db.commit()
    with pytest.raises(CoreError):
        tasks.merge(db, task.id)


def test_merge_allows_an_explicitly_approved_demo_delivery(db, monkeypatch):
    monkeypatch.setattr(tasks.settings, "demo_mode", True)
    task = _task(db)
    out = tasks.merge(db, task.id)
    assert out["ok"] is True
    db.refresh(task)
    assert task.status == TaskStatus.done.value


def test_merge_checks_every_sibling_gate(db, monkeypatch):
    monkeypatch.setattr(tasks.settings, "demo_mode", True)
    task = _task(db)
    sibling = Task(key="T-2", repo_id=task.repo_id, session_id=None, title="other",
                   status=TaskStatus.pr.value, branch=task.branch,
                   qa_summary="VERDICT: INCONCLUSIVE", review_summary="VERDICT: APPROVED")
    # A shared branch makes this a delivery sibling even without a session only
    # when the session relationship exists; create it explicitly for the test.
    session = ScopeSession(repo_id=task.repo_id, title="scope", status="scoped")
    db.add(session)
    db.commit()
    task.session_id = session.id
    sibling.session_id = session.id
    db.add(task)
    db.add(sibling)
    db.commit()

    with pytest.raises(CoreError):
        tasks.merge(db, task.id)


def test_generic_task_update_cannot_skip_delivery_gates(db):
    task = _task(db, status=TaskStatus.scoped.value)

    with pytest.raises(CoreError):
        tasks.update(db, task.id, {"status": TaskStatus.done.value})

    db.refresh(task)
    assert task.status == TaskStatus.scoped.value


def test_interrupted_scope_is_persisted_as_blocked(db):
    repo = Repo(name="repo", org="org", git_url="https://example.com/org/repo.git")
    db.add(repo)
    db.commit()
    session = ScopeSession(repo_id=repo.id, title="scope", status="scoped")
    db.add(session)
    db.commit()
    task = Task(key="T-1", repo_id=repo.id, session_id=session.id, title="change",
                status=TaskStatus.in_dev.value, approved=True, current_agent="dev")
    db.add(task)
    db.commit()

    orchestrator._mark_scope_blocked(session.id, "test interruption")
    db.refresh(task)
    db.refresh(session)
    assert task.status == TaskStatus.blocked.value
    assert task.current_agent is None
    assert "test interruption" in task.review_summary
    assert session.status == "failed"


class TestDevFailureStopsTheScope:
    """A Dev failure stopped the run only when it had ALSO written nothing:

        if res["error"] and not committed:   # stop

    An agent killed part-way through has usually written something — the live
    quota cutoff that prompted this ran 865s and 1.2M input tokens before it
    died, so `committed` was True and the `and` let the pipeline walk into QA
    to judge a half-finished change. The jury cannot tell a truncated edit set
    from a complete one by reading the diff."""

    def test_a_partial_commit_does_not_earn_a_qa_pass(self, monkeypatch):
        """The regression itself: errored Dev + files committed must still stop
        before QA."""
        from app.services import orchestrator

        assert orchestrator._should_stop_after_dev({"error": "session limit"}, committed=True)

    def test_an_errored_dev_that_wrote_nothing_still_stops(self):
        from app.services import orchestrator

        assert orchestrator._should_stop_after_dev({"error": "boom"}, committed=False)

    def test_a_clean_dev_run_proceeds(self):
        from app.services import orchestrator

        assert not orchestrator._should_stop_after_dev({"error": None}, committed=True)

    def test_a_successful_dev_that_changed_nothing_still_proceeds(self):
        """No error and no diff is a legitimate outcome (already-satisfied
        criteria); QA is what decides whether that is acceptable."""
        from app.services import orchestrator

        assert not orchestrator._should_stop_after_dev({"error": ""}, committed=False)


class TestTestOutcomeIsHonest:
    """`passed is None` covered three different situations that all printed
    "no suite/deps": no tests exist, the runner could not start, and the suite
    ran but timed out. On gitea the third is the common one — `go test ./...`
    compiles 3,024 files against a 600s limit — and calling it "no suite"
    invites the wrong conclusion, that there was nothing for QA to check."""

    def test_a_timeout_is_not_reported_as_a_missing_suite(self):
        from app.services import orchestrator

        out = orchestrator._test_outcome(
            None, "Could not run tests: Command '['go', 'test', './...']' "
                  "timed out after 600 seconds")
        assert "TIMED OUT" in out
        assert "no suite" not in out

    def test_a_genuinely_absent_suite_still_says_so(self):
        from app.services import orchestrator

        assert orchestrator._test_outcome(None, "No recognized test suite found.") \
            == "no suite/deps"

    def test_pass_and_fail_are_unchanged(self):
        from app.services import orchestrator

        assert orchestrator._test_outcome(True, "ok") == "passed"
        assert orchestrator._test_outcome(False, "--- FAIL: TestX") == "failures"

    def test_a_runner_that_could_not_start_is_distinguished(self):
        from app.services import orchestrator

        out = orchestrator._test_outcome(None, "Could not run tests: [Errno 2] no go")
        assert "could not run" in out and "no suite" not in out
