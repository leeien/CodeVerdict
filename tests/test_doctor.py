"""Preflight checks.

The whole point of `doctor` is to be the one command that never fails on a
broken machine — it reports problems instead of raising them. So these tests
push the failure cases through it: a missing binary, a stage pointed at a
provider with no key, and a coding CLI that isn't installed.
"""

from __future__ import annotations

import pytest
from app.core import doctor


def _flat(report: dict) -> dict[str, dict]:
    return {c["name"]: c for g in report["groups"] for c in g["checks"]}


@pytest.fixture()
def report() -> dict:
    return doctor.check()


class TestShape:
    def test_reports_every_group_and_a_verdict(self, report):
        names = [g["name"] for g in report["groups"]]
        assert names == ["Environment", "Knowledge base", "Pipeline stages",
                         "Coding CLIs detected", "Delivery"]
        assert isinstance(report["ready"], bool)
        assert report["failures"] == 0 or not report["ready"]

    def test_every_check_carries_a_known_status(self, report):
        for check in _flat(report).values():
            assert check["status"] in (doctor.OK, doctor.WARN, doctor.FAIL)
            assert check["name"]

    def test_ready_is_exactly_the_absence_of_failures(self, report):
        failures = [c for c in _flat(report).values() if c["status"] == doctor.FAIL]
        assert report["ready"] is (not failures)
        assert report["failures"] == len(failures)

    def test_json_serialisable(self, report):
        """`codeverdict doctor --json` prints this straight out."""
        import json

        json.loads(json.dumps(report))

    def test_one_row_per_pipeline_stage(self, report):
        from app.services import providers

        stages = {c["name"] for c in report["groups"][2]["checks"]}
        assert stages == {f"stage: {s}" for s in providers.STAGES}


class TestDegradedEnvironment:
    def test_missing_ripgrep_warns_but_does_not_block(self, monkeypatch):
        """ripgrep only makes localization coarser — it must never be fatal."""
        real = doctor.shutil.which
        monkeypatch.setattr(doctor.shutil, "which",
                            lambda exe: None if exe == "rg" else real(exe))
        rg = _flat(doctor.check())["ripgrep"]
        assert rg["status"] == doctor.WARN
        assert rg["hint"]

    def test_missing_git_is_fatal(self, monkeypatch):
        """The agents clone and branch a working copy; without git there is no run."""
        real = doctor.shutil.which
        monkeypatch.setattr(doctor.shutil, "which",
                            lambda exe: None if exe == "git" else real(exe))
        out = doctor.check()
        assert _flat(out)["git"]["status"] == doctor.FAIL
        assert out["ready"] is False

    def test_missing_code_graph_warns_because_the_kb_has_a_fallback(self, monkeypatch):
        from app.services.knowledge import graph

        monkeypatch.setattr(graph, "probe", lambda: {"ok": False, "output": "not found"})
        row = _flat(doctor.check())["code graph (codebase-memory-mcp)"]
        assert row["status"] == doctor.WARN
        assert "codebase-memory-mcp" in row["hint"]


class TestStageReadiness:
    def test_a_stage_with_no_usable_provider_fails_with_both_ways_out(self, monkeypatch):
        from app.services import providers

        monkeypatch.setattr(providers, "can_chat", lambda pid: False)
        out = doctor.check()
        assert out["ready"] is False
        row = _flat(out)["stage: dev"]
        assert row["status"] == doctor.FAIL
        # The hint has to name a repair, not just the problem.
        assert "/model dev" in row["hint"] or "/settings" in row["hint"]

    def test_an_uninstalled_cli_is_only_a_warning_on_its_own(self, monkeypatch):
        """You need one working path to an LLM, not every CLI on the machine."""
        from app.services import agent_backends

        monkeypatch.setattr(agent_backends, "availability", lambda: {
            "claude-code": {"available": False, "version": "", "reason": "not on PATH",
                            "connect_hint": "install it"},
        })
        rows = {c["name"]: c for c in doctor.check()["groups"][3]["checks"]}
        assert rows["claude-code"]["status"] == doctor.WARN


class TestDelivery:
    def test_demo_mode_off_is_flagged_because_it_pushes_for_real(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "demo_mode", False)
        row = _flat(doctor.check())["demo mode"]
        assert row["status"] == doctor.WARN
        assert "real" in row["detail"].lower()

    def test_half_configured_jira_is_flagged(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "jira_base_url", "https://x.atlassian.net")
        monkeypatch.setattr(settings, "jira_email", "a@b.c")
        monkeypatch.setattr(settings, "jira_api_token", "")
        monkeypatch.setattr(settings, "jira_project_key", "")
        assert _flat(doctor.check())["jira"]["status"] == doctor.WARN


class TestRendering:
    def test_renders_without_touching_a_palette_name(self, report):
        """Renderables shared with Textual panels must resolve styles to literals."""
        from app.cli import render, theme
        from rich.console import Console

        console = Console(width=100, record=True, force_terminal=False)
        console.print(render.doctor(report, theme.Glyphs(True)))
        assert "Preflight" in console.export_text()
