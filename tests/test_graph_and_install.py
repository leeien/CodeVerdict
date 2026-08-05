"""Code-graph backend (codebase-memory-mcp wrapper) and one-click CLI install."""

import subprocess
import sys

import pytest
from app.config import settings
from app.services import agent_backends
from app.services.knowledge import graph


@pytest.fixture(autouse=True)
def _restore_graph_settings():
    saved = (settings.graph_enabled, settings.graph_binary, settings.graph_index_mode)
    yield
    (settings.graph_enabled, settings.graph_binary, settings.graph_index_mode) = saved


# --- code graph: availability + fail-open --------------------------------------

def test_disabled_means_unavailable():
    settings.graph_enabled = False
    assert graph.binary() is None
    assert graph.available() is False


def test_missing_binary_is_unavailable():
    settings.graph_enabled = True
    settings.graph_binary = "definitely-not-a-real-binary-xyz"
    assert graph.binary() is None
    assert graph.available() is False


def test_probe_disabled_explains():
    settings.graph_enabled = False
    r = graph.probe()
    assert r["ok"] is False and "disabled" in r["output"].lower()


def test_probe_missing_binary_explains():
    settings.graph_enabled = True
    settings.graph_binary = "definitely-not-a-real-binary-xyz"
    r = graph.probe()
    assert r["ok"] is False and "not found" in r["output"].lower()


def test_queries_fail_open_when_unavailable():
    """Every query returns an empty value (never raises) when the binary is
    gone — callers then degrade to the symbol map + git grep tier."""
    settings.graph_enabled = False
    url = "https://github.com/x/y"
    assert graph.search(url, "anything") == []
    assert graph.semantic(url, "anything") == []
    assert graph.lookup(url, "Foo") == []
    assert graph.callers(url, "Foo") == []
    assert graph.outline(url, "a/b.py") == []
    assert graph.impact(url) == {}
    assert graph.architecture(url) == {}
    assert graph.overview_text(url) == ""
    assert graph.bootstrap_text(url) == ""
    assert graph.indexed(url) is False
    assert graph.ensure_indexed(url) is False


def test_run_returns_empty_on_bad_binary(monkeypatch, tmp_path):
    """A binary that emits non-JSON must not crash the wrapper.

    Uses the running interpreter rather than /bin/echo so the check holds on
    Windows too — what's under test is 'stdout wasn't JSON', not which binary
    produced it.
    """
    settings.graph_enabled = True
    monkeypatch.setattr(graph, "binary", lambda: sys.executable)
    monkeypatch.setattr(graph.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, stdout="not json at all\n", stderr=""))
    assert graph._run("search_graph", {"project": "p", "query": "q"}) == {}


def test_run_parses_last_json_line(monkeypatch):
    settings.graph_enabled = True

    class _Proc:
        returncode = 0
        stdout = 'level=info msg=whatever\n{"results": [{"name": "f"}]}\n'
        stderr = ""

    monkeypatch.setattr(graph, "binary", lambda: "x")
    monkeypatch.setattr(graph.subprocess, "run", lambda *a, **k: _Proc())
    out = graph.search("u", "q")
    assert out == [{"name": "f"}]


def test_label_and_int_coercion():
    # labels(n) comes back as a JSON-encoded string; lines as strings.
    assert graph.node_label('["Function"]') == "Function"
    assert graph.node_label("Method") == "Method"
    assert graph.as_int("75") == 75
    assert graph.as_int(None) is None


def test_lookup_shapes_cypher_rows(monkeypatch):
    settings.graph_enabled = True
    rows = [["record_delivery", '["Function"]',
             "backend/app/x.py", "75", "(a, b)", "app.x.record_delivery"]]
    monkeypatch.setattr(graph, "cypher", lambda *a, **k: rows)
    hits = graph.lookup("u", "record_delivery")
    assert hits == [{"name": "record_delivery", "label": "Function",
                     "file_path": "backend/app/x.py", "start_line": 75,
                     "signature": "(a, b)",
                     "qualified_name": "app.x.record_delivery"}]


def test_lookup_tolerates_a_row_without_a_qualified_name(monkeypatch):
    """An older index (or a node type that carries no qualified name) must still
    resolve to a location — `snippet` loses its source, `lookup` keeps working."""
    monkeypatch.setattr(graph, "cypher", lambda *a, **k: [
        ["f", '["Function"]', "x.py", "1"]])
    assert graph.lookup("u", "f")[0]["qualified_name"] == ""


# --- one-click CLI install -----------------------------------------------------

def test_install_unknown_backend_fails_soft():
    r = agent_backends.install("no-such-tool")
    assert r["ok"] is False and "unknown" in r["output"]


def test_install_not_installable():
    r = agent_backends.install("antigravity")  # no headless mode → no installer
    assert r["ok"] is False and "no auto-install" in r["output"]


def test_install_missing_prerequisite(monkeypatch):
    b = agent_backends.BACKENDS["codex"]
    monkeypatch.setattr(type(b), "install_requires", "definitely-missing-prereq")
    r = agent_backends.install("codex")
    assert r["ok"] is False and "definitely-missing-prereq" in r["output"]


def test_install_runs_cmd_and_redetects(monkeypatch):
    b = agent_backends.BACKENDS["codex"]
    monkeypatch.setattr(type(b), "install_cmd", ("true",))  # trivially succeeds
    monkeypatch.setattr(type(b), "install_requires", "")
    # After "installing", point the path at a real binary so re-detect succeeds.
    monkeypatch.setattr("app.config.settings.codex_cli_path", "ls")
    b.reset_detection()
    r = agent_backends.install("codex")
    assert r["ok"] is True and r["detect"]["available"] is True
    b.reset_detection()


def test_refresh_reprobes_all():
    avail = agent_backends.refresh()
    assert set(avail) == set(agent_backends.BACKENDS)
    for det in avail.values():
        assert "installable" in det
