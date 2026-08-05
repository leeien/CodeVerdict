"""The Dev agent's callable retrieval tools (services/knowledge/tools.py).

Covers the three surfaces that must behave identically regardless of which
model/backend the Dev stage runs on: the dispatcher, the `.codeverdict/kb` shim
installed for headless CLI backends, and the request-block protocol the HTTP
SEARCH/REPLACE loop parses. No graph binary is needed — the tools degrade to the
symbol map / git grep, which is exactly the path asserted here.
"""

from __future__ import annotations

import os
import subprocess

import pytest
from app.services import openai_agent, prompts
from app.services.knowledge import tools


@pytest.fixture()
def repo(tmp_path):
    """A tiny real git repo — git grep is the no-graph fallback path."""
    (tmp_path / "ansi.py").write_text(
        "STRIP_RE = None\n\n\ndef cell_width(text):\n    return len(text)\n")
    (tmp_path / "console.py").write_text(
        "from ansi import cell_width\n\n\ndef render(text):\n    return cell_width(text)\n")
    for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.email=t@t", "-c", "user.name=t",
                                                "commit", "-qm", "init"]):
        subprocess.run(["git", *cmd], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


class TestDispatcher:
    def test_lookup_finds_a_real_definition_without_the_graph(self, repo):
        out = tools.call("example__repo", str(repo), "lookup", "cell_width")
        assert "ansi.py" in out

    def test_lookup_reports_a_missing_symbol_as_new(self, repo):
        out = tools.call("example__repo", str(repo), "lookup", "does_not_exist")
        assert "NOT FOUND" in out

    def test_grep_returns_file_line_hits(self, repo):
        out = tools.call("example__repo", str(repo), "grep", "cell_width")
        assert "ansi.py:" in out

    def test_unknown_tool_names_the_available_ones(self, repo):
        out = tools.call("example__repo", str(repo), "frobnicate", "x")
        assert "unknown tool" in out and "lookup" in out

    def test_missing_argument_is_reported_not_raised(self, repo):
        assert "missing argument" in tools.call("example__repo", str(repo), "lookup", "  ")

    def test_a_failing_tool_degrades_to_a_message(self, repo, monkeypatch):
        monkeypatch.setattr(tools.graph, "available", lambda: True)
        monkeypatch.setattr(tools.graph, "lookup",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("index gone")))
        out = tools.call("example__repo", str(repo), "lookup", "cell_width")
        assert "failed" in out and "index gone" in out


class TestShim:
    def test_installed_shim_answers_a_real_query(self, repo):
        cmd = tools.install(str(repo), "example__repo")
        assert cmd == (".codeverdict/kb.cmd" if os.name == "nt" else ".codeverdict/kb")
        args = [str(repo / cmd), "lookup", "cell_width"]
        p = subprocess.run(args, cwd=repo, capture_output=True, text=True, timeout=120,
                           shell=os.name == "nt")
        assert p.returncode == 0, p.stderr
        assert "ansi.py" in p.stdout

    def test_shim_is_hidden_from_git_so_it_never_lands_in_the_dev_diff(self, repo):
        tools.install(str(repo), "example__repo")
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                                capture_output=True, text=True).stdout
        assert ".codeverdict" not in status

    def test_install_fails_open_on_an_unwritable_path(self):
        if os.name == "nt":
            pytest.skip("/proc is not an unwritable path on Windows")
        assert tools.install("/proc/nonexistent-codeverdict", "example__repo") == ""


class TestPromptSurfaces:
    def test_cli_prompt_advertises_every_tool(self, repo):
        block = prompts.dev("T-1", "t", "d", ["c"], tools_cmd=".codeverdict/kb")
        for name in tools.TOOL_NAMES:
            assert f".codeverdict/kb {name}" in block

    def test_prompt_without_tools_installed_keeps_the_grep_fallback(self):
        block = prompts.dev("T-1", "t", "d", ["c"])
        assert ".codeverdict/kb" not in block
        assert "ONE extra grep" in block

    def test_dev_is_told_it_may_overrule_the_pm_localization(self):
        block = prompts.dev("T-1", "t", "d", ["c"], affected_files=["console.py"],
                            target_symbols=["render"])
        assert "HYPOTHESIS" in block and "EXPECTED to disagree" in block

    def test_revise_gets_the_same_tools(self):
        block = prompts.revise("T-1", "t", ["c"], "review", "qa", tools_cmd=".codeverdict/kb")
        assert ".codeverdict/kb search" in block


class TestHttpLoopProtocol:
    def test_every_tool_has_a_request_block(self):
        text = "SUMMARY: looking\n" + "\n".join(
            f"<<<{name.upper()} arg{i}>>>" for i, name in enumerate(tools.TOOL_NAMES))
        _s, edits, files, _opens, queries, _done = openai_agent._parse_edits(text)
        assert not edits and not files
        assert [t for t, _a in queries] == list(tools.TOOL_NAMES)

    def test_request_blocks_inside_file_content_are_not_executed(self):
        text = ("SUMMARY: writing\n<<<FILE docs/protocol.md>>>\n"
                "<<<LOOKUP NotARealRequest>>>\n<<<END>>>\n<<<SEARCH real query>>>\n")
        _s, _e, files, _o, queries, _d = openai_agent._parse_edits(text)
        assert files and files[0][0] == "docs/protocol.md"
        assert queries == [("search", "real query")]

    def test_system_prompt_offers_the_tools_and_licenses_disagreement(self):
        assert "<<<CALLERS" in openai_agent._CODE_SYSTEM
        assert "HYPOTHESIS" in openai_agent._CODE_SYSTEM


class TestHttpLoopEndToEnd:
    """The loop must actually ANSWER a tool request — parsing it isn't enough."""

    def test_a_query_round_feeds_real_index_results_back_to_the_model(self, repo, monkeypatch):
        prompts_seen: list[str] = []
        replies = iter([
            "SUMMARY: locating\n<<<LOOKUP cell_width>>>\nSTATUS: CONTINUE",
            ("SUMMARY: fixing the real site\n<<<EDIT ansi.py>>>\n<<<SEARCH>>>\n"
             "    return len(text)\n<<<REPLACE>>>\n    return len(text) - 1\n<<<END>>>\n"
             "STATUS: CONTINUE"),
            "SUMMARY: done\nSTATUS: DONE",
        ])

        def fake_chat(system, user, **kw):
            prompts_seen.append(user)
            return {"text": next(replies), "tokens_in": 1, "tokens_out": 1,
                    "cost": 0.0, "error": None}

        monkeypatch.setattr(openai_agent, "chat", fake_chat)
        res = openai_agent.code(str(repo), "T-1", "fix width", "d", ["c"],
                                affected_files=["console.py"], target_symbols=["render"])
        assert res["error"] is None and res["files"] == ["ansi.py"]
        # Round 2's prompt carries the LOOKUP answer — the real definition site,
        # which is NOT the file the PM pinned.
        assert "LOOKUP cell_width" in prompts_seen[1] and "ansi.py" in prompts_seen[1]


class TestServerBackedShim:
    """The shim must ask the running server: the embedded vector store is
    single-writer, so a second process silently loses the semantic channel."""

    def test_install_mints_a_token_bound_to_one_repo_and_working_copy(self, repo):
        tools.install(str(repo), "example__repo")
        script = (repo / ".codeverdict" / "kb.py").read_text(encoding="utf-8")
        token = script.split('TOKEN = ')[1].splitlines()[0].strip().strip("'\"")
        assert tools.resolve_token(token) == ("example__repo", str(repo))

    def test_an_unknown_token_resolves_to_nothing(self):
        assert tools.resolve_token("not-a-real-token") is None

    def test_the_endpoint_runs_the_tool_for_a_valid_token(self, repo):
        from app.routers import kb_tools

        tools.install(str(repo), "example__repo")
        token = (repo / ".codeverdict" / "kb.py").read_text(encoding="utf-8").split(
            'TOKEN = ')[1].splitlines()[0].strip().strip("'\"")
        out = kb_tools.run_tool(kb_tools.ToolCall(token=token, tool="lookup", arg="cell_width"))
        assert "ansi.py" in out["result"]

    def test_the_endpoint_refuses_an_unknown_token(self):
        import fastapi
        from app.routers import kb_tools

        with pytest.raises(fastapi.HTTPException) as exc:
            kb_tools.run_tool(kb_tools.ToolCall(token="forged", tool="lookup", arg="x"))
        assert exc.value.status_code == 403

    def test_the_shim_falls_back_in_process_when_no_server_answers(self, repo, monkeypatch):
        monkeypatch.setenv("PORT", "1")  # nothing listens there
        cmd = tools.install(str(repo), "example__repo")
        p = subprocess.run([str(repo / cmd), "lookup", "cell_width"], cwd=repo,
                           capture_output=True, text=True, timeout=120,
                           shell=os.name == "nt")
        assert p.returncode == 0 and "ansi.py" in p.stdout


class TestToolBlockPlacement:
    """Live-run finding: buried behind the knowledge + verified-locations dumps
    (tens of thousands of characters), the tools were never invoked."""

    def test_tools_come_before_the_knowledge_dump(self):
        block = prompts.dev("T-1", "t", "d", ["c"], context="KNOWLEDGE-SLICE",
                            verified="VERIFIED-PINS", tools_cmd=".codeverdict/kb")
        assert block.index(".codeverdict/kb") < block.index("KNOWLEDGE-SLICE") < block.index("VERIFIED-PINS")

    def test_revise_puts_them_first_too(self):
        block = prompts.revise("T-1", "t", ["c"], "rev", "qa", context="KNOWLEDGE-SLICE",
                               verified="VERIFIED-PINS", tools_cmd=".codeverdict/kb")
        assert block.index(".codeverdict/kb") < block.index("KNOWLEDGE-SLICE")
