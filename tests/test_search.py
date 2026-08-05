"""The lexical search engine (services/search.py).

Two engines answer the same contract — ripgrep when the binary is present, and
`git grep` when it is not. Every behavioural test here runs against BOTH, because
the fallback is not decoration: it is what a box without ripgrep actually uses,
and a silent divergence between the two is a localization bug that only appears
on someone else's machine. (It already happened once: `git grep -E` matches the
literal text `?:` rather than erroring on a non-capturing group, so every
language-aware definition pattern returned zero hits with no sign of trouble.)

The ref-pinned worktree — how a search reaches a COMMITTED tree now that the
engine searches directories rather than git refs — is covered at the end.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from app.config import settings
from app.services import git_ops, search


def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
                   cwd=cwd, check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path):
    """A small polyglot repo: one definition per language, plus a test file and
    a doc file, so type filtering and pathspec translation have something to
    actually discriminate between."""
    (tmp_path / "ansi.py").write_text(
        "STRIP_RE = None\n\n\ndef cell_width(text):\n    return len(text)\n\n\n"
        "class Table:\n    pass\n")
    (tmp_path / "widget.js").write_text(
        "export function cellWidth(t) { return t.length; }\n"
        "const Table = 1;\n")
    (tmp_path / "server.go").write_text(
        "package main\n\nfunc CellWidth(s string) int {\n\treturn len(s)\n}\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ansi.py").write_text(
        "from ansi import cell_width\n\n\ndef test_cell_width():\n    assert cell_width('a') == 1\n")
    (tmp_path / "README.md").write_text("cell_width is documented here.\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


@pytest.fixture(params=["ripgrep", "git-grep"])
def engine(request, monkeypatch):
    """Run the test body once per engine. ripgrep is skipped, not failed, when
    the binary isn't installed — but the git-grep tier always runs."""
    if request.param == "ripgrep":
        if shutil.which(settings.ripgrep_path or "rg") is None:
            pytest.skip("ripgrep not installed")
        monkeypatch.setattr(settings, "ripgrep_enabled", True)
    else:
        monkeypatch.setattr(settings, "ripgrep_enabled", False)
    return request.param


class TestBothEngines:
    def test_lines_returns_file_line_text(self, repo, engine):
        out = search.lines(str(repo), r"def cell_width", max_lines=5)
        assert "ansi.py:4:" in out

    def test_no_match_is_a_message_not_an_error(self, repo, engine):
        assert search.lines(str(repo), "ZzzNoSuchThing") == "(no matches)"

    def test_an_invalid_regex_is_a_no_match_not_a_crash(self, repo, engine):
        # The caller is often a model that guessed at a pattern; a raised
        # exception here would take down its whole round.
        assert search.lines(str(repo), "[unclosed") == "(no matches)"

    def test_files_lists_paths_not_lines(self, repo, engine):
        assert search.files(str(repo), r"cell_width", max_files=10) == sorted(
            search.files(str(repo), r"cell_width", max_files=10))
        assert "ansi.py" in search.files(str(repo), r"cell_width", max_files=10)

    def test_pathspec_crosses_directories_like_a_git_pathspec(self, repo, engine):
        # '*test*' must mean 'any path containing test', as it did under git
        # grep — a ripgrep glob is segment-scoped by default and would miss
        # tests/test_ansi.py entirely.
        hits = search.files(str(repo), r"cell_width", pathspec="*test*", max_files=10)
        assert hits == ["tests/test_ansi.py"]

    def test_definitions_finds_the_definition_site(self, repo, engine):
        assert search.definitions(str(repo), "cell_width") == ["ansi.py"]

    def test_definitions_is_empty_for_an_invented_symbol(self, repo, engine):
        assert search.definitions(str(repo), "ZzzInventedSymbol") == []

    def test_definitions_handles_non_python_declaration_syntax(self, repo, engine):
        # The pattern this replaced knew `def|class|function|const|var|let`, so a
        # Go `func` receiver-less declaration was invisible to it.
        assert search.definitions(str(repo), "CellWidth") == ["server.go"]

    def test_definitions_are_deterministic(self, repo, engine):
        # Callers pin work to definitions()[0]; a set that reorders between two
        # identical runs is not a localization.
        assert search.definitions(str(repo), "Table", max_files=5) == \
            search.definitions(str(repo), "Table", max_files=5)

    def test_a_hint_path_scopes_the_search_to_that_language(self, repo, engine):
        assert search.definitions(str(repo), "Table", hint_path="ansi.py") == ["ansi.py"]

    def test_mentions_finds_prose_not_only_code(self, repo, engine):
        assert "README.md" in search.mentions(str(repo), "cell_width", max_files=10)

    def test_count_files_agrees_with_files(self, repo, engine):
        assert search.count_files(str(repo), r"cell_width") == \
            len(search.files(str(repo), r"cell_width", max_files=200))

    def test_line_cap_is_reported_not_silently_truncated(self, repo, engine):
        out = search.lines(str(repo), r"cell_width|cellWidth|CellWidth", max_lines=1)
        assert "more matches" in out


class TestEngineSelection:
    def test_probe_reports_which_engine_is_live(self, monkeypatch):
        monkeypatch.setattr(settings, "ripgrep_enabled", False)
        out = search.probe()
        # A missing binary is not a failure — the fallback is a working engine,
        # so probe stays ok and says what the operator loses.
        assert out["ok"] is True and "git grep" in out["output"]

    def test_disabling_ripgrep_is_honoured(self, monkeypatch):
        monkeypatch.setattr(settings, "ripgrep_enabled", False)
        assert search.available() is False

    def test_ere_translation_drops_non_capturing_groups(self):
        # git grep -E treats '(?:' as literal text and reports no matches rather
        # than erroring, which is why this needs its own assertion.
        assert search._to_ere(r"(?:def|class)X") == r"(def|class)X"

    def test_ere_translation_leaves_no_gnu_extensions_behind(self):
        """`-E` is POSIX ERE *plus whatever the platform adds*, and the additions
        are not portable: a build without them reads `\\s` as a literal `s` and
        returns zero hits with no error. Every pattern the product ships must
        come out of the translation using only strict-ERE constructs."""
        for pattern, _types in search._DEFINITION_PATTERNS.values():
            ere = search._to_ere(pattern.format(n="cell_width"))
            leftover = [ext for ext in (r"\s", r"\w", r"\d", r"\b", "(?:") if ext in ere]
            assert not leftover, f"{leftover} survived translation in {ere}"
        generic = search._to_ere(search._GENERIC_DEFINITION.format(n="cell_width"))
        assert not [e for e in (r"\s", r"\w", r"\d", r"\b", "(?:") if e in generic]

    def test_the_ere_tier_still_finds_a_definition(self, repo, monkeypatch):
        """The translation has to stay *correct*, not merely extension-free —
        this is the assertion that would have caught the macOS regression."""
        monkeypatch.setattr(settings, "ripgrep_enabled", False)
        monkeypatch.setattr(search, "_supports_pcre", lambda root: False)
        assert search.definitions(str(repo), "cell_width") == ["ansi.py"]
        assert search.definitions(str(repo), "CellWidth") == ["server.go"]
        assert search.definitions(str(repo), "ZzzInventedSymbol") == []


class TestRefWorktree:
    def test_tilde_workspace_path_is_expanded(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(settings, "repos_dir", "~/.codeverdict/workspace")
        path = git_ops.workdir("https://example.com/org/repo.git")
        assert path == home / ".codeverdict" / "workspace" / "org__repo"
        assert path.parent.is_dir()

    def test_it_pins_a_checkout_at_origins_default_branch(self, repo, tmp_path, monkeypatch):
        origin = tmp_path / "origin.git"
        subprocess.run(["git", "clone", "--bare", "-q", str(repo), str(origin)],
                       check=True, capture_output=True)
        clone_root = tmp_path / "workspace"
        clone_root.mkdir()
        monkeypatch.setattr(settings, "repos_dir", str(clone_root))
        url = str(origin)
        git_ops.ensure_clone(url)

        wt = git_ops.ref_worktree(url)
        assert wt, "a cloned repo should get a ref-pinned worktree"
        main = str(git_ops.workdir(url))
        assert git_ops.rev_parse(wt, "HEAD") == git_ops.rev_parse(
            main, f"origin/{git_ops.default_branch(main)}")
        assert git_ops.ref_worktree(url) == wt, "should be reused, not recreated"

    def test_it_does_not_see_work_on_an_unmerged_branch(self, repo, tmp_path, monkeypatch):
        """The whole reason this exists: a scoping-time search must not report
        code that only exists on a previous run's agent branch as real."""
        origin = tmp_path / "origin.git"
        subprocess.run(["git", "clone", "--bare", "-q", str(repo), str(origin)],
                       check=True, capture_output=True)
        clone_root = tmp_path / "workspace"
        clone_root.mkdir()
        monkeypatch.setattr(settings, "repos_dir", str(clone_root))
        url = str(origin)
        main = git_ops.ensure_clone(url)
        _git(main, "checkout", "-q", "-b", "agent/scope-1")
        (git_ops.workdir(url) / "unmerged.py").write_text("def only_on_the_branch():\n    pass\n")
        _git(main, "add", "-A")
        _git(main, "commit", "-qm", "unmerged work")

        assert search.definitions(main, "only_on_the_branch") == ["unmerged.py"]
        wt = git_ops.ref_worktree(url)
        assert wt
        assert search.definitions(wt, "only_on_the_branch") == []

    def test_it_returns_empty_for_a_repo_that_was_never_cloned(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "repos_dir", str(tmp_path / "empty"))
        assert git_ops.ref_worktree("https://example.com/nobody/nothing") == ""
