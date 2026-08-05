"""Language registry: symbol extraction across languages, test-file detection,
the edit-time parse gate, and test-ecosystem detection/output parsing.
"""

from pathlib import Path

from app.services import lang


def _names(symbols, kind=None):
    return [s["n"] for s in symbols if kind is None or s["k"] == kind]


class TestPythonSymbols:
    SRC = (
        "import os\n\n"
        "LIMIT = 5\n\n"
        "class Client:\n"
        "    def get(self):\n"
        "        pass\n\n"
        "def helper():\n"
        "    pass\n"
    )

    def test_extracts_all_kinds(self):
        syms = lang.extract_symbols("a/b.py", self.SRC)
        assert _names(syms, "class") == ["Client"]
        assert _names(syms, "function") == ["helper"]
        assert _names(syms, "const") == ["LIMIT"]
        method = next(s for s in syms if s["k"] == "method")
        assert method["n"] == "get" and method["p"] == "Client"

    def test_unparseable_returns_empty(self):
        assert lang.extract_symbols("x.py", "def broken(:") == []


class TestJsTsSymbols:
    SRC = (
        "import { thing } from './util';\n"
        "export const MAX_RETRIES = 3;\n"
        "export class Session {\n"
        "  connect(url) {\n"
        "    if (url) { return; }\n"
        "  }\n"
        "  async close() {}\n"
        "}\n"
        "export function parse(input) {}\n"
        "export const render = (node) => node;\n"
    )

    def test_classes_functions_methods(self):
        syms = lang.extract_symbols("src/session.js", self.SRC)
        assert _names(syms, "class") == ["Session"]
        assert set(_names(syms, "function")) == {"parse", "render"}
        methods = [s for s in syms if s["k"] == "method"]
        assert {m["n"] for m in methods} == {"connect", "close"}
        assert all(m["p"] == "Session" for m in methods)
        # control-flow keywords inside the class body must not read as methods
        assert "if" not in _names(syms)

    def test_ts_interface_and_type(self):
        src = "export interface Options { a: number }\nexport type Mode = 'a' | 'b';\n"
        syms = lang.extract_symbols("x.ts", src)
        assert set(_names(syms, "class")) == {"Options", "Mode"}

    def test_imports(self):
        src = "import fs from 'fs';\nconst x = require('./local');\n"
        assert lang.extract_imports("a.js", src) == ["fs", "./local"]


class TestGoSymbols:
    SRC = (
        "package main\n\n"
        'import (\n\t"fmt"\n\t"net/http"\n)\n\n'
        "type Server struct{}\n\n"
        "func (s *Server) Start() error { return nil }\n\n"
        "func NewServer() *Server { return &Server{} }\n\n"
        "var DefaultPort = 8080\n"
    )

    def test_symbols(self):
        syms = lang.extract_symbols("server.go", self.SRC)
        assert _names(syms, "class") == ["Server"]
        assert _names(syms, "function") == ["NewServer"]
        method = next(s for s in syms if s["k"] == "method")
        assert method["n"] == "Start" and method["p"] == "Server"
        assert "DefaultPort" in _names(syms, "const")

    def test_imports_block_form(self):
        assert lang.extract_imports("server.go", self.SRC) == ["fmt", "net/http"]


class TestRustSymbols:
    SRC = (
        "pub struct Engine {}\n\n"
        "impl Engine {\n"
        "    pub fn start(&self) {}\n"
        "}\n\n"
        "pub fn run() {}\n\n"
        "pub const VERSION: &str = \"1.0\";\n"
        "use std::collections::HashMap;\n"
    )

    def test_symbols(self):
        syms = lang.extract_symbols("engine.rs", self.SRC)
        assert _names(syms, "class") == ["Engine"]
        assert _names(syms, "function") == ["run"]
        method = next(s for s in syms if s["k"] == "method")
        assert method["n"] == "start" and method["p"] == "Engine"
        assert "VERSION" in _names(syms, "const")
        assert lang.extract_imports("engine.rs", self.SRC) == ["std::collections::HashMap"]


class TestJavaRubySymbols:
    def test_java(self):
        src = (
            "package com.example;\n"
            "import java.util.List;\n"
            "public class OrderService {\n"
            "    public List<String> findAll() { return null; }\n"
            "}\n"
        )
        syms = lang.extract_symbols("OrderService.java", src)
        assert _names(syms, "class") == ["OrderService"]
        method = next(s for s in syms if s["k"] == "method")
        assert method["n"] == "findAll" and method["p"] == "OrderService"
        assert lang.extract_imports("OrderService.java", src) == ["java.util.List"]

    def test_ruby(self):
        src = "class Parser\n  def parse!(input)\n  end\nend\n\ndef helper\nend\n"
        syms = lang.extract_symbols("parser.rb", src)
        assert _names(syms, "class") == ["Parser"]
        assert _names(syms, "function") == ["helper"]
        method = next(s for s in syms if s["k"] == "method")
        assert method["n"] == "parse!" and method["p"] == "Parser"

    def test_unsupported_language_fails_open(self):
        assert lang.extract_symbols("main.swift", "func x() {}") == []


class TestRegexFallbackTier:
    """The regex extractors must stay fully functional with tree-sitter absent —
    the symbol-extraction tests above run through whichever tier is installed
    (identical expectations = the parity check), so this class forces the
    fallback path explicitly."""

    def test_js_symbols_without_treesitter(self, monkeypatch):
        monkeypatch.setattr(lang, "_ts_symbols", lambda _e, _s: None)
        syms = lang.extract_symbols("src/session.js", TestJsTsSymbols.SRC)
        assert _names(syms, "class") == ["Session"]
        assert {s["n"] for s in syms if s["k"] == "method"} == {"connect", "close"}

    def test_gate_fails_open_without_treesitter(self, monkeypatch):
        monkeypatch.setattr(lang, "_ts_parser", lambda _l: None)
        assert lang.syntax_error("x.rs", "fn broken( {") is None

    def test_empty_treesitter_result_falls_through_to_regex(self, monkeypatch):
        # A walker gap (parsed fine but yielded nothing) must not lose symbols
        # the regex tier would find.
        monkeypatch.setattr(lang, "_ts_symbols", lambda _e, _s: [])
        syms = lang.extract_symbols("src/session.js", TestJsTsSymbols.SRC)
        assert _names(syms, "class") == ["Session"]


class TestTestFileDetection:
    def test_positive(self):
        for p in ("tests/test_cli.py", "pkg/store_test.go", "src/app.test.tsx",
                  "src/__tests__/util.js", "src/test/java/FooTest.java",
                  "spec/models/user_spec.rb", "foo/bar.spec.ts"):
            assert lang.is_test_file(p), p

    def test_negative(self):
        for p in ("httpie/cli/definition.py", "src/server.go", "contest/entry.js",
                  "protester.py", "src/latest.ts"):
            assert not lang.is_test_file(p), p


class TestSyntaxGate:
    def test_js_esm_broken_rejected_when_node_present(self):
        import shutil as _sh

        import pytest as _pt
        if not _sh.which("node"):
            _pt.skip("node not installed")
        # `node --check <file>` alone silently passes ESM — the gate must
        # catch broken code in BOTH dialects.
        assert lang.syntax_error("x.js", "export function broken( { return }") is not None
        assert lang.syntax_error("x.js", "export function ok(a) { return a; }") is None
        assert lang.syntax_error("x.js", "const y = require('z');") is None  # valid CJS


    def test_python_bad(self):
        assert "line" in lang.syntax_error("x.py", "def broken(:")

    def test_python_good(self):
        assert lang.syntax_error("x.py", "def ok():\n    return 1\n") is None

    def test_treesitter_gates_when_available_else_fails_open(self):
        # With the optional tree-sitter tier these languages ARE gated; without
        # it they fail open (no checker) — both behaviors are correct.
        for rel, broken, ok in (("x.rs", "fn broken( {", "fn ok() {}\n"),
                                ("x.ts", "const x: = broken", "const x = 1;\n"),
                                ("A.java", "class A { void x( { }", "class A {}\n"),
                                ("a.rb", "def broken(\n", "def ok\nend\n")):
            if lang.treesitter_available("." + rel.rsplit(".", 1)[-1]):
                assert lang.syntax_error(rel, broken) is not None, rel
            else:
                assert lang.syntax_error(rel, broken) is None, rel
            assert lang.syntax_error(rel, ok) is None, rel


class TestRunnerDetection:
    def test_python_markers_win(self, tmp_path: Path):
        """Precedence: a repo that is genuinely Python AND carries a
        package.json (docs site, frontend assets) tests as Python. The .py file
        is load-bearing — a manifest with no Python source behind it is tooling
        config, not an ecosystem (see TestRunnerNeedsSourceNotJustAManifest)."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "x.py").write_text("x = 1\n")
        runner = lang.detect_runner(tmp_path)
        assert runner is not None and runner.kind == "python"

    def test_go(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(lang.shutil, "which", lambda _: "/usr/bin/tool")
        (tmp_path / "go.mod").write_text("module x\n")
        assert lang.detect_runner(tmp_path).kind == "go"

    def test_cargo(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(lang.shutil, "which", lambda _: "/usr/bin/tool")
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
        assert lang.detect_runner(tmp_path).kind == "cargo"

    def test_node(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(lang.shutil, "which", lambda _: "/usr/bin/tool")
        (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}')
        assert lang.detect_runner(tmp_path).kind == "node"

    def test_maven(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(lang.shutil, "which", lambda _: "/usr/bin/tool")
        (tmp_path / "pom.xml").write_text("<project/>")
        assert lang.detect_runner(tmp_path).kind == "maven"

    def test_gradle_requires_wrapper(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(lang.shutil, "which", lambda _: "/usr/bin/tool")
        (tmp_path / "build.gradle").write_text("")
        # no checked-in gradlew → the static ./gradlew command couldn't run
        assert lang.detect_runner(tmp_path) is None
        (tmp_path / "gradlew").write_text("#!/bin/sh\n")
        assert lang.detect_runner(tmp_path).kind == "gradle"

    def test_rspec(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(lang.shutil, "which", lambda _: "/usr/bin/tool")
        (tmp_path / "spec").mkdir()
        (tmp_path / "Gemfile").write_text("source 'https://rubygems.org'\n")
        assert lang.detect_runner(tmp_path).kind == "rspec"

    def test_unknown_repo(self, tmp_path: Path):
        assert lang.detect_runner(tmp_path) is None


class TestFailureParsing:
    def test_pytest_ids(self):
        out = ("FAILED tests/test_a.py::test_one - AssertionError\n"
               "ERROR tests/test_b.py::test_two\n1 failed\n")
        runner = lang.detect_runner_by_kind("python")
        assert lang.parse_failures(runner, out) == {
            "tests/test_a.py::test_one", "tests/test_b.py::test_two"}

    def test_go_ids(self):
        out = "--- FAIL: TestStart (0.00s)\nFAIL\nFAIL\texample.com/pkg\t0.01s\n"
        runner = lang.detect_runner_by_kind("go")
        assert lang.parse_failures(runner, out) == {"TestStart"}

    def test_cargo_ids(self):
        out = "test engine::tests::starts ... FAILED\n\nfailures:\n"
        runner = lang.detect_runner_by_kind("cargo")
        assert lang.parse_failures(runner, out) == {"engine::tests::starts"}

    def test_node_cannot_baseline(self):
        runner = lang.detect_runner_by_kind("node")
        assert runner.fail_re is None
        assert lang.parse_failures(runner, "anything") is None

    def test_gradle_ids(self):
        out = ("com.example.FooTest > testBar FAILED\n"
               "    org.junit.ComparisonFailure at FooTest.java:42\n"
               "5 tests completed, 1 failed\n")
        runner = lang.detect_runner_by_kind("gradle")
        assert lang.parse_failures(runner, out) == {"com.example.FooTest > testBar"}

    def test_rspec_ids(self):
        out = ("Failures:\n\nFailed examples:\n\n"
               "rspec ./spec/models/user_spec.rb:12 # User validates email\n")
        runner = lang.detect_runner_by_kind("rspec")
        assert lang.parse_failures(runner, out) == {"./spec/models/user_spec.rb:12"}

    def test_maven_cannot_baseline(self):
        runner = lang.detect_runner_by_kind("maven")
        assert runner.fail_re is None
        assert lang.parse_failures(runner, "anything") is None

    def test_no_tests_found(self):
        py = lang.detect_runner_by_kind("python")
        assert lang.no_tests_found(py, 5, "")
        assert not lang.no_tests_found(py, 1, "1 failed")
        go = lang.detect_runner_by_kind("go")
        assert lang.no_tests_found(go, 0, "?\texample.com/x\t[no test files]\n")
        rspec = lang.detect_runner_by_kind("rspec")
        assert lang.no_tests_found(rspec, 0, "No examples found.\n0 examples, 0 failures\n")
        # a real 10-example green run must NOT read as "no tests"
        assert not lang.no_tests_found(rspec, 0, "10 examples, 0 failures\n")


class TestRunnerNeedsSourceNotJustAManifest:
    """gitea ships a pyproject.toml that exists only to pin three Python
    linters (djlint, yamllint, zizmor) — 0 .py files against 3,024 .go files.
    On the manifest alone this returned `pytest -q`, so QA would have run pytest
    over a Go repo, matched nothing, and reported INCONCLUSIVE for every
    delivery of the whole benchmark."""

    def test_a_tooling_only_pyproject_does_not_make_it_a_python_repo(self, tmp_path):
        from app.services import lang

        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "gitea"\n\n[dependency-groups]\ndev = ["djlint==1.42.1"]\n')
        (tmp_path / "go.mod").write_text("module code.gitea.io/gitea\n")
        (tmp_path / "main.go").write_text("package main\n")

        runner = lang.detect_runner(tmp_path)
        assert runner is None or runner.kind != "python"

    def test_a_real_python_repo_still_resolves_to_pytest(self, tmp_path):
        from app.services import lang

        (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
        (tmp_path / "app.py").write_text("x = 1\n")
        assert lang.detect_runner(tmp_path).kind == "python"

    def test_python_source_in_a_subdirectory_counts(self, tmp_path):
        from app.services import lang

        (tmp_path / "setup.py").write_text("")
        pkg = tmp_path / "src" / "pkg"
        pkg.mkdir(parents=True)
        (pkg / "mod.py").write_text("x = 1\n")
        assert lang.detect_runner(tmp_path).kind == "python"

    def test_vendored_python_does_not_vouch_for_the_repo(self, tmp_path):
        """node_modules/ and vendor/ carry other ecosystems' code; letting them
        answer 'is this a Python repo?' is how a Go repo gets pytest."""
        from app.services import lang

        (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
        (tmp_path / "go.mod").write_text("module x\n")
        vendored = tmp_path / "node_modules" / "some-pkg"
        vendored.mkdir(parents=True)
        (vendored / "setup.py").write_text("x = 1\n")

        runner = lang.detect_runner(tmp_path)
        assert runner is None or runner.kind != "python"


class TestNodeRunnerNeedsATestScript:
    """gitea's package.json declares 106 dependencies and no `test` script — its
    frontend tests run through `make test-frontend` → vitest. Selecting npm on
    the manifest alone gave QA a runner whose only possible output is "missing
    script: test": it can never pass and never fail."""

    def _pkg(self, tmp_path, scripts):
        import json as _json
        (tmp_path / "package.json").write_text(_json.dumps({"scripts": scripts}))
        (tmp_path / "index.js").write_text("//\n")

    def test_no_test_script_means_no_runner(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lang.shutil, "which", lambda _: "/usr/bin/npm")
        self._pkg(tmp_path, {"build": "vite build"})
        assert lang.detect_runner(tmp_path) is None

    def test_the_npm_init_placeholder_does_not_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lang.shutil, "which", lambda _: "/usr/bin/npm")
        self._pkg(tmp_path, {"test": 'echo "Error: no test specified" && exit 1'})
        assert lang.detect_runner(tmp_path) is None

    def test_a_real_test_script_still_selects_npm(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lang.shutil, "which", lambda _: "/usr/bin/npm")
        self._pkg(tmp_path, {"test": "vitest run"})
        assert lang.detect_runner(tmp_path).kind == "node"

    def test_unreadable_manifest_fails_open_to_no_runner(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lang.shutil, "which", lambda _: "/usr/bin/npm")
        (tmp_path / "package.json").write_text("{not json")
        assert lang.detect_runner(tmp_path) is None
