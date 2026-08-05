"""The retrieval pipeline (services/knowledge/{retriever,expand,rerank,snippets}).

Five stages — fuse, expand, refine, rerank, snippets — of which four are
switchable. Two properties matter more than any individual stage and are
asserted throughout:

  * with every optional stage off, the pipeline is the plain hybrid search it
    grew out of. That equivalence is what makes the ablation numbers mean
    anything;
  * the prompt header names the stages that ACTUALLY ran. A header that always
    claims expansion and refinement is a lie the moment one is disabled, and
    these are disabled on purpose during measurement.

The graph binary is never required here: `graph.search` and `embed.search` are
stubbed, which is also how the no-binary degradation path gets covered.
"""

from __future__ import annotations

import subprocess

import pytest
from app.config import settings
from app.services.knowledge import expand, graph, rerank, retriever
from app.services.knowledge import snippets as snippets_mod


def hit(name, path, line=10, label="Function", **extra):
    return {"name": name, "file_path": path, "start_line": line, "label": label,
            "qualified_name": f"proj.{path.replace('/', '.')}.{name}",
            "signature": "()", **extra}


@pytest.fixture()
def stub_index(monkeypatch):
    """A deterministic two-channel index: BM25 and dense return fixed rankings
    so fusion, expansion and reranking are observed rather than guessed at."""
    bm25 = [hit("escape", "rich/markup.py"), hit("cell_len", "rich/cells.py")]
    dense = [hit("cell_len", "rich/cells.py"), hit("CellTable", "rich/cells.py", label="Class")]
    monkeypatch.setattr(graph, "available", lambda: True)
    monkeypatch.setattr(graph, "search", lambda *a, **k: list(bm25))
    monkeypatch.setattr(retriever.embed, "search", lambda *a, **k: list(dense))
    monkeypatch.setattr(expand, "neighbors", lambda *a, **k: [])
    monkeypatch.setattr(retriever, "_refine", lambda *a, **k: [])
    monkeypatch.setattr(snippets_mod, "prepare", lambda *a, **k: "")
    return {"bm25": bm25, "dense": dense}


@pytest.fixture(autouse=True)
def default_flags(monkeypatch):
    monkeypatch.setattr(settings, "graph_expansion", False)
    monkeypatch.setattr(settings, "grep_refine", False)
    monkeypatch.setattr(settings, "snippet_context", False)
    monkeypatch.setattr(settings, "rerank_mode", "deterministic")
    monkeypatch.setattr(settings, "graph_hops", 1)


class TestStageComposition:
    def test_all_optional_stages_off_is_plain_hybrid_search(self, stub_index):
        out = retriever.retrieve_context("repo", "measure the cell width", limit=5)
        assert "hybrid search + deterministic rerank" in out
        assert "expansion" not in out and "lexical refinement" not in out

    def test_the_header_names_expansion_only_when_it_found_something(
            self, stub_index, monkeypatch):
        monkeypatch.setattr(settings, "graph_expansion", True)
        monkeypatch.setattr(expand, "neighbors", lambda *a, **k: [])
        assert "expansion" not in retriever.retrieve_context("repo", "q", limit=5)

        monkeypatch.setattr(expand, "neighbors",
                            lambda *a, **k: [hit("chop_cells", "rich/cells.py", expanded=True)])
        out = retriever.retrieve_context("repo", "q", limit=5)
        assert "1-hop call-graph expansion" in out and "chop_cells" in out

    def test_expansion_carries_its_provenance_into_the_prompt(self, stub_index, monkeypatch):
        monkeypatch.setattr(settings, "graph_expansion", True)
        monkeypatch.setattr(expand, "neighbors", lambda *a, **k: [
            hit("chop_cells", "rich/cells.py", expanded=True, via="CALLS from cell_len")])
        out = retriever.retrieve_context("repo", "q", limit=6)
        # "you got this because cell_len calls it" is the difference between
        # context and an unexplained pile of code.
        assert "CALLS from cell_len" in out

    def test_the_hop_count_in_the_header_reflects_the_setting(self, stub_index, monkeypatch):
        monkeypatch.setattr(settings, "graph_expansion", True)
        monkeypatch.setattr(settings, "graph_hops", 2)
        monkeypatch.setattr(expand, "neighbors", lambda *a, **k: [hit("x", "a.py", expanded=True)])
        assert "2-hop call-graph expansion" in retriever.retrieve_context("repo", "q", limit=5)

    def test_an_empty_index_yields_nothing_rather_than_a_header(self, monkeypatch):
        monkeypatch.setattr(graph, "available", lambda: False)
        assert retriever.retrieve_context("repo", "anything") == ""


class TestLexicalRefinement:
    def test_only_identifier_shaped_terms_are_grepped(self):
        # Grepping ordinary prose would return the whole repo; grepping
        # `cell_len` or `--no-color` returns the thing itself.
        assert retriever._identifiers("measure the printable width") == []
        assert set(retriever._identifiers("cell_len and NO_COLOR and --timeout")) == {
            "cell_len", "NO_COLOR", "timeout"}

    def test_plan_symbols_are_grepped_even_from_a_prose_query(self):
        idents = retriever._identifiers("make the borders line up",
                                        ["rich/table.py::_measure_column"])
        assert idents == ["_measure_column"]


class TestDeterministicRerank:
    def test_a_prose_word_does_not_promote_a_name_that_contains_it(self):
        """The regression this scoring exists to prevent.

        A query saying "…ansi escape sequences" once promoted `markup.escape`
        over the entire correct cluster, purely because the English word
        "escape" appears in its name — reproducing exactly the confident-but-
        wrong lexical failure the RRF dense weighting was tuned to suppress.
        """
        hits = [hit("cell_len", "rich/cells.py"), hit("escape", "rich/markup.py")]
        ranked = rerank.rank(hits, "measure printable cell width ignoring ansi escape sequences")
        assert [h["name"] for h in ranked] == ["cell_len", "escape"]

    def test_an_identifier_in_the_query_does_promote_its_symbol(self):
        hits = [hit("render", "rich/console.py"), hit("cell_len", "rich/cells.py")]
        ranked = rerank.rank(hits, "cell_len returns the wrong width")
        assert ranked[0]["name"] == "cell_len"

    def test_a_planner_verified_symbol_outranks_the_fused_order(self):
        hits = [hit("render", "rich/console.py"), hit("_measure_column", "rich/table.py")]
        ranked = rerank.rank(hits, "the borders are crooked",
                             plan_symbols=["rich/table.py::_measure_column"])
        assert ranked[0]["name"] == "_measure_column"

    def test_every_test_ranks_below_every_source_hit(self):
        """A band, not a nudge. A prose query puts test names near the top in
        NUMBERS (three of a top five on rich-A), which no per-item penalty
        reliably fixes."""
        hits = [hit(f"test_{i}", f"tests/test_{i}.py") for i in range(4)]
        hits += [hit("cell_len", "rich/cells.py"), hit("chop_cells", "rich/cells.py")]
        names = [h["name"] for h in rerank.rank(hits, "measure the printable cell width")]
        assert names[:2] == ["cell_len", "chop_cells"]
        assert all(n.startswith("test_") for n in names[2:])

    def test_documentation_ranks_below_tests(self):
        hits = [hit("cells", "docs/source/cells.md", label=""),
                hit("test_cells", "tests/test_cells.py"),
                hit("cell_len", "rich/cells.py")]
        assert [h["file_path"] for h in rerank.rank(hits, "cell width")] == \
            ["rich/cells.py", "tests/test_cells.py", "docs/source/cells.md"]

    def test_a_plan_that_targets_a_test_file_lifts_it_out_of_the_test_band(self):
        # "Add a regression test for X" makes the test file the work. A prior
        # derived from paths must not overrule a decision made against the graph.
        hits = [hit("cell_len", "rich/cells.py"),
                hit("test_cell_len", "tests/test_cells.py")]
        ranked = rerank.rank(hits, "add a regression test",
                             plan_symbols=["tests/test_cells.py::test_cell_len"])
        assert ranked[0]["name"] == "test_cell_len"

    def test_an_expanded_neighbour_ranks_below_anything_that_matched(self):
        hits = [hit("chop_cells", "rich/cells.py", expanded=True, hop=1),
                hit("cell_len", "rich/cells.py")]
        assert [h["name"] for h in rerank.rank(hits, "cell width")] == \
            ["cell_len", "chop_cells"]

    def test_an_explicit_is_test_flag_beats_the_path_guess(self):
        # The graph knows better than the filename when it says so.
        hits = [hit("looks_like_source", "rich/helper.py", is_test=True),
                hit("plain", "rich/other.py")]
        assert rerank.rank(hits, "helper")[0]["name"] == "plain"

    def test_it_never_returns_more_than_the_limit(self):
        hits = [hit(f"f{i}", f"m{i}.py") for i in range(30)]
        assert len(rerank.rank(hits, "f1", limit=5)) == 5

    def test_an_empty_pool_is_an_empty_result(self):
        assert rerank.rank([], "anything") == []


class TestRerankTierFallback:
    def test_an_llm_tier_that_fails_keeps_the_deterministic_order(self, monkeypatch):
        monkeypatch.setattr(settings, "rerank_mode", "llm")
        monkeypatch.setattr(rerank, "_llm_rank", lambda *a, **k: None)
        hits = [hit("test_x", "tests/test_a.py"), hit("x", "src/a.py")]
        # Falls back to deterministic — which still demotes the test — rather
        # than to the unranked input order.
        assert [h["name"] for h in rerank.rank(hits, "x")] == ["x", "test_x"]

    def test_a_partial_llm_order_keeps_the_dropped_candidates(self, monkeypatch):
        monkeypatch.setattr(settings, "rerank_mode", "llm")
        monkeypatch.setattr(rerank, "_llm_rank", lambda hits, q: [1])
        hits = [hit("a", "a.py"), hit("b", "b.py"), hit("c", "c.py")]
        names = [h["name"] for h in rerank.rank(hits, "q")]
        assert names[0] == "b" and set(names) == {"a", "b", "c"}

    def test_deterministic_is_always_an_available_tier(self):
        assert "deterministic" in rerank.available_tiers()


class TestSnippets:
    @pytest.fixture()
    def graph_with_source(self, monkeypatch):
        def _snippet(repo, qn):
            return {"source": "def small():\n    return 1\n" if "small" in qn
                    else "def huge():\n" + "    pass\n" * 4000}
        monkeypatch.setattr(graph, "available", lambda: True)
        monkeypatch.setattr(snippets_mod.graph, "snippet", _snippet)
        monkeypatch.setattr(snippets_mod.graph, "indexed_sha", lambda r: "abc123")
        monkeypatch.setattr(settings, "snippet_context", True)

    def test_a_small_node_is_inlined_verbatim(self, graph_with_source):
        out = snippets_mod.prepare("repo", [hit("small", "a.py")])
        assert "def small():" in out and "[source]" in out

    def test_an_oversized_node_is_summarized_not_truncated(self, graph_with_source, monkeypatch):
        monkeypatch.setattr(settings, "snippet_summarize", True)
        monkeypatch.setattr(snippets_mod, "summarize", lambda *a, **k: "It does the thing.")
        out = snippets_mod.prepare("repo", [hit("huge", "b.py")])
        assert "It does the thing." in out and "summary" in out

    def test_an_oversized_node_falls_back_to_truncated_source(self, graph_with_source, monkeypatch):
        # No summarizer available: truncated real source is worse than a summary
        # but better than silence, and it says which it is.
        monkeypatch.setattr(settings, "snippet_summarize", False)
        out = snippets_mod.prepare("repo", [hit("huge", "b.py")])
        assert "truncated" in out

    def test_the_budget_is_respected(self, graph_with_source, monkeypatch):
        monkeypatch.setattr(settings, "snippet_summarize", False)
        out = snippets_mod.prepare("repo", [hit("huge", f"f{i}.py") for i in range(6)],
                                   budget=1500)
        assert len(out) < 2500

    def test_a_hit_with_no_graph_node_is_skipped_not_invented(self, graph_with_source):
        # A lexical hit knows a location but not a node; quoting a line range for
        # it would be guessing at content, which is what this layer avoids.
        lexical = hit("thing", "c.py")
        lexical["qualified_name"] = ""
        assert snippets_mod.prepare("repo", [lexical]) == ""

    def test_it_is_off_when_the_setting_is_off(self, graph_with_source, monkeypatch):
        monkeypatch.setattr(settings, "snippet_context", False)
        assert snippets_mod.prepare("repo", [hit("small", "a.py")]) == ""


class TestExpansion:
    def test_it_is_empty_without_a_graph(self, monkeypatch):
        monkeypatch.setattr(graph, "available", lambda: False)
        assert expand.neighbors("repo", ["Table"]) == []
        assert "unavailable" in expand.ego("repo", "Table")

    def test_zero_hops_expands_nothing(self, monkeypatch):
        monkeypatch.setattr(graph, "available", lambda: True)
        assert expand.neighbors("repo", ["Table"], hops=0) == []

    def test_structural_neighbours_are_dropped(self, monkeypatch):
        """A module "calls" a symbol, but naming rich/table.py as a caller tells
        a coding agent nothing it can act on — and crowds out the function that
        genuinely calls it."""
        monkeypatch.setattr(graph, "available", lambda: True)
        monkeypatch.setattr(expand, "_rows", lambda repo, names, *, incoming: [
            ["Table", "CALLS", "rich/table.py", '["Module"]', "rich/table.py", "1", "p.t"],
            ["Table", "CALLS", "make_card", '["Function"]', "rich/__main__.py", "39", "p.m"],
        ])
        names = [n["name"] for n in expand.neighbors("repo", ["Table"])]
        assert names == ["make_card"]

    def test_a_graph_failure_degrades_to_no_expansion(self, monkeypatch):
        monkeypatch.setattr(graph, "available", lambda: True)
        monkeypatch.setattr(expand, "_rows",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("index gone")))
        assert expand.neighbors("repo", ["Table"]) == []


class TestAgainstARealRepo:
    """One end-to-end pass over a real git repo, no graph binary — the tier a
    default install actually runs on."""

    @pytest.fixture()
    def repo(self, tmp_path):
        (tmp_path / "cells.py").write_text(
            "def cell_len(text):\n    return len(text)\n")
        for cmd in (["init", "-q"], ["add", "-A"],
                    ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "i"]):
            subprocess.run(["git", *cmd], cwd=tmp_path, check=True, capture_output=True)
        return tmp_path

    def test_refinement_finds_a_symbol_the_index_never_saw(self, repo):
        rows = retriever._refine("repo", str(repo), ["cell_len"], set())
        assert rows and rows[0]["file_path"] == "cells.py"
        assert rows[0]["via"] == "lexical match on cell_len"

    def test_refinement_skips_locations_already_known(self, repo):
        assert retriever._refine("repo", str(repo), ["cell_len"], {"cells.py:cell_len"}) == []
