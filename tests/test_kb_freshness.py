"""KB freshness + survival guarantees.

The invariants the pipeline's compounding knowledge depends on:

  1. Cross-run artifacts (delivery notes, distilled lessons) live in the JSON
     store and are the ONLY thing the store persists — a graph reindex never
     touches them, so history survives a rebuild by construction.
  2. Freshness brings both localization layers to origin/HEAD deterministically:
     the symbol map syncs per changed file, and the code graph reindexes when
     its SHA watermark drifts.

Plus: the symbol map's incremental per-file update must work for every language
the analyzer supports, not just Python — otherwise non-Python repos silently go
stale at the localization fallback tier.
"""

from __future__ import annotations

import pytest
from app.config import settings
from app.services.knowledge import freshness, store, symbol_map
from app.services.knowledge.facts import KnowledgeDocument

REPO = "https://github.com/example/freshness-test"


def _doc(doc_id: str, doc_type: str, name: str = "") -> KnowledgeDocument:
    return KnowledgeDocument(id=doc_id, type=doc_type, name=name or doc_id,
                             summary=f"summary of {doc_id}")


class TestStoreKeepsOnlyNotes:
    def test_notes_persist_and_reload(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "knowledge_dir", str(tmp_path))
        store.save(REPO, _doc("delivery_scope_1", "delivery_note"))
        store.save(REPO, _doc("lessons_core", "lesson"))
        ids = {e["id"] for e in store.load_index(REPO)}
        assert ids == {"delivery_scope_1", "lessons_core"}
        assert store.load(REPO, "delivery_scope_1") is not None
        assert store.load(REPO, "lessons_core") is not None


class TestFreshnessSyncsBothLayers:
    def test_reindexes_graph_and_syncs_symbol_map_on_drift(self, monkeypatch, tmp_path):
        """When origin/HEAD has moved past both watermarks, freshness rebuilds
        the symbol map and reindexes the code graph — both deterministic."""
        monkeypatch.setattr(settings, "knowledge_dir", str(tmp_path))
        monkeypatch.setattr(settings, "kb_auto_refresh", True)
        (tmp_path / ".git").mkdir()

        monkeypatch.setattr(freshness.git_ops, "workdir", lambda url: tmp_path)
        monkeypatch.setattr(freshness.git_ops, "default_branch", lambda p: "main")
        monkeypatch.setattr(freshness.git_ops, "rev_parse", lambda p, ref=None: "newsha999")

        # No symbol map yet → build; graph available with a stale watermark → reindex.
        built = {}

        def _build(url, sha):
            built["sha"] = sha
            return _FakeMap(sha)

        monkeypatch.setattr(freshness.symbol_map, "load", lambda url: None)
        monkeypatch.setattr(freshness.symbol_map, "build", _build)

        reindexed = {}

        def _reindex(url, sha):
            reindexed["sha"] = sha
            return True

        monkeypatch.setattr(freshness.graph, "available", lambda: True)
        monkeypatch.setattr(freshness.graph, "indexed_sha", lambda url: "oldsha000")
        monkeypatch.setattr(freshness.graph, "ensure_indexed", _reindex)
        monkeypatch.setattr(freshness.write_back, "reconcile_unmerged",
                            lambda url, path, on_event=None: 0)

        out = freshness.refresh_if_stale(REPO)
        assert built["sha"] == "newsha999"
        assert reindexed["sha"] == "newsha999"
        assert "graph_reindexed" in out["action"]
        assert "symbol_map_built" in out["action"]

    def test_graph_skipped_when_watermark_current(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "knowledge_dir", str(tmp_path))
        monkeypatch.setattr(settings, "kb_auto_refresh", True)
        (tmp_path / ".git").mkdir()
        monkeypatch.setattr(freshness.git_ops, "workdir", lambda url: tmp_path)
        monkeypatch.setattr(freshness.git_ops, "default_branch", lambda p: "main")
        monkeypatch.setattr(freshness.git_ops, "rev_parse", lambda p, ref=None: "sha_same")
        monkeypatch.setattr(freshness.symbol_map, "load", lambda url: _FakeMap("sha_same"))
        monkeypatch.setattr(freshness.graph, "available", lambda: True)
        monkeypatch.setattr(freshness.graph, "indexed_sha", lambda url: "sha_same")

        called = {"reindex": False}

        def _no(url, sha):
            called["reindex"] = True
            return True

        monkeypatch.setattr(freshness.graph, "ensure_indexed", _no)
        monkeypatch.setattr(freshness.write_back, "reconcile_unmerged",
                            lambda url, path, on_event=None: 0)
        out = freshness.refresh_if_stale(REPO)
        assert called["reindex"] is False  # watermark current → no reindex
        assert out["action"] == "fresh"

    def test_disabled_short_circuits(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "knowledge_dir", str(tmp_path))
        monkeypatch.setattr(settings, "kb_auto_refresh", False)
        assert freshness.refresh_if_stale(REPO) == {"action": "disabled"}


class _FakeMap:
    def __init__(self, sha):
        self.sha = sha
        self.files = {}

    def symbol_count(self):
        return 0


class TestSymbolMapMultiLanguage:
    def test_analyze_one_indexes_non_python(self, tmp_path):
        (tmp_path / "server.go").write_text(
            "package main\n\ntype Server struct{}\n\nfunc NewServer() *Server { return nil }\n")
        (tmp_path / "app.js").write_text("export function render() {}\n")
        (tmp_path / "notes.txt").write_text("not code\n")

        go_info = symbol_map._analyze_one(tmp_path, "server.go")
        assert go_info["language"] == "Go"
        assert {s["n"] for s in go_info["symbols"]} == {"Server", "NewServer"}

        js_info = symbol_map._analyze_one(tmp_path, "app.js")
        assert js_info["language"] == "JavaScript"
        assert [s["n"] for s in js_info["symbols"]] == ["render"]

        assert symbol_map._analyze_one(tmp_path, "notes.txt") is None


class TestDenseIndexResumeKeying:
    """The dense index is rebuilt from the graph's nodes, so its resume stamp
    must be the GRAPH's watermark. The two diverge exactly where it matters:
    the pipeline checks out `agent/scope-N` before calling freshness, so a
    working-tree sha would stamp the agent branch onto an index built from
    origin/HEAD — mismatching on every later run and forcing a full rebuild
    (an hour, for a repo the size of gitea) each time.

    The choice lives in `_index_sha` so it is testable on a core-only install;
    the end-to-end path needs Qdrant and is skipped where that is absent, the
    same split the download-capture tests use.
    """

    def test_the_stamp_follows_the_graph_not_the_checked_out_branch(self, monkeypatch):
        from app.services.knowledge import embed

        monkeypatch.setattr(embed.graph, "indexed_sha", lambda u: "graphsha")
        monkeypatch.setattr(embed.git_ops, "rev_parse", lambda *a, **k: "agentbranchsha")
        assert embed._index_sha("https://x/repo") == "graphsha"

    def test_an_unreadable_watermark_is_empty_not_an_exception(self, monkeypatch):
        """An empty sha disables resume (every build is fresh) — correct, and
        never a crash inside the build subprocess."""
        from app.services.knowledge import embed

        def boom(_u):
            raise OSError("no graph dir")

        monkeypatch.setattr(embed.graph, "indexed_sha", boom)
        assert embed._index_sha("https://x/repo") == ""

    def test_the_build_stamps_what_index_sha_returned(self, tmp_path, monkeypatch):
        """Faithful end-to-end version — needs the optional Qdrant stack."""
        pytest.importorskip("qdrant_client",
                            reason="qdrant ships with the [semantic] extra")
        from app.config import settings
        from app.services.knowledge import embed

        monkeypatch.setattr(settings, "qdrant_path", str(tmp_path))
        monkeypatch.setattr(embed, "_index_sha", lambda u: "graphsha")
        monkeypatch.setattr(embed, "_nodes", lambda u: [
            {"name": "f", "qn": "a.go::f", "file": "a.go", "start": 1, "end": 2,
             "label": "Function", "sig": ""}])
        monkeypatch.setattr(embed, "_read_bodies", lambda u, n: {})

        class FakeVec(list):
            def tolist(self):
                return list(self)

        monkeypatch.setattr(embed, "_get_model",
                            lambda: type("M", (), {"embed": lambda self, t, batch_size=8:
                                                   [FakeVec([0.0] * embed._DIM) for _ in t]})())

        class FakeClient:
            def collection_exists(self, c):
                return False

            def create_collection(self, c, **k):
                pass

            def upsert(self, c, points):
                pass

        monkeypatch.setattr(embed, "_get_client", FakeClient)
        embed.build_inprocess("https://x/repo")
        assert embed._read_stamp("https://x/repo") == "graphsha"

    def test_a_moved_graph_forces_a_full_rebuild(self, tmp_path, monkeypatch):
        """Point ids are derived from the qualified name, so a node whose body
        changed keeps its id — a naive resume would leave a stale vector in
        place forever."""
        pytest.importorskip("qdrant_client",
                            reason="qdrant ships with the [semantic] extra")
        from app.config import settings
        from app.services.knowledge import embed

        monkeypatch.setattr(settings, "qdrant_path", str(tmp_path))
        embed._write_stamp("https://x/repo", "oldsha")
        monkeypatch.setattr(embed, "_index_sha", lambda u: "newsha")
        monkeypatch.setattr(embed, "_nodes", lambda u: [])

        dropped: list = []

        class FakeClient:
            def collection_exists(self, c):
                return True

            def delete_collection(self, c):
                dropped.append(c)

        monkeypatch.setattr(embed, "_get_client", FakeClient)
        assert embed.build_inprocess("https://x/repo") == 0


class TestReEmbedIsVisible:
    """A sync must never go silent for minutes.

    Re-embedding is the longest stage of a refresh and the only one with no
    output of its own — on a 16k-node repo it took 10m44s and printed nothing
    between "code graph reindexed" and "re-embedded N nodes", which is the exact
    shape of a hang. These pin the progress reporting that closed that window.
    """

    def test_progress_is_throttled_but_always_reaches_100(self):
        from app.services.knowledge import freshness

        lines: list[str] = []
        report = freshness._embed_reporter(
            lambda _lvl, msg: lines.append(msg), step=5, min_gap=0.0)
        for done in range(0, 2001, 40):      # 51 callbacks
            report(done, 2000)

        assert len(lines) < 25, "a log row per batch would flood the run log"
        assert "100%" in lines[-1]
        assert "2,000/2,000 nodes" in lines[-1]

    def test_a_zero_total_reports_nothing(self):
        """Guards the division: an empty node set must not raise inside a
        callback the embed subprocess drives."""
        from app.services.knowledge import freshness

        lines: list[str] = []
        freshness._embed_reporter(lambda _lvl, msg: lines.append(msg))(5, 0)
        assert lines == []

    def test_the_interval_floor_suppresses_a_burst(self):
        from app.services.knowledge import freshness

        lines: list[str] = []
        report = freshness._embed_reporter(
            lambda _lvl, msg: lines.append(msg), step=1, min_gap=60.0)
        report(10, 100)
        report(50, 100)
        assert len(lines) == 1

    def test_the_first_line_survives_a_freshly_booted_machine(self, monkeypatch):
        """time.monotonic() counts from boot on Linux, so a 0.0 sentinel made the
        opening progress line conditional on the host's uptime — it appeared on a
        laptop and vanished on a CI runner that had been up for eight seconds."""
        from app.services.knowledge import freshness

        monkeypatch.setattr(freshness.time, "monotonic", lambda: 8.0)
        lines: list[str] = []
        freshness._embed_reporter(
            lambda _lvl, msg: lines.append(msg), min_gap=60.0)(0, 16032)
        assert len(lines) == 1
        assert "0/16,032 nodes" in lines[0]

    def test_a_failed_re_embed_says_so(self, monkeypatch, tmp_path):
        """build() swallows low RAM, a locked store and its own timeout, and
        returns 0. Silence there would leave 'the slow part of a sync' as the
        last thing the run ever said."""
        from app.services.knowledge import embed, freshness, graph, symbol_map, write_back

        monkeypatch.setattr(settings, "kb_auto_refresh", True)
        monkeypatch.setattr(settings, "knowledge_dir", str(tmp_path))
        (tmp_path / "clone" / ".git").mkdir(parents=True)

        from app.services import git_ops
        monkeypatch.setattr(git_ops, "workdir", lambda u: tmp_path / "clone")
        monkeypatch.setattr(git_ops, "default_branch", lambda p: "main")
        monkeypatch.setattr(git_ops, "rev_parse", lambda p, r: "newsha00000")
        monkeypatch.setattr(symbol_map, "load", lambda u: None)
        monkeypatch.setattr(symbol_map, "build",
                            lambda u, h: type("S", (), {"files": [], "sha": h,
                                                        "symbol_count": lambda self: 0})())
        monkeypatch.setattr(graph, "available", lambda: True)
        monkeypatch.setattr(graph, "indexed_sha", lambda u: "oldsha")
        monkeypatch.setattr(graph, "ensure_indexed", lambda u, h: True)
        monkeypatch.setattr(embed, "available", lambda: True)
        monkeypatch.setattr(embed, "build", lambda u, on_progress=None: 0)
        monkeypatch.setattr(write_back, "reconcile_unmerged", lambda *a: 0)

        seen: list[tuple[str, str]] = []
        out = freshness.refresh_if_stale("https://x/repo",
                                         on_event=lambda lvl, msg: seen.append((lvl, msg)))

        assert "re_embed_failed" in out["action"]
        assert any(lvl == "warn" and "previous index" in msg for lvl, msg in seen)
