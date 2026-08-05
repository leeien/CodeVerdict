"""Local dense-embedding channel: availability gating, memory guards, and the
weighted RRF fusion that merges it with the graph's BM25.

These run without the [semantic] extra installed and without touching a model —
the fusion maths and the fail-open paths are what must never regress.
"""

import contextlib
import io
import types

import pytest
from app.config import settings
from app.services.knowledge import embed, retriever


@pytest.fixture(autouse=True)
def _restore():
    saved = (settings.local_embeddings, settings.rrf_k, settings.rrf_dense_weight,
             settings.embedding_model)
    yield
    (settings.local_embeddings, settings.rrf_k, settings.rrf_dense_weight,
     settings.embedding_model) = saved


# --- availability + fail-open -------------------------------------------------

def test_disabled_means_unavailable():
    settings.local_embeddings = False
    assert embed.enabled() is False
    assert embed.available() is False


def test_search_returns_empty_when_disabled():
    settings.local_embeddings = False
    assert embed.search("https://github.com/x/y", "anything") == []


def test_build_returns_zero_when_disabled():
    settings.local_embeddings = False
    assert embed.build("https://github.com/x/y") == 0


def test_free_mb_reports_a_number():
    mb = embed.free_mb()
    assert mb == -1 or mb > 0


def test_ram_guard_blocks_when_tight(monkeypatch):
    """Under memory pressure the channel declines rather than inviting the OOM
    killer — retrieval degrades to BM25, it does not crash."""
    monkeypatch.setattr(embed, "free_mb", lambda: 100)
    assert embed._enough_ram() is False
    monkeypatch.setattr(embed, "free_mb", lambda: 4000)
    assert embed._enough_ram() is True


def test_threads_scale_with_available_ram(monkeypatch):
    """A roomy machine is not throttled to a dev-box budget.

    Core count is pinned here too: it is now half the decision, so leaving it to
    whatever the test host happens to have made this assert the machine rather
    than the policy.
    """
    import os as _os

    monkeypatch.setattr(_os, "cpu_count", lambda: 16)

    monkeypatch.setattr(embed, "free_mb", lambda: 1200)
    tight = embed._threads()
    monkeypatch.setattr(embed, "free_mb", lambda: 2500)
    middling = embed._threads()
    monkeypatch.setattr(embed, "free_mb", lambda: 12000)
    roomy = embed._threads()

    assert tight == 1
    assert tight < middling < roomy
    assert roomy <= embed._MAX_THREADS


def test_doc_text_is_capped_and_prefixed():
    """The nomic task prefix is required, and the cap is what keeps the ONNX
    arena flat (measured: 2KB inputs → 1.2GB and climbing; 700B → 459MB flat)."""
    node = {"label": "Class", "name": "Table", "sig": "(rows, cols)",
            "file": "rich/table.py"}
    text = embed._doc_text(node, "x" * 5000)
    assert text.startswith("search_document: rich/table.py")
    assert "Class Table(rows, cols)" in text
    assert len(text) <= embed._DOC_CHARS


def test_point_id_is_stable_and_unique():
    a = embed._point_id("pkg.mod.Klass.method")
    assert a == embed._point_id("pkg.mod.Klass.method")
    assert a != embed._point_id("pkg.mod.Klass.other")


# --- weighted RRF fusion ------------------------------------------------------

def _hit(name, file, line=1):
    return {"name": name, "file_path": file, "start_line": line,
            "qualified_name": f"{file}::{name}", "label": "Function"}


class TestWeightedRRF:
    def test_weight_lets_dense_outrank_confident_bm25(self):
        """The measured failure mode: BM25's wrong-but-confident top hits crowd
        out dense's correct lower-ranked ones under equal weights."""
        bm25 = [_hit("wrong_a", "a.py"), _hit("wrong_b", "b.py"),
                _hit("wrong_c", "c.py"), _hit("wrong_d", "d.py")]
        dense = [_hit("wrong_a", "a.py")] * 0 + [
            _hit("x1", "x.py"), _hit("x2", "x.py"), _hit("x3", "x.py"),
            _hit("right", "target.py")]

        equal = retriever._rrf([(bm25, 1.0), (dense, 1.0)], 60)
        weighted = retriever._rrf([(bm25, 1.0), (dense, 2.0)], 60)

        def rank_of(scores, key):
            return sorted(scores, key=lambda k: -scores[k]).index(key)

        key = "target.py::right"
        assert rank_of(weighted, key) < rank_of(equal, key)

    def test_zero_weight_channel_is_ignored(self):
        bm25 = [_hit("only", "a.py")]
        scores = retriever._rrf([(bm25, 1.0), ([_hit("d", "d.py")], 0.0)], 60)
        assert "d.py::d" not in scores
        assert "a.py::only" in scores

    def test_agreement_between_channels_boosts(self):
        """A node both rankers like should beat one only a single ranker likes."""
        shared = _hit("shared", "s.py")
        bm25 = [shared, _hit("bm_only", "b.py")]
        dense = [shared, _hit("dn_only", "d.py")]
        scores = retriever._rrf([(bm25, 1.0), (dense, 2.0)], 60)
        assert scores["s.py::shared"] > scores["b.py::bm_only"]
        assert scores["s.py::shared"] > scores["d.py::dn_only"]

    def test_builtin_and_null_nodes_are_dropped(self):
        junk = [{"name": "print", "file_path": "<python-builtins>",
                 "qualified_name": "builtins.print"}]
        scores = retriever._rrf([(junk, 1.0)], 60)
        assert scores == {}


class TestDownloadProgressIsContained:
    """fastembed downloads the model with a bare tqdm and hardcodes
    show_progress=True. tqdm redraws with '\\r', which collapses to one line only
    on a TTY — under the CLI's Rich spinner every 1KB chunk became a NEW line, so
    the first run printed thousands of 'Downloading bytes: ####' rows and buried
    the interface."""

    def test_direct_stream_writes_are_captured(self):
        """The mechanism, without depending on the optional stack: anything a
        library writes straight to stdout/stderr must land in the buffer."""
        import sys

        from app.services.knowledge import embed

        with embed._quiet_progress() as buf:
            print("a stray library print")
            sys.stderr.write("\rDownloading bytes: ####  12.3MB\r")
        assert "a stray library print" in buf.getvalue()
        assert "Downloading bytes" in buf.getvalue()
        # Streams are restored even though the library wrote to them.
        assert sys.stdout is not buf and sys.stderr is not buf

    def test_a_real_tqdm_bar_never_reaches_the_terminal(self):
        """The faithful version. tqdm arrives with fastembed, so it is absent on
        a core-only install (which is what CI runs) — skip rather than pretend."""
        import sys

        tqdm = pytest.importorskip("tqdm", reason="tqdm ships with the [semantic] extra")
        from app.services.knowledge import embed

        with embed._quiet_progress() as buf:
            for _ in tqdm.tqdm(range(5), total=5):
                pass
        assert buf.getvalue(), "the bar must be captured, not merely suppressed"
        assert sys.stdout is not buf and sys.stderr is not buf

    def test_streams_are_restored_when_the_load_raises(self):
        import sys

        from app.services.knowledge import embed

        out, err = sys.stdout, sys.stderr
        with contextlib.suppress(RuntimeError), embed._quiet_progress():
            raise RuntimeError("download failed")
        assert sys.stdout is out and sys.stderr is err

    def test_logging_still_reaches_the_real_stream(self):
        """logging.StreamHandler binds sys.stderr at construction, so swapping it
        afterwards must not swallow the pipeline's own log output."""
        import logging

        from app.services.knowledge import embed

        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        log = logging.getLogger("embed-test")
        log.addHandler(_Capture())
        log.setLevel(logging.INFO)
        try:
            with embed._quiet_progress():
                log.info("still logged")
        finally:
            log.handlers.clear()
        assert records == ["still logged"]


class TestEmbedUsesRealGraphHelpers:
    """`_nodes` called `graph._int` / `graph._label` long after they were renamed
    to `as_int` / `node_label`. Nothing caught it: the AttributeError happened
    inside the build subprocess, whose entrypoint prints ERROR= and returns
    VECTORS=0, which `build()` reports as "no vectors" — indistinguishable from
    a repo with nothing to embed. The dense channel was silently dead and
    retrieval had quietly degraded to BM25-only."""

    def test_every_graph_attribute_embed_references_exists(self):
        import re
        from pathlib import Path

        from app.services.knowledge import embed, graph

        source = Path(embed.__file__).read_text(encoding="utf-8")
        missing = [n for n in sorted(set(re.findall(r"\bgraph\.([A-Za-z_]\w*)", source)))
                   if not hasattr(graph, n)]
        assert missing == []

    def test_nodes_maps_a_graph_row_without_touching_missing_helpers(self, monkeypatch):
        """The mapping itself, over a row shaped like the graph's cypher output:
        labels(n) arrives as a JSON string, line numbers as strings."""
        from app.services.knowledge import embed

        monkeypatch.setattr(embed.graph, "cypher", lambda url, q: [
            ["compile", "glob.ts::compile", "web_src/js/utils/glob.ts",
             "62", "80", '["Method"]', "(s: string)"],
        ])
        nodes = embed._nodes("https://x/repo")
        assert nodes == [{"name": "compile", "qn": "glob.ts::compile",
                          "file": "web_src/js/utils/glob.ts", "start": 62, "end": 80,
                          "label": "Method", "sig": "(s: string)"}]

    def test_a_row_missing_its_qualified_name_is_dropped(self, monkeypatch):
        """qn is the point id and the handle snippet() needs — a row without one
        cannot be stored or resolved later."""
        from app.services.knowledge import embed

        monkeypatch.setattr(embed.graph, "cypher", lambda url, q: [
            ["orphan", None, "a.go", "1", "2", '["Function"]', ""],
            ["kept", "a.go::kept", "a.go", "3", "4", '["Function"]', ""],
        ])
        assert [n["name"] for n in embed._nodes("https://x/repo")] == ["kept"]


class TestBuildStreamsProgress:
    """Embedding is the longest stage of a KB build — ~20 minutes for gitea's
    15,991 nodes — and it reported a single static 88% for all of it, which
    reads exactly like a hang. The child's stdout is streamed so the number can
    move while the work happens."""

    def _fake_popen(self, lines, stderr=""):
        class FakeProc:
            def __init__(self, *a, **k):
                self.stdout = iter(lines)
                self.stderr = io.StringIO(stderr)

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass
        return FakeProc

    def _patch(self, monkeypatch, lines, stderr=""):
        from app.services.knowledge import embed

        monkeypatch.setattr(embed, "available", lambda: True)
        monkeypatch.setattr(embed.graph, "available", lambda: True)
        monkeypatch.setattr(embed, "_enough_ram", lambda: True)
        monkeypatch.setattr(embed.subprocess, "Popen", self._fake_popen(lines, stderr))
        return embed

    def test_progress_lines_reach_the_callback_and_vectors_are_returned(self, monkeypatch):
        embed = self._patch(monkeypatch, [
            "PROGRESS=200/1000\n", "PROGRESS=600/1000\n", "VECTORS=1000\n"])
        seen = []
        assert embed.build("https://x/r", on_progress=lambda d, t: seen.append((d, t))) == 1000
        assert seen == [(200, 1000), (600, 1000)]

    def test_a_build_with_no_callback_still_works(self, monkeypatch):
        """The pipeline passes one; a bare call from a script does not."""
        embed = self._patch(monkeypatch, ["PROGRESS=1/2\n", "VECTORS=2\n"])
        assert embed.build("https://x/r") == 2

    def test_a_malformed_progress_line_is_ignored_not_fatal(self, monkeypatch):
        embed = self._patch(monkeypatch, [
            "PROGRESS=notanumber\n", "PROGRESS=5/10\n", "VECTORS=10\n"])
        seen = []
        assert embed.build("https://x/r", on_progress=lambda d, t: seen.append((d, t))) == 10
        assert seen == [(5, 10)]

    def test_a_child_that_reports_nothing_returns_zero(self, monkeypatch):
        embed = self._patch(monkeypatch, ["some noise\n"], stderr="ImportError: boom")
        assert embed.build("https://x/r") == 0


class TestPipelineMapsEmbeddingOntoTheBar:
    def test_embedding_progress_moves_the_percentage(self, monkeypatch):
        """85→99 belongs to embedding, so the bar advances during the long pole
        instead of parking on one tick."""
        from app.services.knowledge import pipeline

        seen: list[tuple[str, int]] = []

        def fake_build(repo_url, on_progress=None):
            for done in (0, 8000, 15991):
                on_progress(done, 15991)
            return 15991

        monkeypatch.setattr(pipeline.graph, "available", lambda: True)
        monkeypatch.setattr(pipeline.graph, "ensure_indexed", lambda *a: True)
        monkeypatch.setattr(pipeline.graph, "architecture",
                            lambda u: {"total_nodes": 120521, "total_edges": 317530})
        monkeypatch.setattr(pipeline.embed, "available", lambda: True)
        monkeypatch.setattr(pipeline.embed, "build", fake_build)
        monkeypatch.setattr(pipeline.git_ops, "rev_parse", lambda p: "abc123")
        monkeypatch.setattr(pipeline.symbol_map, "build", lambda *a: None)
        monkeypatch.setattr(pipeline.retriever, "overview", lambda u: "")
        monkeypatch.setattr(pipeline.analyzer, "analyze_repo",
                            lambda u: types.SimpleNamespace(files={"a.go": {}}))
        from app.services.knowledge import freshness
        monkeypatch.setattr(freshness, "write_meta", lambda *a, **k: None)

        result = pipeline.generate("https://x/r", progress=lambda s, p: seen.append((s, p)))

        # One opening tick announces the stage before any batch has landed; the
        # counted updates are the ones that carry "done/total".
        assert [p for s, p in seen if s.startswith("Embedding") and "/" not in s] == [85]
        assert [p for s, p in seen if "/" in s] == [85, 92, 99]
        assert result.counts["vectors"] == 15991
        # The node counts ride along so /kb can say which channel is live.
        assert "15,991" in [s for s, _ in seen if s.startswith("Embedding")][-1]


class TestHardwareScaling:
    """The old tiers topped out at 4 threads and batch 8 regardless of the
    machine, so a 32-core workstation indexed a repo at laptop speed — the
    dev-box budget this was tuned under had become everyone's ceiling."""

    def _at(self, monkeypatch, cores, mb):
        import os as _os

        from app.services.knowledge import embed

        monkeypatch.setattr(embed, "free_mb", lambda: mb)
        monkeypatch.setattr(_os, "cpu_count", lambda: cores)
        return embed._threads(), embed._batch()

    def test_a_big_machine_is_actually_used(self, monkeypatch):
        threads, batch = self._at(monkeypatch, 32, 64000)
        assert threads == 16          # _MAX_THREADS, where ONNX scaling flattens
        assert batch == 64

    def test_a_small_machine_stays_conservative(self, monkeypatch):
        threads, batch = self._at(monkeypatch, 12, 1110)
        assert threads == 1
        assert batch == 8

    def test_cores_cap_threads_when_ram_is_plentiful(self, monkeypatch):
        """RAM can back more than the CPU has; the smaller of the two wins, and
        one core is left for the rest of the system."""
        threads, _ = self._at(monkeypatch, 4, 64000)
        assert threads == 3

    def test_ram_caps_threads_when_cores_are_plentiful(self, monkeypatch):
        threads, _ = self._at(monkeypatch, 64, 2300)
        assert threads == 3           # (2300-900)//700 + 1

    def test_the_reserved_floor_is_never_spent_on_threads(self, monkeypatch):
        """_MIN_FREE_MB is the OOM guard; budgeting threads out of it is exactly
        what this module exists to avoid."""
        threads, _ = self._at(monkeypatch, 64, 900)
        assert threads == 1

    def test_an_unknown_platform_stays_modest(self, monkeypatch):
        threads, batch = self._at(monkeypatch, 64, -1)
        assert threads == 4
        assert batch == 8

    def test_a_single_core_box_still_gets_one_thread(self, monkeypatch):
        threads, _ = self._at(monkeypatch, 1, 64000)
        assert threads == 1


class TestBuildReleasesTheStoreLock:
    """Embedded Qdrant takes an EXCLUSIVE file lock, and the build runs in a
    SUBPROCESS. A process that has answered even one search already holds that
    lock, so the child died on "Storage folder is already accessed by another
    instance" and reported zero vectors. Observed live: a graph reindex on
    gitea left the dense index stamped at the previous SHA, because the rebuild
    could never open the store."""

    def test_the_cached_client_is_released_before_the_child_starts(self, monkeypatch):
        from app.services.knowledge import embed

        monkeypatch.setattr(embed, "available", lambda: True)
        monkeypatch.setattr(embed.graph, "available", lambda: True)
        monkeypatch.setattr(embed, "_enough_ram", lambda: True)

        closed_before_spawn = []

        class FakeProc:
            def __init__(self, *a, **k):
                closed_before_spawn.append(embed._client is None)
                self.stdout = iter(["VECTORS=5\n"])
                self.stderr = io.StringIO("")

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        embed._client = object()          # stand in for a live, lock-holding client
        monkeypatch.setattr(embed.subprocess, "Popen", FakeProc)
        try:
            assert embed.build("https://x/repo") == 5
        finally:
            embed._client = None
        assert closed_before_spawn == [True], "the lock was still held when the child spawned"


class TestIncrementalReEmbed:
    """A merge that touches four files used to cost a full rebuild — an hour of
    CPU to redo 15,987 vectors that were still correct. Only the files the diff
    touched are re-embedded now."""

    class FakePoint:
        def __init__(self, pid, file):
            self.id, self.payload = pid, {"file": file}

    def _client(self, stored):
        points = [self.FakePoint(pid, f) for pid, f in stored.items()]

        class FakeClient:
            deleted: list = []

            def collection_exists(self, c):
                return True

            def scroll(self, c, limit=4096, offset=None, **k):
                return points, None

            def delete(self, c, points_selector):
                FakeClient.deleted.extend(points_selector)

        FakeClient.deleted = []
        return FakeClient()

    def _run(self, monkeypatch, stored, changed, deleted):
        from app.services.knowledge import embed

        monkeypatch.setattr(embed.git_ops, "changed_files",
                            lambda p, a, b: (changed, deleted))
        monkeypatch.setattr(embed.git_ops, "workdir", lambda u: "/tmp/x")
        monkeypatch.setattr(embed, "_write_stamp", lambda u, s: None)
        client = self._client(stored)
        surviving, fresh = embed._incremental(client, "c", "https://x/r", "old", "new", 4)
        return surviving, fresh, client

    def test_only_touched_files_lose_their_vectors(self, monkeypatch):
        stored = {"id1": "a.go", "id2": "a.go", "id3": "b.go", "id4": "c.go"}
        surviving, fresh, client = self._run(monkeypatch, stored, ["a.go"], [])
        assert fresh is False
        assert surviving == {"id3", "id4"}
        assert sorted(type(client).deleted) == ["id1", "id2"]

    def test_deleted_files_lose_their_orphaned_vectors(self, monkeypatch):
        """A node that no longer exists is never re-embedded, so its vector
        would otherwise survive forever and keep matching queries."""
        stored = {"id1": "gone.go", "id2": "kept.go"}
        surviving, _, client = self._run(monkeypatch, stored, [], ["gone.go"])
        assert surviving == {"id2"}
        assert type(client).deleted == ["id1"]

    def test_an_uncomputable_range_falls_back_to_a_full_rebuild(self, monkeypatch):
        """An old SHA gc'd away or a shallow clone — trusting a partial diff
        there would silently leave stale vectors behind."""
        from app.services.knowledge import embed

        monkeypatch.setattr(embed.git_ops, "changed_files", lambda p, a, b: None)
        monkeypatch.setattr(embed.git_ops, "workdir", lambda u: "/tmp/x")
        surviving, fresh = embed._incremental(self._client({}), "c", "https://x/r",
                                              "old", "new", 4)
        assert fresh is True and surviving == set()

    def test_an_empty_diff_keeps_everything(self, monkeypatch):
        stored = {"id1": "a.go", "id2": "b.go"}
        surviving, fresh, client = self._run(monkeypatch, stored, [], [])
        assert fresh is False
        assert surviving == {"id1", "id2"}
        assert type(client).deleted == []
