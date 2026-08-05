"""Local dense-embedding channel — real semantic recall over the graph's nodes.

The code graph's own `semantic_query` uses *static token embeddings* (a frozen
word-vector table), which measured as near-useless (it returned Makefiles for
"bold coloured text"). This module replaces it with a real contextual embedding
model run locally via fastembed (ONNX, CPU, no Docker, no API key), stored in an
embedded Qdrant collection per repo and fused with the graph's BM25 via RRF.

Design (hybrid recipe borrowed from SocratiCode, adapted to our graph):
  * We embed the GRAPH'S NODES (functions/methods/classes). The graph already
    did AST-accurate chunking, so we reuse its symbol inventory instead of
    re-chunking files: one vector per symbol, keyed by qualified name, and every
    hit maps straight back to a real file:line plus its callers.
  * Doc text is `search_document: <file>\n<label> <name><sig>\n<head of body>`;
    query text is `search_query: <q>`. Those task prefixes are REQUIRED by the
    nomic family — omitting them measurably degrades retrieval.

Memory discipline (this runs on small dev boxes, and ONNX is greedy):
  ONNX Runtime's CPU arena grows with SEQUENCE LENGTH and never gives it back —
  measured 1.2 GB and climbing with 2 KB inputs, versus a flat 459 MB plateau
  with ~600-char inputs at batch 8, single-threaded. So we:
    1. cap doc text (`_DOC_CHARS`) — a symbol's file, signature, docstring and
       body head carry nearly all of its semantic signal anyway,
    2. embed in small batches, single-threaded,
    3. run the whole BUILD in a subprocess, so every byte is returned to the OS
       when it exits (the server process never holds the model during a build),
    4. refuse to load the model when free RAM is under `_MIN_FREE_MB`, degrading
       to BM25-only instead of inviting the OOM killer.
Nothing here is a quality compromise: the model and its vectors are unchanged.

Everything is optional and fail-open: without the [semantic] extra, with the
feature off, or under memory pressure, `search()` returns [] and retrieval falls
back to the graph's BM25 channel.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from ...config import settings
from .. import git_ops
from . import graph

logger = logging.getLogger(__name__)

_DIM = 768          # nomic-embed-text / jina-code family
_DOC_CHARS = 700    # per-node doc text cap — keeps the ONNX arena flat
_BODY_LINES = 12    # source lines past the definition folded into the doc text
_BATCH = 8          # small embed batches: bounded ONNX arena
_UPSERT_EVERY = 200  # but flush to Qdrant in big chunks — local writes are the
                     # slow part, and points are tiny in RAM (768 floats each)
_MIN_FREE_MB = 900  # refuse to load the model below this much available RAM
# Threads and batch size are the two arena multipliers, and both scale with the
# machine. The old fixed tiers topped out at 4 threads and batch 8 no matter what
# they ran on, so a 32-core workstation indexed a repo at laptop speed — the
# dev-box budget this was tuned under had become everyone's ceiling.
#
# RAM is the real limit (each intra-op thread carries its own arena), cores are
# the other, and the smaller wins. One core is left for the rest of the system so
# a build never makes the machine unusable.
_ARENA_MB_PER_THREAD = 700  # measured plateau per thread at _DOC_CHARS=700
_MAX_THREADS = 16           # past this, ONNX intra-op scaling flattens
_BATCH_TIERS = ((8000, 64), (4000, 32), (2000, 16))  # (min free MB, batch)
_BUILD_TIMEOUT = 3600
_UUID_NS = uuid.UUID("6ba7b812-9dad-11d1-80b4-00c04fd430c8")

_model = None       # cached fastembed TextEmbedding (query path only)
_model_name = ""
_client = None      # cached embedded Qdrant client


# --- Availability + memory guard ---------------------------------------------

def enabled() -> bool:
    return bool(settings.local_embeddings)


def available() -> bool:
    """Feature on AND the optional stack importable."""
    if not enabled():
        return False
    try:
        import fastembed  # noqa: F401
        import qdrant_client  # noqa: F401
    except ImportError:
        return False
    return True


def free_mb() -> int:
    """Available (not merely free) RAM in MB; -1 when unknown."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def _enough_ram() -> bool:
    mb = free_mb()
    if mb < 0:
        return True  # unknown platform — don't block
    if mb < _MIN_FREE_MB:
        logger.warning("knowledge.embed: only %d MB RAM available (< %d) — "
                       "skipping the dense channel for now (BM25 still active)",
                       mb, _MIN_FREE_MB)
        return False
    return True


# --- Model / store handles ----------------------------------------------------

def _model_id() -> str:
    return settings.embedding_model or "nomic-ai/nomic-embed-text-v1.5-Q"


def _threads() -> int:
    """How many ONNX threads this machine can afford right now.

    The minimum of what RAM can back and what the CPU actually has. Headroom is
    measured above `_MIN_FREE_MB`, because that floor is reserved — spending it
    on threads is what invites the OOM killer this module exists to avoid.
    """
    cores = os.cpu_count() or 2
    usable_cores = max(1, cores - 1)          # leave one for everything else
    mb = free_mb()
    if mb < 0:                                 # unknown platform — stay modest
        return min(4, usable_cores)
    headroom = max(0, mb - _MIN_FREE_MB)
    by_ram = 1 + int(headroom // _ARENA_MB_PER_THREAD)
    return max(1, min(usable_cores, by_ram, _MAX_THREADS))


def _batch() -> int:
    """Embed batch size, scaled to available RAM.

    Throughput improves with batch size; so does the ONNX arena, which is why
    this is a function of free memory rather than a constant. `_BATCH` is the
    floor a small box gets.
    """
    mb = free_mb()
    if mb < 0:
        return _BATCH
    for floor, size in _BATCH_TIERS:
        if mb >= floor:
            return size
    return _BATCH


def _limit_threads(n: int = 1) -> None:
    """Cap the BLAS/OpenMP pools too — each thread carries its own arena."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)


@contextlib.contextmanager
def _quiet_progress():
    """Swallow fastembed's download progress bar.

    fastembed downloads the model with a bare ``tqdm`` and hardcodes
    ``show_progress=True`` — there is no flag to pass through ``TextEmbedding``,
    and tqdm 4.68 ignores ``TQDM_DISABLE``. tqdm redraws with ``\\r``, which
    collapses to a single line only on a TTY; under the CLI's Rich spinner every
    1 KB chunk becomes a NEW line instead, so a 90 MB download printed thousands
    of "Downloading bytes: ####" rows and buried the interface.

    Swapping the streams is safe for logging: ``logging.StreamHandler`` binds
    ``sys.stderr`` when it is constructed, so handlers made earlier keep writing
    to the real one and only tqdm's direct writes land in the buffer.
    """
    buf = io.StringIO()
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = buf
    try:
        yield buf
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err


def _get_model():
    global _model, _model_name
    name = _model_id()
    if _model is None or _model_name != name:
        n = _threads()
        _limit_threads(n)
        from fastembed import TextEmbedding
        logger.info("knowledge.embed: loading embedding model %s (%d thread(s), "
                    "%d MB free) — first run downloads ~90 MB, once", name, n, free_mb())
        started = time.monotonic()
        with _quiet_progress() as noise:
            _model = TextEmbedding(name, threads=n)
        elapsed = time.monotonic() - started
        # Report the download as ONE line, and only when there actually was one:
        # a warm cache loads in well under a second and deserves no announcement.
        if "Downloading" in noise.getvalue() or elapsed > 5:
            logger.info("knowledge.embed: model %s ready in %.0fs", name, elapsed)
        _model_name = name
    return _model


def unload_model() -> None:
    """Drop the cached model and return its memory to the OS. Called after a
    build and available to callers that want the server slim between runs."""
    global _model, _model_name
    if _model is not None:
        _model = None
        _model_name = ""
        import gc
        gc.collect()


def _close_client() -> None:
    """Close the embedded store before interpreter teardown — qdrant-client's
    __del__ runs after sys.meta_path is torn down and raises a noisy ImportError
    otherwise (harmless, but it pollutes every log and test run)."""
    global _client
    client, _client = _client, None
    if client is not None:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


def _get_client():
    global _client
    if _client is None:
        import atexit

        from qdrant_client import QdrantClient
        path = Path(settings.qdrant_path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        _client = QdrantClient(path=str(path))
        atexit.register(_close_client)
    return _client


def _collection(repo_url: str) -> str:
    return "kn_emb_" + git_ops.slug(repo_url).replace("-", "_").replace(".", "_")


def _point_id(qualified_name: str) -> str:
    return str(uuid.uuid5(_UUID_NS, qualified_name))


# --- Node enumeration + doc-text assembly -------------------------------------

def _nodes(repo_url: str) -> list[dict]:
    """Every embeddable graph node (function/method/class) with its span."""
    rows = graph.cypher(
        repo_url,
        "MATCH (n) WHERE (n:Function OR n:Method OR n:Class) "
        "AND n.file_path IS NOT NULL "
        "RETURN n.name, n.qualified_name, n.file_path, n.start_line, n.end_line, "
        "labels(n), n.signature LIMIT 40000")
    out = []
    for r in rows:
        if not isinstance(r, list) or len(r) < 6 or not r[1] or not r[2]:
            continue
        out.append({
            "name": r[0], "qn": r[1], "file": r[2],
            "start": graph.as_int(r[3]), "end": graph.as_int(r[4]),
            "label": graph.node_label(r[5]), "sig": (r[6] or "") if len(r) > 6 else "",
        })
    return out


def _doc_text(node: dict, body: str) -> str:
    head = f"{node['label']} {node['name']}{node['sig']}"
    parts = [f"search_document: {node['file']}", head]
    if body:
        parts.append(body)
    return "\n".join(parts)[:_DOC_CHARS]


def _read_bodies(repo_url: str, nodes: list[dict]) -> dict[str, str]:
    """First `_BODY_LINES` of each node, one file read per file."""
    path = git_ops.workdir(repo_url)
    by_file: dict[str, list[dict]] = {}
    for n in nodes:
        by_file.setdefault(n["file"], []).append(n)
    bodies: dict[str, str] = {}
    for rel, ns in by_file.items():
        try:
            lines = (path / rel).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for n in ns:
            s = (n["start"] or 1) - 1
            if 0 <= s < len(lines):
                bodies[n["qn"]] = "\n".join(lines[s:s + _BODY_LINES]).strip()[:_DOC_CHARS]
    return bodies


# --- Build (runs in a subprocess) ---------------------------------------------

def build(repo_url: str, on_progress=None, resume: bool = True) -> int:
    """(Re)build the dense index for a repo, in a SUBPROCESS so the model's
    memory is fully reclaimed when it finishes. Returns the vector count
    (0 on failure/unavailable). Never raises.

    `on_progress(done, total)` is called as the child reports batches. Embedding
    is by far the longest stage of a KB build — ~20 minutes for gitea's 15,991
    nodes on a laptop — and it used to sit at a static 88% for all of it, which
    is indistinguishable from a hang. The child's stdout is streamed rather than
    captured at exit so that number can move while the work is happening.
    """
    if not available() or not graph.available():
        return 0
    if not _enough_ram():
        return 0
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1"}
    # Hand the store's lock to the child. Embedded Qdrant takes an EXCLUSIVE file
    # lock, and this process has usually taken it already by answering a search;
    # the child then dies on "Storage folder is already accessed by another
    # instance" and reports zero vectors. Observed live: a graph reindex on
    # gitea left the dense index stamped at the previous SHA because the rebuild
    # could never open the store. The parent reopens lazily on its next query.
    _close_client()

    vectors, deadline = 0, time.monotonic() + _BUILD_TIMEOUT
    argv = [sys.executable, "-m", "app.services.knowledge.embed", repo_url]
    if not resume:
        argv.append("--fresh")
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            errors="replace", env=env,
            cwd=str(Path(__file__).resolve().parents[3]),  # backend/
        )
    except OSError as exc:
        logger.warning("knowledge.embed: build subprocess failed: %s", exc)
        return 0

    try:
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("PROGRESS="):
                if on_progress:
                    done, _, total = line[len("PROGRESS="):].partition("/")
                    with contextlib.suppress(ValueError, TypeError):
                        on_progress(int(done), int(total))
            elif line.startswith("VECTORS="):
                with contextlib.suppress(ValueError):
                    vectors = int(line.split("=", 1)[1] or 0)
            if time.monotonic() > deadline:
                proc.kill()
                logger.warning("knowledge.embed: build exceeded %ss — killed", _BUILD_TIMEOUT)
                return 0
        proc.wait(timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        proc.kill()
        logger.warning("knowledge.embed: build subprocess failed: %s", exc)
        return 0

    if vectors:
        logger.info("knowledge.embed: indexed %d node vectors for %s", vectors, repo_url)
        return vectors
    stderr = (proc.stderr.read() if proc.stderr else "") or ""
    logger.warning("knowledge.embed: build produced no result (%s)", stderr[-200:])
    return 0


def _index_sha(repo_url: str) -> str:
    """The commit this dense index belongs to: the GRAPH's watermark, never the
    working tree's HEAD.

    These vectors are built from the graph's nodes, and the two diverge exactly
    where it matters — the pipeline checks out `agent/scope-N` before calling
    freshness, so a working-tree sha would stamp the agent branch onto an index
    built from origin/HEAD, mismatching on every later run and paying for a full
    rebuild each time. Split out from the build so the choice can be tested
    without the optional Qdrant stack installed.
    """
    with contextlib.suppress(Exception):
        return graph.indexed_sha(repo_url) or ""
    return ""


def _stamp_path(repo_url: str) -> Path:
    return Path(settings.qdrant_path).resolve() / f"{_collection(repo_url)}.sha"


def _read_stamp(repo_url: str) -> str:
    try:
        return _stamp_path(repo_url).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _write_stamp(repo_url: str, sha: str) -> None:
    with contextlib.suppress(OSError):
        _stamp_path(repo_url).write_text(sha, encoding="utf-8")


def _existing_ids(client, coll: str) -> set:
    """Point ids already in the collection, so a resumed build can skip them."""
    return set(_existing_points(client, coll))


def _existing_points(client, coll: str) -> dict:
    """``{point_id: file_path}`` for everything stored — the payload is what lets
    an incremental rebuild find the vectors belonging to changed files."""
    found: dict = {}
    offset = None
    while True:
        points, offset = client.scroll(coll, limit=4096, offset=offset,
                                       with_payload=True, with_vectors=False)
        for p in points:
            found[p.id] = (p.payload or {}).get("file") or ""
        if offset is None:
            break
    return found


def _stale_ids(stored: dict, touched: set) -> set:
    """Stored points whose file changed, was renamed away, or was deleted."""
    return {pid for pid, path in stored.items() if path in touched}


def _incremental(client, coll: str, repo_url: str, stamped: str, sha: str,
                 total_nodes: int) -> tuple[set, bool]:
    """Drop only the vectors for files the diff touched.

    Returns ``(surviving_ids, fresh)``. `fresh` is True when the range could not
    be computed — an old SHA gc'd away, a shallow clone — in which case the
    caller falls back to a full rebuild rather than trusting a partial diff.
    """
    delta = git_ops.changed_files(str(git_ops.workdir(repo_url)), stamped, sha)
    if delta is None:
        logger.info("knowledge.embed: cannot diff %s..%s — full rebuild",
                    stamped[:9], sha[:9])
        return set(), True

    changed, deleted = delta
    touched = set(changed) | set(deleted)
    if not touched:
        return _existing_ids(client, coll), False

    stored = _existing_points(client, coll)
    stale = _stale_ids(stored, touched)
    if stale:
        client.delete(coll, points_selector=list(stale))
    surviving = set(stored) - stale
    logger.info("knowledge.embed: %s..%s touched %d file(s) — re-embedding %d node(s), "
                "keeping %d of %d", stamped[:9], sha[:9], len(touched),
                total_nodes - len(surviving), len(surviving), total_nodes)
    _write_stamp(repo_url, sha)
    return surviving, False


def build_inprocess(repo_url: str, progress=None, resume: bool = True) -> int:
    """The actual build. Runs in the subprocess spawned by `build()`.

    Resumable, because this is an hour of CPU on a laptop and losing all of it
    to a closed lid is not an acceptable failure mode — a power cut at 9,600 of
    gitea's 15,991 nodes cost the whole run once already.

    Resume is only safe while the repo has not moved: `_point_id` is derived
    from the qualified name, so a node whose *body* changed keeps its id and a
    naive skip would leave a stale vector in place forever. The commit sha is
    stamped beside the collection and a mismatch forces a full rebuild.
    """
    from qdrant_client import models as qm

    nodes = _nodes(repo_url)
    if not nodes:
        return 0
    client = _get_client()
    coll = _collection(repo_url)

    sha = _index_sha(repo_url)

    done_ids: set = set()
    fresh = True
    if client.collection_exists(coll):
        stamped = _read_stamp(repo_url)
        if resume and sha and stamped == sha:
            done_ids = _existing_ids(client, coll)
            fresh = False
            logger.info("knowledge.embed: resuming at %d/%d nodes for %s",
                        len(done_ids), len(nodes), repo_url)
        elif resume and sha and stamped:
            # The repo moved. Re-embedding all 15,991 nodes because one merge
            # touched four files is an hour of CPU to redo work that is still
            # correct, so drop only the vectors belonging to files the diff
            # touched and keep the rest. Deleted and renamed-away paths are in
            # `deleted`, so their orphaned vectors go too.
            done_ids, fresh = _incremental(client, coll, repo_url, stamped, sha, len(nodes))
        if fresh:
            # No usable diff, or told to start over. Stale vectors keyed by an
            # unchanged qualified name would otherwise survive forever.
            client.delete_collection(coll)
    if fresh:
        client.create_collection(
            coll,
            vectors_config=qm.VectorParams(size=_DIM, distance=qm.Distance.COSINE,
                                           on_disk=True),
            on_disk_payload=True)
        _write_stamp(repo_url, sha)

    todo = [n for n in nodes if _point_id(n["qn"]) not in done_ids]
    if not todo:
        return len(done_ids)

    bodies = _read_bodies(repo_url, todo)
    model = _get_model()
    total = len(done_ids)
    pending: list = []

    def flush() -> None:
        if pending:
            client.upsert(coll, points=pending)
            pending.clear()

    # Sized once, from the memory available when the build starts — re-reading
    # it per chunk would let the batch shrink and grow with unrelated processes.
    size = _batch()
    logger.info("knowledge.embed: %d node(s) to embed — batch %d, %d thread(s), "
                "%d MB free", len(todo), size, _threads(), free_mb())
    for i in range(0, len(todo), size):
        batch = todo[i:i + size]
        texts = [_doc_text(n, bodies.get(n["qn"], "")) for n in batch]
        vecs = [v.tolist() for v in model.embed(texts, batch_size=size)]
        pending.extend(
            qm.PointStruct(
                id=_point_id(n["qn"]), vector=v,
                payload={"name": n["name"], "qn": n["qn"], "file": n["file"],
                         "line": n["start"], "label": n["label"]})
            for n, v in zip(batch, vecs, strict=False))
        total += len(batch)
        if len(pending) >= _UPSERT_EVERY:
            flush()
            if progress:
                progress(total, len(nodes))
    flush()
    if progress:
        progress(total, len(nodes))
    return total


# --- Search (in-process, guarded) ---------------------------------------------

def search(repo_url: str, query: str, *, limit: int = 10) -> list[dict]:
    """Dense semantic search over the node vectors. Returns node hits
    [{name, file_path, start_line, label, qualified_name, score}] or []."""
    if not available() or not query.strip():
        return []
    try:
        client = _get_client()
        coll = _collection(repo_url)
        if not client.collection_exists(coll):
            return []
        if not _enough_ram() and _model is None:
            return []  # don't load the model into a machine that's already tight
        vector = next(iter(_get_model().embed([f"search_query: {query}"]))).tolist()
        # query_points(), not the removed .search() — qdrant-client >=1.10.
        hits = client.query_points(collection_name=coll, query=vector,
                                   limit=limit, with_payload=True).points
    except Exception as exc:  # noqa: BLE001
        # WARNING, not debug: a silently-empty dense channel is indistinguishable
        # from "no matches" and cost us a whole A/B round once.
        logger.warning("knowledge.embed: search failed for %s: %s", repo_url, exc)
        return []
    out = []
    for h in hits:
        p = h.payload or {}
        out.append({"name": p.get("name"), "file_path": p.get("file"),
                    "start_line": p.get("line"), "label": p.get("label"),
                    "qualified_name": p.get("qn"), "score": float(h.score)})
    return out


def delete(repo_url: str) -> None:
    if not available():
        return
    try:
        _get_client().delete_collection(_collection(repo_url))
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":  # subprocess entrypoint (see build())
    _limit_threads()
    args = sys.argv[1:]
    url = next((a for a in args if not a.startswith("--")), "")
    resume = "--fresh" not in args

    def _emit(done: int, total: int) -> None:
        # The parent's only window into a 20-minute stage. Unbuffered, or it
        # arrives in one burst at exit and reports nothing while it matters.
        print(f"PROGRESS={done}/{total}", flush=True)

    try:
        count = build_inprocess(url, progress=_emit, resume=resume)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR={exc}", file=sys.stderr)
        count = 0
    print(f"VECTORS={count}", flush=True)
