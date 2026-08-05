"""Knowledge-base helpers: repo slugging, the delivery-note JSON store, and the
in-process TF-IDF ranking over notes (retriever.notes) that replaced the vector
index.
"""

import pytest
from app.config import settings
from app.services import git_ops
from app.services.knowledge import retriever, store
from app.services.knowledge.facts import KnowledgeDocument

REPO = "https://github.com/example/knowledge-test"


class TestSlug:
    def test_https_url(self):
        assert git_ops.slug("https://github.com/pallets/click") == "pallets__click"

    def test_dot_git_and_trailing_slash_stripped(self):
        assert git_ops.slug("https://github.com/pallets/click.git/") == "pallets__click"

    def test_ssh_url(self):
        assert git_ops.slug("git@github.com:pallets/click.git") == "pallets__click"


class TestKnowledgeDocument:
    def test_roundtrips_through_dict(self):
        doc = KnowledgeDocument(id="d1", type="delivery_note", name="Delivered: x",
                                summary="did a thing", tags=["a"], related=["d2"],
                                content={"files": ["x.py"], "gotchas": ["watch out"]})
        assert KnowledgeDocument.from_dict(doc.to_dict()) == doc


@pytest.fixture()
def _isolated_kb(tmp_path):
    saved = settings.knowledge_dir
    settings.knowledge_dir = str(tmp_path)
    yield
    settings.knowledge_dir = saved


class TestNoteStore:
    def test_save_load_remove(self, _isolated_kb):
        doc = KnowledgeDocument(id="delivery_1", type="delivery_note", name="d",
                                summary="s")
        store.save(REPO, doc)
        assert store.has_knowledge(REPO) is True
        assert store.load(REPO, "delivery_1") == doc
        assert [d.id for d in store.load_all(REPO)] == ["delivery_1"]
        store.remove(REPO, "delivery_1")
        assert store.load(REPO, "delivery_1") is None
        assert store.has_knowledge(REPO) is False


class TestNotesRanking:
    def _note(self, doc_id, summary, **content):
        return KnowledgeDocument(id=doc_id, type="delivery_note", name=doc_id,
                                 summary=summary, content=content)

    def test_ranks_relevant_note_first(self, _isolated_kb):
        store.save(REPO, self._note("d_auth", "Added password login and session cookies",
                                    files=["auth.py"], gotchas=["sessions expire in 1h"]))
        store.save(REPO, self._note("d_csv", "Implemented CSV export of reports",
                                    files=["export.py"]))
        hits = retriever.notes(REPO, "how does login work")
        assert hits and hits[0][0].id == "d_auth"

    def test_no_match_returns_empty(self, _isolated_kb):
        store.save(REPO, self._note("d_csv", "Implemented CSV export", files=["export.py"]))
        assert retriever.notes(REPO, "zzzq qqzz xyzzy") == []

    def test_notes_context_renders_confidence_caveat(self, _isolated_kb):
        d = KnowledgeDocument(id="d_unmerged", type="delivery_note",
                              name="[UNMERGED] Delivered: no-color flag",
                              summary="Added a --no-color flag", content={"files": ["cli.py"]})
        d.confidence = "LOW"
        store.save(REPO, d)
        ctx = retriever.notes_context(REPO, "no-color flag")
        assert "[UNMERGED] Delivered: no-color flag" in ctx  # renders the NAME
        assert "LOW" in ctx and "verify" in ctx.lower()
