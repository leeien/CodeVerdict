"""PM retrieval-query formulation (services/pm_agent.py).

The measured localization failures were not retrieval-quality failures — the
index found what it was asked for. They were query failures: the PM restated the
user's symptom, so the search answered in symptom vocabulary. The PM now issues
a mechanism query plus alternate phrasings, and every phrasing is searched.
"""

from __future__ import annotations

from app.services import pm_agent


class TestQueryFanOut:
    def test_alt_queries_ride_along_with_the_primary(self):
        assert pm_agent._queries({
            "retrieval_query": "measure printable cell width ignoring ansi escapes",
            "alt_queries": ["strip ansi sequences", "text width calculation"],
        }) == ["measure printable cell width ignoring ansi escapes",
               "strip ansi sequences", "text width calculation"]

    def test_a_bare_string_alt_query_is_accepted(self):
        assert pm_agent._queries({"retrieval_query": "a", "alt_queries": "b"}) == ["a", "b"]

    def test_duplicates_and_blanks_are_dropped(self):
        assert pm_agent._queries({"retrieval_query": "same query",
                                  "alt_queries": ["SAME QUERY", "  ", "other"]}) \
            == ["same query", "other"]

    def test_at_most_two_alternates_are_searched(self):
        assert len(pm_agent._queries({"retrieval_query": "q",
                                      "alt_queries": ["a", "b", "c", "d"]})) == 3

    def test_missing_alt_queries_still_yields_the_primary(self):
        assert pm_agent._queries({"retrieval_query": "q"}) == ["q"]

    def test_every_phrasing_is_actually_retrieved(self, monkeypatch):
        asked: list[str] = []
        monkeypatch.setattr(pm_agent.knowledge_retriever, "localize",
                            lambda repo, q, **k: asked.append(q) or f"hits for {q}")
        monkeypatch.setattr(pm_agent.knowledge_retriever, "code_hits", lambda repo, q, **k: "")
        monkeypatch.setattr(pm_agent.knowledge_retriever, "notes", lambda repo, q, **k: [])
        blocks = pm_agent._retrieve_more("repo", ["mechanism query", "synonym query"], set())
        assert asked == ["mechanism query", "synonym query"]
        assert len(blocks) == 2

    def test_identical_blocks_across_phrasings_are_not_duplicated(self, monkeypatch):
        monkeypatch.setattr(pm_agent.knowledge_retriever, "localize",
                            lambda repo, q, **k: "the same hits")
        monkeypatch.setattr(pm_agent.knowledge_retriever, "code_hits", lambda repo, q, **k: "")
        monkeypatch.setattr(pm_agent.knowledge_retriever, "notes", lambda repo, q, **k: [])
        assert pm_agent._retrieve_more("repo", ["one", "two"], set()) == ["the same hits"]


class TestPrompt:
    def test_the_pm_is_told_to_query_the_mechanism_not_the_symptom(self):
        assert "MECHANISM" in pm_agent._SYSTEM
        assert "alt_queries" in pm_agent._SYSTEM
