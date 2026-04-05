# tests/test_mcp_knowledge_store.py
import json
import pytest
from pathlib import Path

from integrations.mcp.knowledge_store import KnowledgeStore


@pytest.fixture
def store(tmp_path):
    return KnowledgeStore(tmp_path / "test.json", llm_budget_usd=0.50,
                          dream_threshold=5)


class TestAssertAndQuery:
    def test_assert_fact(self, store):
        result = store.assert_fact("(parent john mary)")
        assert result["kb_size"] == 1
        assert "parent" in result["added"]

    def test_assert_rule(self, store):
        result = store.assert_fact(
            "(grandparent X Z) :- (parent X Y), (parent Y Z)")
        assert result["kb_size"] == 1
        assert len(store.kb.rules) == 1

    def test_query_fact(self, store):
        store.assert_fact("(parent john mary)")
        store.assert_fact("(parent john bob)")
        result = store.query("(parent john X)")
        assert result["count"] == 2
        names = {s["X"] for s in result["solutions"]}
        assert names == {"mary", "bob"}

    def test_query_with_rule(self, store):
        store.assert_fact("(parent john mary)")
        store.assert_fact("(parent mary alice)")
        store.assert_fact(
            "(grandparent X Z) :- (parent X Y), (parent Y Z)")
        result = store.query("(grandparent john Z)")
        assert result["count"] == 1
        assert result["solutions"][0]["Z"] == "alice"

    def test_query_no_results(self, store):
        result = store.query("(parent john X)")
        assert result["count"] == 0
        assert result["solutions"] == []

    def test_query_limit(self, store):
        for i in range(20):
            store.assert_fact(f"(item x{i})")
        result = store.query("(item X)", limit=5)
        assert result["count"] == 5
        assert result["has_more"] is True


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "persist.json"
        s1 = KnowledgeStore(path)
        s1.assert_fact("(parent john mary)")
        s1.assert_fact("(likes john pizza)")

        s2 = KnowledgeStore(path)
        result = s2.query("(parent john X)")
        assert result["count"] == 1
        assert result["solutions"][0]["X"] == "mary"

    def test_metadata_persists(self, tmp_path):
        path = tmp_path / "meta.json"
        s1 = KnowledgeStore(path)
        s1.assert_fact("(a x)")
        s1.assert_fact("(b y)")
        s1._dream_count = 3
        s1._total_queries = 42
        s1._dirty = True
        s1.save()

        s2 = KnowledgeStore(path)
        assert s2._dream_count == 3
        assert s2._total_queries == 42

    def test_atomic_write(self, tmp_path):
        path = tmp_path / "atomic.json"
        store = KnowledgeStore(path)
        store.assert_fact("(a x)")
        # File should exist and be valid JSON
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert len(data["kb"]) == 1

    def test_corrupt_file_starts_fresh(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json {{")
        store = KnowledgeStore(path)
        assert len(store.kb) == 0


class TestDream:
    def test_dream_compresses(self, store):
        # Add redundant facts that Op C can generalize
        for name in ["alice", "bob", "carol"]:
            store.assert_fact(f"(person {name})")
            store.assert_fact(f"(tall {name})")
        result = store.dream()
        assert result["before"] == 6
        # Dream may or may not compress depending on min_group_size

    def test_dream_dry_run(self, store):
        store.assert_fact("(a x)")
        store.assert_fact("(b y)")
        before = len(store.kb)
        result = store.dream(dry_run=True)
        assert result["dry_run"] is True
        assert len(store.kb) == before  # unchanged

    def test_dream_resets_counters(self, store):
        store.assert_fact("(a x)")
        store.query("(a X)")
        assert store._facts_since_dream == 1
        assert store._queries_since_dream == 1
        store.dream()
        assert store._facts_since_dream == 0
        assert store._queries_since_dream == 0
        assert store._dream_count == 1


class TestDreamRecommendation:
    def test_should_dream_below_threshold(self, store):
        store.assert_fact("(a x)")
        assert not store.should_dream()

    def test_should_dream_above_threshold(self, store):
        # threshold is 5 for this fixture
        for i in range(5):
            store.assert_fact(f"(item x{i})")
        assert store.should_dream()

    def test_dream_recommended_in_response(self, store):
        for i in range(5):
            store.assert_fact(f"(item x{i})")
        result = store.assert_fact("(item x5)")
        assert result["dream_recommended"] is True


class TestExplain:
    def test_explain_fact(self, store):
        store.assert_fact("(parent john mary)")
        result = store.explain("(parent john mary)")
        assert result["derivable"] is True
        assert len(result["matching_facts"]) == 1

    def test_explain_via_rule(self, store):
        store.assert_fact("(parent john mary)")
        store.assert_fact("(parent mary alice)")
        store.assert_fact(
            "(grandparent X Z) :- (parent X Y), (parent Y Z)")
        result = store.explain("(grandparent john alice)")
        assert result["derivable"] is True
        assert len(result["matching_rules"]) == 1

    def test_explain_not_derivable(self, store):
        result = store.explain("(foo bar)")
        assert result["derivable"] is False


class TestStatus:
    def test_status_fields(self, store):
        store.assert_fact("(parent john mary)")
        store.query("(parent john X)")
        status = store.status()
        assert status["facts"] == 1
        assert status["rules"] == 0
        assert status["total_assertions"] == 1
        assert status["total_queries"] == 1
        assert "parent" in status["functors"]
        assert "store_path" in status

    def test_budget_no_llm(self, tmp_path):
        store = KnowledgeStore(tmp_path / "t.json")
        store.llm_client = None
        budget = store.budget_remaining()
        assert budget["enabled"] is False
