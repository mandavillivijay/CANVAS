"""Tests for storage backends and the open_backend / open_store factories."""
from __future__ import annotations

import json

import numpy as np
import pytest

from canvas_heal.backends import SQLiteBackend, open_backend
from canvas_heal.backends.base import IntentStoreBackend
from canvas_heal.descriptor import SemanticDescriptor
from canvas_heal.resolver import IntentStore, open_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_descriptor(**kw) -> SemanticDescriptor:
    defaults = dict(
        tag="button", role="button", label="Submit", element_type="submit",
        placeholder="", text_content="Submit", parent_tag="form",
        parent_role="", section_heading="", landmark="",
    )
    defaults.update(kw)
    return SemanticDescriptor(**defaults)


def _dummy_blob(dim: int = 8) -> bytes:
    return np.ones(dim, dtype=np.float32).tobytes()


def _dummy_embedding(dim: int = 8) -> np.ndarray:
    return np.ones(dim, dtype=np.float32)


# ---------------------------------------------------------------------------
# SQLiteBackend unit tests
# ---------------------------------------------------------------------------

class TestSQLiteBackend:
    def test_is_backend_subclass(self):
        assert issubclass(SQLiteBackend, IntentStoreBackend)

    def test_store_and_get(self):
        b = SQLiteBackend()
        b.store("btn", "#btn", '{"tag":"button"}', _dummy_blob(), "model-A", "http://x", "tester")
        row = b.get("btn")
        assert row is not None
        selector, desc_json, blob, page_url = row
        assert selector == "#btn"
        assert json.loads(desc_json)["tag"] == "button"
        assert page_url == "http://x"
        b.close()

    def test_get_missing_returns_none(self):
        b = SQLiteBackend()
        assert b.get("nonexistent") is None
        b.close()

    def test_all_returns_sorted(self):
        b = SQLiteBackend()
        b.store("z-intent", "#z", "{}", _dummy_blob(), "", "", "")
        b.store("a-intent", "#a", "{}", _dummy_blob(), "", "", "")
        names = [r[0] for r in b.all()]
        assert names == sorted(names)
        b.close()

    def test_count(self):
        b = SQLiteBackend()
        assert b.count() == 0
        b.store("x", "#x", "{}", _dummy_blob(), "", "", "")
        assert b.count() == 1
        b.store("y", "#y", "{}", _dummy_blob(), "", "", "")
        assert b.count() == 2
        b.close()

    def test_store_overwrites_existing(self):
        b = SQLiteBackend()
        b.store("btn", "#old", '{}', _dummy_blob(), "", "", "")
        b.store("btn", "#new", '{}', _dummy_blob(), "", "", "")
        assert b.get("btn")[0] == "#new"
        assert b.count() == 1
        b.close()

    def test_version_history_newest_first(self):
        b = SQLiteBackend()
        b.store("btn", "#v1", '{}', _dummy_blob(), "", "", "")
        b.store("btn", "#v2", '{}', _dummy_blob(), "", "", "")
        history = b.get_version_history("btn")
        assert len(history) == 2
        assert history[0]["selector"] == "#v2"  # newest first
        b.close()

    def test_version_history_empty_for_unknown(self):
        b = SQLiteBackend()
        assert b.get_version_history("ghost") == []
        b.close()

    def test_rollback(self):
        b = SQLiteBackend()
        b.store("btn", "#v1", '{}', _dummy_blob(), "", "", "")
        history = b.get_version_history("btn")
        v1_id = history[0]["id"]
        b.store("btn", "#v2", '{}', _dummy_blob(), "", "", "")
        assert b.get("btn")[0] == "#v2"
        ok = b.rollback("btn", v1_id, "ci-user")
        assert ok is True
        assert b.get("btn")[0] == "#v1"
        b.close()

    def test_rollback_unknown_version(self):
        b = SQLiteBackend()
        assert b.rollback("ghost", 9999, "") is False
        b.close()

    def test_context_manager(self):
        with SQLiteBackend() as b:
            b.store("k", "#k", '{}', _dummy_blob(), "", "", "")
            assert b.count() == 1


# ---------------------------------------------------------------------------
# Team / project namespace isolation
# ---------------------------------------------------------------------------

class TestNamespaceIsolation:
    def test_team_isolation(self, tmp_path):
        db = str(tmp_path / "shared.db")
        b_alpha = SQLiteBackend(db, team_id="alpha")
        b_beta = SQLiteBackend(db, team_id="beta")

        b_alpha.store("btn", "#alpha-btn", '{}', _dummy_blob(), "", "", "")
        b_beta.store("btn", "#beta-btn", '{}', _dummy_blob(), "", "", "")

        assert b_alpha.get("btn")[0] == "#alpha-btn"
        assert b_beta.get("btn")[0] == "#beta-btn"
        assert b_alpha.count() == 1
        assert b_beta.count() == 1

        b_alpha.close()
        b_beta.close()

    def test_project_isolation(self, tmp_path):
        db = str(tmp_path / "shared.db")
        b1 = SQLiteBackend(db, project_id="proj-1")
        b2 = SQLiteBackend(db, project_id="proj-2")

        b1.store("btn", "#p1", '{}', _dummy_blob(), "", "", "")
        assert b2.get("btn") is None
        assert b2.count() == 0

        b1.close()
        b2.close()

    def test_all_scoped_to_namespace(self, tmp_path):
        db = str(tmp_path / "shared.db")
        b_a = SQLiteBackend(db, team_id="a")
        b_b = SQLiteBackend(db, team_id="b")

        b_a.store("x", "#x", '{}', _dummy_blob(), "", "", "")
        b_b.store("y", "#y", '{}', _dummy_blob(), "", "", "")
        b_b.store("z", "#z", '{}', _dummy_blob(), "", "", "")

        assert [r[0] for r in b_a.all()] == ["x"]
        assert [r[0] for r in b_b.all()] == ["y", "z"]

        b_a.close()
        b_b.close()


# ---------------------------------------------------------------------------
# open_backend factory
# ---------------------------------------------------------------------------

class TestOpenBackend:
    def test_sqlite_memory(self):
        b = open_backend("sqlite:///:memory:")
        assert isinstance(b, SQLiteBackend)
        b.close()

    def test_sqlite_file(self, tmp_path):
        url = f"sqlite:///{tmp_path / 'test.db'}"
        b = open_backend(url)
        b.store("x", "#x", '{}', _dummy_blob(), "", "", "")
        assert b.count() == 1
        b.close()

    def test_sqlite_with_team_id(self):
        b = open_backend("sqlite:///:memory:", team_id="ci")
        assert b.team_id == "ci"
        b.close()

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="Unsupported store URL"):
            open_backend("redis://localhost")

    def test_postgres_missing_dep_raises(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "psycopg_pool" in name:
                raise ImportError("psycopg[pool] not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="psycopg"):
            open_backend("postgresql://localhost/test")


# ---------------------------------------------------------------------------
# open_store factory (IntentStore-level)
# ---------------------------------------------------------------------------

class TestOpenStore:
    def test_returns_intent_store(self):
        store = open_store("sqlite:///:memory:")
        assert isinstance(store, IntentStore)
        store.close()

    def test_round_trip_through_store(self):
        from canvas_heal.embedder import IntentEmbedder
        store = open_store("sqlite:///:memory:")
        embedder = IntentEmbedder.get()
        desc = _make_descriptor()
        emb = embedder.embed_descriptor(desc)
        store.store("submit-btn", "#submit", desc, emb, model_name="all-MiniLM-L6-v2", page_url="http://localhost/form")
        result = store.get("submit-btn")
        assert result is not None
        sel, desc_out, emb_out, url = result
        assert sel == "#submit"
        assert url == "http://localhost/form"
        np.testing.assert_array_almost_equal(emb, emb_out, decimal=5)
        store.close()

    def test_team_id_forwarded(self):
        store = open_store("sqlite:///:memory:", team_id="myteam", project_id="myproject")
        assert store._backend.team_id == "myteam"
        assert store._backend.project_id == "myproject"
        store.close()


# ---------------------------------------------------------------------------
# IntentStore backward-compatibility (db_path= still works)
# ---------------------------------------------------------------------------

class TestIntentStoreCompat:
    def test_db_path_memory_still_works(self):
        store = IntentStore(":memory:")
        assert store.count() == 0
        store.close()

    def test_db_path_none_uses_default(self, monkeypatch, tmp_path):
        import canvas_heal.resolver as resolver_mod
        monkeypatch.setattr(resolver_mod, "_DEFAULT_DB", tmp_path / "default.db")
        store = IntentStore()
        store.close()

    def test_pii_scrubbing_still_works(self):
        from canvas_heal.embedder import IntentEmbedder
        store = IntentStore(":memory:", store_raw_text=False)
        embedder = IntentEmbedder.get()
        desc = _make_descriptor(label="user@example.com", text_content="Contact")
        emb = embedder.embed_descriptor(desc)
        store.store("contact", "#c", desc, emb)
        row = store._backend.get("contact")
        stored_desc = json.loads(row[1])
        assert "user@example.com" not in stored_desc.get("label", "")
        store.close()
