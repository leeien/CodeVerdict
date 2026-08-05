"""Shared test fixtures.

Everything here keeps the suite hermetic: a throwaway SQLite database per test
session, a deterministic encryption key, and no network. The LLM/agent boundary
is never called for real — tests that exercise the pipeline monkeypatch it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Isolate global state BEFORE app.config imports and builds its settings
# singleton: a temp DB, a fixed Fernet passphrase, and safe defaults. These have
# to be set at import time because config.Settings() reads the environment once.
_TMP = tempfile.mkdtemp(prefix="codeverdict-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TMP, 'test.db').as_posix()}"
os.environ["CODEVERDICT_SECRET_KEY"] = "test-suite-fixed-key"
os.environ["REPOS_DIR"] = str(Path(_TMP, "workspace"))
os.environ["GENERATE_KNOWLEDGE"] = "false"
# Installing the shim starts the loopback tools endpoint for real. The suite
# exercises install() directly, and a uvicorn thread booting mid-test would run
# the app's lifespan against the test database — which is how seeding the jury
# roster started leaking between tests.
os.environ["CODEVERDICT_NO_TOOLS_SERVER"] = "1"
os.environ.setdefault("RAG_EMBEDDINGS", "tfidf")
os.environ.pop("ADMIN_PASSWORD", None)
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)


@pytest.fixture(scope="session", autouse=True)
def _init_schema():
    from app.database import init_db

    init_db()
    yield


@pytest.fixture
def db():
    """A fresh session; each test wraps its own writes and the table set is
    reset between tests so ordering never matters."""
    from app.database import engine
    from sqlmodel import Session, SQLModel

    with Session(engine) as session:
        yield session
    # Wipe every table so tests are independent regardless of run order.
    with engine.begin() as conn:
        for table in reversed(SQLModel.metadata.sorted_tables):
            conn.exec_driver_sql(f"DELETE FROM {table.name}")


@pytest.fixture
def client(db):
    """A TestClient over the tools endpoint.

    There is no UI and no login: the only thing served is the agents' index
    endpoint, gated by a per-run token rather than a session."""
    from app.main import app
    from fastapi.testclient import TestClient

    return TestClient(app)
