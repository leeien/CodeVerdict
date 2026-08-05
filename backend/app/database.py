from collections.abc import Iterator

from sqlmodel import Session, SQLModel, create_engine

from .config import settings

_connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)

engine = create_engine(settings.database_url, echo=False, connect_args=_connect_args)


def _migrate() -> None:
    """Lightweight additive migrations for SQLite — create_all() does not add
    new columns to tables that already exist."""
    if not settings.database_url.startswith("sqlite"):
        return
    adds = {
        "scopesession": [
            ("pm_tokens_input", "INTEGER DEFAULT 0"),
            ("pm_tokens_output", "INTEGER DEFAULT 0"),
            ("pm_cost_usd", "FLOAT DEFAULT 0"),
            ("plan", "JSON"),
        ],
        "task": [
            ("jira_key", "VARCHAR"),
            ("jira_url", "VARCHAR"),
            ("affected_files", "JSON"),
            ("target_symbols", "JSON"),
            ("review_findings", "JSON"),
        ],
        "repo": [
            ("kb_knowledge_count", "INTEGER DEFAULT 0"),
            ("kb_views", "JSON"),
            ("kb_tokens_in", "INTEGER DEFAULT 0"),
            ("kb_tokens_out", "INTEGER DEFAULT 0"),
            ("kb_cost_usd", "FLOAT DEFAULT 0"),
        ],
        "agentrun": [
            ("usage_unknown", "BOOLEAN DEFAULT 0"),
        ],
        "user": [
            ("github_login", "VARCHAR"),
            ("github_name", "VARCHAR"),
            ("github_token", "VARCHAR"),
        ],
    }
    with engine.begin() as conn:
        for table, cols in adds.items():
            try:
                existing = {row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")}
            except Exception:
                continue
            if not existing:
                continue  # table not created yet
            for name, decl in cols:
                if name not in existing:
                    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def init_db() -> None:
    # Import models so they register on SQLModel.metadata before create_all.
    from . import models  # noqa: F401

    SQLModel.metadata.create_all(engine)
    _migrate()


def get_session() -> Iterator[Session]:
    # expire_on_commit=False: keep attributes loaded after commit so freshly
    # created/updated rows still serialize correctly in the response.
    with Session(engine, expire_on_commit=False) as session:
        yield session
