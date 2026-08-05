"""Terminal-client session state: what's booted, who's acting, what's selected.

The server keeps this in a cookie and a lifespan hook. A CLI has neither, so it
lives here: boot the same way the server's lifespan does, resolve an acting
user, and remember which repository and which scope the person is working on
between invocations — otherwise every command would need a --repo flag and the
tool would be exhausting to use.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from ..config import settings
from ..database import engine, init_db
from ..models import Repo, ScopeSession, User, UserRole


def _state_path() -> Path:
    """Selection state sits beside the database it refers to.

    Keying it to the DB matters: repo and session ids are only meaningful
    within one database, and a machine can hold several CodeVerdict checkouts.
    A single global state file would silently point one project's CLI at
    another's ticket ids.
    """
    url = settings.database_url
    if url.startswith("sqlite:///"):
        return Path(url[len("sqlite:///"):]).with_suffix(".cli.json")
    return Path.home() / ".codeverdict-cli.json"


class Context:
    """One terminal session's view of the system."""

    def __init__(self) -> None:
        self.repo_id: int | None = None
        self.session_id: int | None = None
        self.user_id: int | None = None
        self._loaded = False

    # ── lifecycle ────────────────────────────────────────────────────────────
    def boot(self) -> None:
        """Bring the process up the way the server's lifespan does.

        Identical order, and for the same reasons: schema first, then the
        Settings-screen overrides (so a stage's configured provider wins over
        whatever is in the environment), then the jury seed — which reads those
        overrides to spread the default panel across providers that are
        actually configured.
        """
        from ..core import repos as core_repos
        from ..services import judges, runtime_settings
        from ..services.auth import ensure_bootstrap_admin

        init_db()
        with Session(engine) as db:
            ensure_bootstrap_admin(db)
            runtime_settings.apply_overrides(db)
            judges.ensure_seeded(db)
            core_repos.reconcile_interrupted(db)
        self._load()

    @contextlib.contextmanager
    def db(self) -> Iterator[Session]:
        # expire_on_commit=False mirrors the API's session factory, so rows
        # stay readable after a commit instead of triggering surprise reloads.
        with Session(engine, expire_on_commit=False) as session:
            yield session

    # ── acting user ──────────────────────────────────────────────────────────
    def user(self, db: Session) -> User:
        """Who the terminal acts as.

        A local CLI has no login step — the person at the keyboard already
        proved they own the machine and the database file. But operations like
        opening a PR need a *User row*, because that is where a connected
        GitHub account's token lives. So the terminal binds to the local admin
        account rather than inventing a second identity model.
        """
        if self.user_id is not None:
            user = db.get(User, self.user_id)
            if user is not None:
                return user
        user = db.exec(
            select(User).where(User.role == UserRole.admin.value).order_by(User.id)
        ).first()
        if user is None:  # ensure_bootstrap_admin guarantees one; belt and braces
            user = db.exec(select(User).order_by(User.id)).first()
        if user is None:
            raise RuntimeError("No account exists in this database — boot() should have made one.")
        self.user_id = user.id
        return user

    # ── selection ────────────────────────────────────────────────────────────
    def repo(self, db: Session) -> Repo | None:
        if self.repo_id is None:
            return None
        return db.get(Repo, self.repo_id)

    def scope(self, db: Session) -> ScopeSession | None:
        if self.session_id is None:
            return None
        return db.get(ScopeSession, self.session_id)

    def select_repo(self, repo: Repo) -> None:
        """Switching repository always clears the scope — a scope session belongs
        to exactly one repo, and carrying a stale one across would have the PM
        answering about the wrong codebase."""
        if self.repo_id != repo.id:
            self.session_id = None
        self.repo_id = repo.id
        self.save()

    def select_scope(self, session: ScopeSession | None) -> None:
        self.session_id = session.id if session else None
        if session is not None:
            self.repo_id = session.repo_id
        self.save()

    def ensure_repo(self, db: Session) -> Repo:
        """The selected repo, or a clear explanation of what to do instead."""
        from ..core.errors import conflict

        repo = self.repo(db)
        if repo is not None:
            return repo
        known = db.exec(select(Repo).order_by(Repo.created_at.desc())).all()
        if not known:
            raise conflict("No repository has been indexed yet — run /kb add <git-url> first.")
        if len(known) == 1:
            self.select_repo(known[0])
            return known[0]
        names = ", ".join(f"{r.org}/{r.name}" for r in known[:6])
        raise conflict(f"No repository selected — pick one with /repo <name>. Known: {names}")

    # ── persistence ──────────────────────────────────────────────────────────
    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        path = _state_path()
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        # Validate against the DB rather than trusting the file: rows get
        # deleted between sessions, and a dangling id would fail later with a
        # confusing "not found" from deep inside a command.
        with self.db() as db:
            repo_id = data.get("repo_id")
            if isinstance(repo_id, int) and db.get(Repo, repo_id) is not None:
                self.repo_id = repo_id
            session_id = data.get("session_id")
            if isinstance(session_id, int):
                scope = db.get(ScopeSession, session_id)
                if scope is not None and scope.repo_id == self.repo_id:
                    self.session_id = session_id

    def save(self) -> None:
        path = _state_path()
        try:
            path.write_text(
                json.dumps({"repo_id": self.repo_id, "session_id": self.session_id}, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logging.getLogger(__name__).debug("could not persist CLI state to %s", path)
