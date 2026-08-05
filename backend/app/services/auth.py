"""Authentication + role-based access control.

Design goals: zero new dependencies (stdlib PBKDF2, opaque tokens in an
httponly cookie) and a safe bootstrap — on first boot an ``admin`` user is
created with a *randomly generated* password printed to the server log (or
taken from ``ADMIN_PASSWORD`` if you set it), so a deployment is never
reachable with a guessable default.

Roles (models.UserRole): viewer < member < admin. Guards are FastAPI
dependencies; wire read access at include_router() and write access per-route.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from datetime import timedelta

from fastapi import Depends, HTTPException, Request
from sqlmodel import Session, select

from ..config import settings
from ..database import get_session
from ..models import ROLE_RANK, AppSetting, AuthSession, User, UserRole, utcnow

logger = logging.getLogger(__name__)

COOKIE_NAME = "codeverdict_session"
SESSION_TTL = timedelta(days=14)
_PBKDF2_ITERATIONS = 200_000

BOOTSTRAP_USERNAME = "admin"
# Set while the bootstrap admin still has its generated password; the UI reads
# it via /auth/me and nags until the operator picks their own.
_BOOTSTRAP_PENDING_KEY = "bootstrap_password_pending"


# --- Passwords ---------------------------------------------------------------
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    ).hex()
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt, digest = stored.split("$")
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt), int(iterations)
        ).hex()
        return hmac.compare_digest(candidate, digest)
    except (ValueError, TypeError):
        return False


# --- Sessions ----------------------------------------------------------------
def create_session(db: Session, user: User) -> str:
    token = secrets.token_urlsafe(32)
    db.add(AuthSession(token=token, user_id=user.id, expires_at=utcnow() + SESSION_TTL))
    db.commit()
    return token


def delete_session(db: Session, token: str) -> None:
    row = db.exec(select(AuthSession).where(AuthSession.token == token)).first()
    if row:
        db.delete(row)
        db.commit()


def user_for_token(db: Session, token: str | None) -> User | None:
    if not token:
        return None
    row = db.exec(select(AuthSession).where(AuthSession.token == token)).first()
    if row is None:
        return None
    expires = row.expires_at
    now = utcnow()
    # SQLite returns naive datetimes; compare in naive UTC.
    if expires.tzinfo is None:
        now = now.replace(tzinfo=None)
    if expires < now:
        db.delete(row)
        db.commit()
        return None
    return db.get(User, row.user_id)


# --- FastAPI dependencies ----------------------------------------------------
def current_user(request: Request, db: Session = Depends(get_session)) -> User | None:
    return user_for_token(db, request.cookies.get(COOKIE_NAME))


def require_user(user: User | None = Depends(current_user)) -> User:
    if user is None:
        raise HTTPException(401, "Not authenticated")
    return user


def _require_rank(min_role: str):
    def guard(user: User = Depends(require_user)) -> User:
        if ROLE_RANK.get(user.role, -1) < ROLE_RANK[min_role]:
            raise HTTPException(403, f"Requires the {min_role} role")
        return user

    return guard


require_member = _require_rank(UserRole.member.value)
require_admin = _require_rank(UserRole.admin.value)


# --- Bootstrap ---------------------------------------------------------------
def _set_bootstrap_pending(db: Session, pending: bool) -> None:
    row = db.get(AppSetting, _BOOTSTRAP_PENDING_KEY)
    if pending:
        if row is None:
            db.add(AppSetting(key=_BOOTSTRAP_PENDING_KEY, value="true"))
        else:
            row.value, row.updated_at = "true", utcnow()
            db.add(row)
    elif row is not None:
        db.delete(row)
    db.commit()


def clear_bootstrap_pending(db: Session, user: User) -> None:
    """Called whenever the bootstrap admin's password is changed."""
    if user.username == BOOTSTRAP_USERNAME:
        _set_bootstrap_pending(db, False)


def ensure_bootstrap_admin(db: Session) -> None:
    """First boot: create the ``admin`` account.

    The password comes from ``ADMIN_PASSWORD`` if set; otherwise a
    random one is generated and printed once to the server log. There is no
    guessable default, so an instance exposed to a network before the operator
    logs in is not trivially takeoverable. The UI nags until a generated
    password is replaced (see /auth/me).
    """
    if db.exec(select(User)).first() is not None:
        return

    configured = (settings.admin_password or os.environ.get("ADMIN_PASSWORD") or "").strip()
    password = configured or secrets.token_urlsafe(12)
    db.add(
        User(
            username=BOOTSTRAP_USERNAME,
            password_hash=hash_password(password),
            role=UserRole.admin.value,
        )
    )
    db.commit()

    if configured:
        logger.warning(
            "No users found — created the '%s' account with the password from "
            "ADMIN_PASSWORD.", BOOTSTRAP_USERNAME,
        )
        return

    _set_bootstrap_pending(db, True)
    logger.warning(
        "\n"
        "  ┌──────────────────────────────────────────────────────────────┐\n"
        "  │  First boot — generated an admin account. Sign in with:      │\n"
        "  │                                                              │\n"
        "  │      username: %-46s│\n"
        "  │      password: %-46s│\n"
        "  │                                                              │\n"
        "  │  This is shown ONCE. Change it from Settings → Access,       │\n"
        "  │  or preset one with ADMIN_PASSWORD.                  │\n"
        "  └──────────────────────────────────────────────────────────────┘",
        BOOTSTRAP_USERNAME, password,
    )


def using_default_password(db: Session, user: User) -> bool:
    """True while the bootstrap admin still has its auto-generated password."""
    if user.username != BOOTSTRAP_USERNAME:
        return False
    return db.get(AppSetting, _BOOTSTRAP_PENDING_KEY) is not None
