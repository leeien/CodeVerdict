"""Passwords, the bootstrap admin, and role ranking."""



from app.models import ROLE_RANK, User, UserRole
from app.services import auth


def test_password_hash_verify_roundtrip():
    stored = auth.hash_password("hunter2")
    assert stored != "hunter2"
    assert stored.startswith("pbkdf2_sha256$")
    assert auth.verify_password("hunter2", stored)
    assert not auth.verify_password("wrong", stored)


def test_hash_is_salted():
    # Two hashes of the same password differ (random salt) yet both verify.
    a, b = auth.hash_password("pw"), auth.hash_password("pw")
    assert a != b
    assert auth.verify_password("pw", a)
    assert auth.verify_password("pw", b)


def test_verify_rejects_malformed_stored_value():
    assert not auth.verify_password("x", "garbage")
    assert not auth.verify_password("x", "")


def test_role_rank_ordering():
    assert ROLE_RANK[UserRole.viewer.value] < ROLE_RANK[UserRole.member.value]
    assert ROLE_RANK[UserRole.member.value] < ROLE_RANK[UserRole.admin.value]


def test_bootstrap_uses_configured_password(db, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "chosen-pw")
    monkeypatch.setattr(auth.settings, "admin_password", "chosen-pw", raising=False)
    auth.ensure_bootstrap_admin(db)

    from sqlmodel import select

    user = db.exec(select(User)).first()
    assert user is not None
    assert user.username == "admin"
    assert user.role == UserRole.admin.value
    assert auth.verify_password("chosen-pw", user.password_hash)
    # A preset password is not the "still-default" state.
    assert auth.using_default_password(db, user) is False


def test_bootstrap_generates_random_password_when_unset(db, monkeypatch, caplog):
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.setattr(auth.settings, "admin_password", "", raising=False)
    auth.ensure_bootstrap_admin(db)

    from sqlmodel import select

    user = db.exec(select(User)).first()
    # No guessable default — the classic admin/admin must NOT work.
    assert not auth.verify_password("admin", user.password_hash)
    # Generated password is flagged as pending until the operator changes it.
    assert auth.using_default_password(db, user) is True


def test_bootstrap_is_idempotent(db, monkeypatch):
    monkeypatch.setattr(auth.settings, "admin_password", "pw", raising=False)
    auth.ensure_bootstrap_admin(db)
    auth.ensure_bootstrap_admin(db)

    from sqlmodel import select

    assert len(db.exec(select(User)).all()) == 1


def test_clear_bootstrap_pending(db, monkeypatch):
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.setattr(auth.settings, "admin_password", "", raising=False)
    auth.ensure_bootstrap_admin(db)

    from sqlmodel import select

    user = db.exec(select(User)).first()
    assert auth.using_default_password(db, user) is True

    auth.clear_bootstrap_pending(db, user)
    assert auth.using_default_password(db, user) is False
