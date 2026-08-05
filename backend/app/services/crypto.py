"""Encryption-at-rest for secret settings (API keys).

Secret values persisted in the ``AppSetting`` table are Fernet-encrypted and
prefixed with ``enc:v1:``. Values without the prefix are treated as plaintext and
pass through unchanged, so existing rows keep working and are transparently
re-encrypted the next time they're saved.

The encryption key comes from ``CODEVERDICT_SECRET_KEY`` (or the legacy
``AUTODEV_SECRET_KEY``; any string — derived into a
Fernet key) or, failing that, a generated key file at the agent root
(``.secret.key``, chmod 600). If the key is lost, encrypted values simply become
unreadable (returned empty with a warning) rather than crashing the app.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

_PREFIX = "enc:v1:"
_KEY_FILE = Path(__file__).resolve().parents[3] / ".secret.key"

_fernet_cached: Fernet | None = None


def _derive_key(secret: str) -> bytes:
    """A stable urlsafe-base64 Fernet key from any passphrase."""
    return base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())


def _load_or_create_key() -> bytes:
    # AUTODEV_SECRET_KEY is the pre-rename name. Still honored as a fallback:
    # dropping it would make every already-encrypted API key permanently
    # unreadable on installs that set it.
    env = (os.environ.get("CODEVERDICT_SECRET_KEY", "")
           or os.environ.get("AUTODEV_SECRET_KEY", "")).strip()
    if env:
        return _derive_key(env)
    try:
        if _KEY_FILE.exists():
            return _KEY_FILE.read_text(encoding="utf-8").strip().encode("utf-8")
        key = Fernet.generate_key()
        _KEY_FILE.write_text(key.decode("utf-8"), encoding="utf-8")
        try:
            _KEY_FILE.chmod(0o600)
        except OSError:
            pass
        return key
    except OSError as exc:
        # Can't persist a key file — fall back to a process-local ephemeral key so
        # encryption still works this run (values won't survive a restart, but the
        # masked-write-only contract is preserved).
        logger.warning("Could not read/create %s (%s); using an ephemeral key", _KEY_FILE, exc)
        return Fernet.generate_key()


def _fernet() -> Fernet:
    global _fernet_cached
    if _fernet_cached is None:
        _fernet_cached = Fernet(_load_or_create_key())
    return _fernet_cached


def is_encrypted(value: str) -> bool:
    return isinstance(value, str) and value.startswith(_PREFIX)


def encrypt(plaintext: str) -> str:
    """Encrypt a secret for storage. Empty stays empty; already-encrypted stays as-is."""
    if not plaintext:
        return ""
    if is_encrypted(plaintext):
        return plaintext
    token = _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")
    return _PREFIX + token


def decrypt(value: str) -> str:
    """Decrypt a stored secret. Plaintext (un-prefixed) passes through unchanged.
    A value we can't decrypt (wrong/lost key) is treated as unset."""
    if not value or not is_encrypted(value):
        return value or ""
    try:
        return _fernet().decrypt(value[len(_PREFIX):].encode("utf-8")).decode("utf-8")
    except (InvalidToken, ValueError) as exc:
        logger.warning("Could not decrypt a stored secret (%s) — treating as unset", exc)
        return ""
