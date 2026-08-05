"""Encryption-at-rest for stored secrets."""

from app.services import crypto


def test_roundtrip():
    secret = "sk-super-secret-value-123"
    token = crypto.encrypt(secret)
    assert token != secret
    assert token.startswith("enc:v1:")
    assert crypto.decrypt(token) == secret


def test_empty_stays_empty():
    assert crypto.encrypt("") == ""
    assert crypto.decrypt("") == ""


def test_encrypt_is_idempotent():
    once = crypto.encrypt("hello")
    twice = crypto.encrypt(once)
    assert once == twice  # already-encrypted values pass through unchanged


def test_plaintext_passthrough_on_decrypt():
    # Legacy un-prefixed rows are treated as plaintext, not decrypted.
    assert crypto.decrypt("not-encrypted") == "not-encrypted"


def test_undecryptable_value_treated_as_unset():
    # A corrupt/foreign token must degrade to empty, never raise.
    assert crypto.decrypt("enc:v1:not-a-real-fernet-token") == ""


def test_ciphertext_is_nondeterministic():
    # Fernet embeds a random IV, so two encryptions differ but both decrypt back.
    a, b = crypto.encrypt("same"), crypto.encrypt("same")
    assert a != b
    assert crypto.decrypt(a) == crypto.decrypt(b) == "same"
