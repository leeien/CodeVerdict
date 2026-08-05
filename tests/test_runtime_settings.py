"""Runtime settings: validation, coercion, secret masking, and persistence."""

import pytest
from app.services import runtime_settings as rs
from app.services.runtime_settings import SECRET_MASK


def test_coerce_int_bounds():
    spec = rs.FIELDS["max_revision_rounds"]
    assert rs._coerce("max_revision_rounds", "2", spec) == 2
    with pytest.raises(ValueError):
        rs._coerce("max_revision_rounds", "99", spec)   # above max
    with pytest.raises(ValueError):
        rs._coerce("max_revision_rounds", "-1", spec)   # below min


def test_coerce_bool_from_strings():
    spec = rs.FIELDS["demo_mode"]
    assert rs._coerce("demo_mode", "true", spec) is True
    assert rs._coerce("demo_mode", "off", spec) is False
    assert rs._coerce("demo_mode", "1", spec) is True


def test_update_rejects_unknown_key(db):
    with pytest.raises(ValueError):
        rs.update(db, {"nonexistent_setting": "x"})


def test_update_is_atomic_on_bad_value(db):
    before = rs.settings.max_revision_rounds
    with pytest.raises(ValueError):
        # First value is valid, second is out of range — nothing should apply.
        rs.update(db, {"max_revision_rounds": 1, "dev_max_rounds": 999})
    assert rs.settings.max_revision_rounds == before


def test_update_persists_and_live_applies(db):
    changed = rs.update(db, {"max_revision_rounds": 3})
    assert "max_revision_rounds" in changed
    assert rs.settings.max_revision_rounds == 3

    # Reloading overrides onto the singleton yields the same value.
    rs.settings.max_revision_rounds = 0
    rs.apply_overrides(db)
    assert rs.settings.max_revision_rounds == 3


def test_secret_is_encrypted_at_rest(db):
    rs.update(db, {"openai_api_key": "sk-secret-xyz"})
    from app.models import AppSetting

    row = db.get(AppSetting, "openai_api_key")
    # Stored JSON must not contain the plaintext key.
    assert "sk-secret-xyz" not in row.value
    # ...but the singleton holds the usable plaintext.
    assert rs.settings.openai_api_key == "sk-secret-xyz"


def test_masked_secret_is_ignored_on_update(db):
    rs.update(db, {"openai_api_key": "sk-real-key"})
    # Sending the mask back (the UI's placeholder for "unchanged") is a no-op.
    rs.update(db, {"openai_api_key": SECRET_MASK})
    assert rs.settings.openai_api_key == "sk-real-key"


def test_view_masks_secrets(db):
    rs.update(db, {"openai_api_key": "sk-should-be-hidden"})
    view = rs.view()
    blob = str(view)
    assert "sk-should-be-hidden" not in blob


def test_view_exposes_show_if_gates():
    """Mutually-exclusive fields carry their visibility condition so the UI can
    hide them when the driver field says they don't apply."""
    v = rs.view()
    fields = {f["name"]: f for g in v["groups"] for f in g["fields"]}
    assert fields["graph_binary"]["show_if"] == "graph_enabled=true"
    assert fields["graph_index_mode"]["show_if"] == "graph_enabled=true"
    assert fields["open_real_pr"]["show_if"] == "demo_mode=false"
    assert fields["github_bot_token"]["show_if"] == "demo_mode=false"
    # Ungated fields explicitly carry no condition.
    assert fields["demo_mode"]["show_if"] == ""


def test_view_providers_carry_card_wiring():
    """The Connections cards need each provider's key/url/path field names, and
    every stage-selectable provider a non-empty model catalog or free-text."""
    v = rs.view()
    provs = {p["id"]: p for p in v["providers"]}
    assert provs["codex"]["path_field"] == "codex_cli_path"
    assert provs["cursor-cli"]["path_field"] == "cursor_cli_path"
    assert provs["groq"]["key_field"] == "groq_api_key"
    # Model dropdowns: every provider offers a catalog except the free-text ones.
    for pid, p in provs.items():
        if pid in ("custom", "antigravity"):
            assert p["models"] == []
        else:
            assert p["models"], f"{pid} has an empty model catalog"
