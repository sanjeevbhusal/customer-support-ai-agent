"""Tests for auth helpers in auth.py."""

import pytest

from auth import (
    AUTH_REQUIRED_TOOLS,
    authenticate_user,
    get_user_by_id,
    make_session_token,
    read_session_token,
    resolve_tool_args,
)


# --- Signed session token -----------------------------------------------------


def test_round_trip_returns_user_id():
    """A freshly signed token reads back to the same user id."""
    token = make_session_token(42)
    assert read_session_token(token) == 42


def test_tampered_token_is_rejected():
    """Changing the payload invalidates the signature."""
    token = make_session_token(42)
    # Mutate the first payload character; its bits map directly into byte 0, so
    # the decoded payload really changes and the signature no longer matches.
    tampered = ("A" if token[0] != "A" else "B") + token[1:]
    assert read_session_token(tampered) is None


def test_expired_token_is_rejected():
    """A token older than max_age is rejected."""
    token = make_session_token(42)
    assert read_session_token(token, max_age=-1) is None


def test_garbage_tokens_are_rejected():
    """Missing or malformed tokens return None rather than raising."""
    assert read_session_token("") is None
    assert read_session_token("not-a-token") is None
    assert read_session_token("a.b.c") is None


def test_token_signed_with_other_secret_is_rejected(monkeypatch):
    """A token signed under a different secret does not verify."""
    token = make_session_token(42)
    monkeypatch.setenv("APP_SECRET_KEY", "a-different-secret")
    assert read_session_token(token) is None


# --- Tool-call identity injection ---------------------------------------------


def test_resolve_injects_session_user_id():
    """The signed-in user's id is injected into an auth tool's args."""
    args, error = resolve_tool_args("get_orders", {}, {"id": 42})
    assert error is None
    assert args == {"user_id": 42}


@pytest.mark.parametrize("name", AUTH_REQUIRED_TOOLS)
def test_resolve_blocks_when_not_signed_in(name):
    """Auth tools are refused with an error when no one is signed in."""
    args, error = resolve_tool_args(name, {}, None)
    assert args is None
    assert error is not None
    assert "not signed in" in error["error"].lower()


def test_resolve_overwrites_args_user_id_with_session_id():
    """A `user_id` in the args is overwritten with the session user's id."""
    args, error = resolve_tool_args("get_orders", {"user_id": 999}, {"id": 1})
    assert error is None
    assert args["user_id"] == 1


def test_resolve_passes_through_non_auth_tool():
    """Non-auth tools have their args passed through unchanged."""
    args, error = resolve_tool_args(
        "search_product_inventory", {"query": "roses"}, None
    )
    assert error is None
    assert args == {"query": "roses"}


def test_resolve_does_not_mutate_caller_args():
    """Resolving args does not mutate the caller's original dict."""
    original = {"product_id": "P001", "quantity": 2, "delivery_location": "X"}
    resolve_tool_args("place_order", original, {"id": 7})
    assert "user_id" not in original


# --- Credential checks and user lookups ---------------------------------------


def test_authenticate_success(test_user):
    """Valid credentials return the account without any secret fields."""
    result = authenticate_user(test_user["email"], test_user["password"])
    assert result is not None
    assert result["id"] == test_user["id"]
    assert result["email"] == test_user["email"]
    assert "password" not in result
    assert "password_hash" not in result


def test_authenticate_wrong_password(test_user):
    """A wrong password fails authentication."""
    assert authenticate_user(test_user["email"], "not-the-password") is None


def test_authenticate_unknown_email():
    """An unknown email fails authentication."""
    assert authenticate_user("does-not-exist@example.com", "whatever") is None


def test_authenticate_is_case_insensitive_on_email(test_user):
    """Authentication matches the email case-insensitively."""
    upper = test_user["email"].upper()
    result = authenticate_user(upper, test_user["password"])
    assert result is not None and result["id"] == test_user["id"]


def test_get_user_by_id_returns_user(test_user):
    """get_user_by_id loads the account for an existing id."""
    user = get_user_by_id(test_user["id"])
    assert user is not None
    assert user["id"] == test_user["id"]
    assert user["email"] == test_user["email"]


def test_get_user_by_id_unknown_returns_none():
    """get_user_by_id returns None for an id that does not exist."""
    assert get_user_by_id(-1) is None
