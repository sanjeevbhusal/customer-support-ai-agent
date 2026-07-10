"""Authentication and session helpers.

Everything here supports signing customers in and keeping that identity trusted:
password hashing, credential checks and user lookups, the signed token that
persists a login in the browser, and injecting the authenticated user's id into
tool calls. None of it is exposed to the model - the model only sees the tools
in `tools.py`.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import TYPE_CHECKING

from db import pool

if TYPE_CHECKING:
    from streamlit_local_storage import LocalStorage

_PBKDF2_ROUNDS = 200_000
_MAX_AGE_SECONDS = 7 * 24 * 60 * 60  # 7 days

# Key under which the signed login token is stored in the browser's localStorage.
AUTH_TOKEN_KEY = "auth_token"

# Tools whose `user_id` is injected from the authenticated session.
AUTH_REQUIRED_TOOLS = {"get_current_user", "place_order", "get_orders"}


# --- Password hashing ---------------------------------------------------------


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Return a salted PBKDF2 hash as 'salt_hex$hash_hex' for storage."""
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return f"{salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    """Check a plaintext password against a stored 'salt_hex$hash_hex' value."""
    try:
        salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return secrets.compare_digest(dk.hex(), hash_hex)


# --- Signed session token -----------------------------------------------------


def _secret() -> bytes:
    return os.environ["APP_SECRET_KEY"].encode()


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(payload: bytes, secret: bytes) -> bytes:
    return hmac.new(secret, payload, hashlib.sha256).digest()


def make_session_token(user_id: int) -> str:
    """Return a signed token identifying `user_id`."""
    payload = json.dumps({"user_id": user_id, "issued_at": int(time.time())}).encode()
    return f"{_b64encode(payload)}.{_b64encode(_sign(payload, _secret()))}"


def read_session_token(token: str, max_age: int = _MAX_AGE_SECONDS) -> int | None:
    """Return the user id from a valid token, or None if it is missing,
    malformed, tampered with, or expired."""
    secret = _secret()
    try:
        payload_b64, signature_b64 = token.split(".")
        payload = _b64decode(payload_b64)
        signature = _b64decode(signature_b64)
    except ValueError:
        return None

    if not hmac.compare_digest(signature, _sign(payload, secret)):
        return None

    try:
        data = json.loads(payload)
        user_id = int(data["user_id"])
        issued_at = int(data["issued_at"])
    except (ValueError, KeyError, TypeError):
        return None

    if time.time() - issued_at > max_age:
        return None

    return user_id


# --- User lookups -------------------------------------------------------------


def authenticate_user(email: str, password: str) -> dict | None:
    """Validate an email/password pair against the stored account.

    Used by the app's login UI (not a model-callable tool). Identity established
    here is what gets injected into the auth-required tools.

    Args:
        email: The email address the customer registered with.
        password: The customer's account password.

    Returns:
        The account's details (`id`, `first_name`, `last_name`, `email`) on a
        successful match, or `None` if the email is unknown or the password is
        wrong. The password/hash is never returned.
    """
    email = email.strip().lower()

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, first_name, last_name, email, password_hash
            FROM users
            WHERE email = %s;
            """,
            (email,),
        )
        row = cur.fetchone()

    if row is None:
        return None

    user_id, first_name, last_name, email, password_hash = row
    if not _verify_password(password, password_hash):
        return None

    return {
        "id": user_id,
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
    }


def get_user_by_id(user_id: int) -> dict | None:
    """Load a user's public details by id.

    Returns `{id, first_name, last_name, email}`, or `None` if no such user
    exists.
    """
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, first_name, last_name, email
            FROM users
            WHERE id = %s;
            """,
            (user_id,),
        )
        row = cur.fetchone()

    if row is None:
        return None

    user_id, first_name, last_name, email = row
    return {
        "id": user_id,
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
    }


# --- Tool-call identity injection ---------------------------------------------


def resolve_tool_args(
    tool_name: str, args: dict, auth_user: dict | None
) -> tuple[dict | None, dict | None]:
    """Inject the signed-in customer's id into auth-required tool calls.

    Returns a `(resolved_args, error)` pair with exactly one side non-None:
    `(args, None)` if the call may proceed, or `(None, error_dict)` if an
    auth-required tool was called with no one signed in.
    """
    resolved = dict(args)
    if tool_name not in AUTH_REQUIRED_TOOLS:
        return resolved, None

    if auth_user is None:
        return None, {
            "error": "The customer is not signed in. Ask them to log in using "
            "the sidebar."
        }

    resolved["user_id"] = auth_user["id"]
    return resolved, None


# --- Persisted login (browser localStorage) -----------------------------------


def load_persisted_user(local_storage: "LocalStorage | None") -> dict | None:
    """Restore the signed-in customer from the persisted login token, if valid."""
    if local_storage is None:
        return None

    try:
        token = local_storage.getItem(AUTH_TOKEN_KEY)
    except Exception:
        return None

    if not token:
        return None

    user_id = read_session_token(token)
    if user_id is None:
        return None

    return get_user_by_id(user_id)


def persist_login(local_storage: "LocalStorage | None", user: dict) -> None:
    """Store a signed token for `user` so the login survives a refresh."""
    if local_storage is None:
        return

    token = make_session_token(user["id"])
    try:
        local_storage.setItem(AUTH_TOKEN_KEY, token, key="persist_login")
    except Exception:
        pass


def clear_persisted_login(local_storage: "LocalStorage | None") -> None:
    """Remove the persisted login token."""
    if local_storage is None:
        return
    try:
        local_storage.deleteItem(AUTH_TOKEN_KEY, key="clear_login")
    except Exception:
        pass
