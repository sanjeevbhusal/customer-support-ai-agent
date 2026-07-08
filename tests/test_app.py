"""End-to-end tests of the sidebar auth UI, driven through Streamlit's AppTest.

These boot the real `server.py` and validate credentials against Postgres, but
make no LLM calls, so they are fast and deterministic.
"""

from streamlit.testing.v1 import AppTest


def _app(repo_root):
    at = AppTest.from_file(str(repo_root / "server.py"), default_timeout=60)
    at.run()
    return at


def _button(at, label):
    return [b for b in at.sidebar.button if b.label == label]


def _login(at, email, password):
    at.sidebar.text_input[0].set_value(email)
    at.sidebar.text_input[1].set_value(password)
    _button(at, "Log in")[0].click()
    at.run()


def test_starts_logged_out(repo_root):
    """A fresh app starts signed out and shows the login form."""
    at = _app(repo_root)
    assert at.session_state["auth_user"] is None
    assert _button(at, "Log in"), "login form should be shown when logged out"


def test_login_success(repo_root, test_user):
    """Valid credentials sign the customer in and show their account."""
    at = _app(repo_root)
    _login(at, test_user["email"], test_user["password"])

    auth = at.session_state["auth_user"]
    assert auth is not None and auth["id"] == test_user["id"]

    sidebar_text = " ".join(m.value for m in at.sidebar.markdown)
    assert test_user["email"] in sidebar_text
    assert _button(at, "Log out"), "Log out button should appear once signed in"
    assert not at.sidebar.text_input, "login form should be hidden once signed in"


def test_login_wrong_password_shows_error(repo_root, test_user):
    """A wrong password leaves the app signed out and shows an error."""
    at = _app(repo_root)
    _login(at, test_user["email"], "wrong-password")

    assert at.session_state["auth_user"] is None
    assert at.error, "an error should be shown for a bad password"


def test_logout_clears_identity_and_chat(repo_root, test_user):
    """Logging out clears the identity and resets the conversation."""
    at = _app(repo_root)
    _login(at, test_user["email"], test_user["password"])
    assert at.session_state["auth_user"] is not None

    _button(at, "Log out")[0].click()
    at.run()

    assert at.session_state["auth_user"] is None
    assert len(at.session_state["messages"]) == 2  # system + greeting only
    assert _button(at, "Log in"), "login form should return after logout"
