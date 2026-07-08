"""End-to-end agent tests that make real LLM calls (gpt-4o-mini).

Run with:  uv run pytest --run-slow
"""

import pytest
from langchain.messages import AIMessage, ToolMessage
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.slow


def _app(repo_root):
    at = AppTest.from_file(str(repo_root / "server.py"), default_timeout=180)
    at.run()
    return at


def _login(at, user):
    at.sidebar.text_input[0].set_value(user["email"])
    at.sidebar.text_input[1].set_value(user["password"])
    [b for b in at.sidebar.button if b.label == "Log in"][0].click()
    at.run()


def _ask(at, text):
    at.chat_input[0].set_value(text).run()


def _tool_calls(at, name=None):
    calls = [
        tc
        for m in at.session_state["messages"]
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
    ]
    return [tc for tc in calls if name is None or tc["name"] == name]


def _tool_result(at, tool_call_id):
    for m in at.session_state["messages"]:
        if isinstance(m, ToolMessage) and m.tool_call_id == tool_call_id:
            return m.content
    return None


def _last_ai_text(at):
    for m in reversed(at.session_state["messages"]):
        if isinstance(m, AIMessage) and m.content:
            return m.content
    return ""


def test_logged_in_who_am_i(repo_root, test_user):
    """Signed in, the agent identifies the user via get_current_user."""
    at = _app(repo_root)
    _login(at, test_user)
    _ask(at, "Who am I? Tell me the first name and email on my account.")

    calls = _tool_calls(at, "get_current_user")
    assert calls, "expected the model to call get_current_user"

    result = _tool_result(at, calls[-1]["id"])
    assert test_user["email"] in result
    assert test_user["first_name"].lower() in _last_ai_text(at).lower()


def test_logged_in_show_orders(repo_root, test_user):
    """Signed in, get_orders runs for the session user."""
    at = _app(repo_root)
    _login(at, test_user)
    _ask(at, "Show me my past orders.")

    calls = _tool_calls(at, "get_orders")
    assert calls, "expected the model to call get_orders"

    result = _tool_result(at, calls[-1]["id"])
    assert "not signed in" not in result.lower()


def test_logged_out_order_lookup_is_refused(repo_root):
    """Signed out, an order lookup is either blocked by the guard or the agent
    directs the user to log in."""
    at = _app(repo_root)
    assert at.session_state["auth_user"] is None
    _ask(
        at,
        "Call the get_orders tool right now to list my order history. "
        "Do not ask me anything first, just call the tool.",
    )

    calls = _tool_calls(at, "get_orders")
    if calls:
        # If the model called it, the guard must have blocked it.
        result = _tool_result(at, calls[-1]["id"])
        assert "not signed in" in result.lower()
    else:
        # Otherwise it steered the customer to the sidebar login.
        text = _last_ai_text(at).lower()
        assert any(k in text for k in ["sidebar", "log in", "sign in", "logged in"])
