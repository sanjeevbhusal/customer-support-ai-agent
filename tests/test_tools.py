"""Tests for tools in tools.py."""

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from tools import (
    AUTH_REQUIRED_TOOLS,
    authenticate_user,
    resolve_tool_args,
    tools_by_name,
)


@pytest.mark.parametrize("name", AUTH_REQUIRED_TOOLS)
def test_user_id_hidden_from_model_schema(name):
    """Auth tools do not expose `user_id` in the schema shown to the model."""
    schema = convert_to_openai_tool(tools_by_name[name])
    props = schema["function"]["parameters"].get("properties", {})
    assert "user_id" not in props


def test_place_order_exposes_only_business_args():
    """place_order shows the model only its business arguments."""
    schema = convert_to_openai_tool(tools_by_name["place_order"])
    props = schema["function"]["parameters"]["properties"]
    assert set(props) == {"product_id", "quantity", "delivery_location"}


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


def test_get_current_user_returns_signed_in_user(test_user):
    """get_current_user returns the account details for the given user id."""
    result = tools_by_name["get_current_user"].invoke({"user_id": test_user["id"]})
    assert result["id"] == test_user["id"]
    assert result["email"] == test_user["email"]


def test_get_orders_returns_list(test_user):
    """get_orders returns a list of the user's orders."""
    result = tools_by_name["get_orders"].invoke({"user_id": test_user["id"]})
    assert isinstance(result, list)


def test_place_order_creates_order_and_reduces_stock(test_user, a_product):
    """place_order creates the order, reduces stock, and it shows in get_orders."""
    result = tools_by_name["place_order"].invoke(
        {
            "product_id": a_product["id"],
            "quantity": 1,
            "delivery_location": "Test City",
            "user_id": test_user["id"],
        }
    )
    assert "error" not in result, result
    assert result["product_id"] == a_product["id"]
    assert result["quantity"] == 1
    assert result["remaining_stock"] == a_product["quantity"] - 1

    orders = tools_by_name["get_orders"].invoke({"user_id": test_user["id"]})
    assert any(o["order_id"] == result["order_id"] for o in orders)
