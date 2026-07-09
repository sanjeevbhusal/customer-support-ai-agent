"""Tests for the model-facing tools in tools.py."""

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from auth import AUTH_REQUIRED_TOOLS
from tools import tools_by_name


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
