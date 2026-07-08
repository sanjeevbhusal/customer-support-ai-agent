import hashlib
import secrets
from typing import Annotated

from langchain.tools import tool
from langchain_core.tools import InjectedToolArg

from vector_store import load_model, pool

_PBKDF2_ROUNDS = 200_000


def _hash_password(password: str, salt: bytes | None = None) -> str:
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


@tool
def search_business_faqs(query: str) -> list[dict]:
    """Search the shop's FAQ knowledge base for questions about how the business
    operates: delivery areas and timing, same-day delivery, payment methods,
    refunds and returns, order changes, opening hours, and similar policies.

    Args:
        query: A natural-language description of the customer's question,
            e.g. "do you deliver internationally" or "what is the return policy".

    Returns:
        Up to 3 of the most relevant FAQ entries, each with `question` and `answer`.
    """
    model = load_model()
    query_embedding = model.encode(query, normalize_embeddings=True)

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                question,
                answer
            FROM faqs
            ORDER BY embedding <=> %s
            LIMIT 3;
            """,
            (query_embedding,),
        )

        return [
            {"question": question, "answer": answer}
            for question, answer in cur.fetchall()
        ]


@tool
def search_product_inventory(query: str) -> list[dict]:
    """Search the live product inventory for flowers and arrangements, returning
    their availability and current price. This is the source of truth for what is
    in stock and how much it costs.

    Use this when the customer asks about specific flowers, bouquets, arrangements,
    occasions (e.g. birthday, wedding), colors, types, or budget.

    Args:
        query: A natural-language description of what the customer is looking for,
            e.g. "white tulips for a birthday" or "affordable mixed bouquet".

    Returns:
        Up to 5 matching products, each with `id`, `name`, `price` (USD),
        `quantity` in stock, `type`, and `description`.
    """
    model = load_model()
    query_embedding = model.encode(query, normalize_embeddings=True)

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                id,
                name,
                price,
                quantity,
                type,
                description
            FROM inventory
            ORDER BY embedding <=> %s
            LIMIT 5;
            """,
            (query_embedding,),
        )

        return [
            {
                "id": id,
                "name": name,
                "price": float(price),
                "quantity": quantity,
                "type": type,
                "description": description,
            }
            for id, name, price, quantity, type, description in cur.fetchall()
        ]


def authenticate_user(email: str, password: str) -> dict | None:
    """Validate an email/password pair against the stored account.

    This is the trusted, server-side authentication used by the app's login UI
    (not a model-callable tool). Identity established here is what gets injected
    into the auth-required tools, so the model never supplies a user id.

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


@tool
def get_current_user(user_id: Annotated[int, InjectedToolArg]) -> dict:
    """Look up the details of the customer who is currently signed in.

    Use this to find out who you are talking to - for example to greet them by
    name or to confirm the account an order will be placed under. It works only
    when a customer is signed in; if no one is signed in, you will receive an
    error telling you to ask the customer to log in using the sidebar.

    Returns:
        The signed-in customer's `id`, `first_name`, `last_name`, and `email`.
        On failure, a dict with an `error` message.
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
        return {"error": "No signed-in customer was found."}

    user_id, first_name, last_name, email = row
    return {
        "id": user_id,
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
    }


@tool
def place_order(
    product_id: str,
    quantity: int,
    delivery_location: str,
    user_id: Annotated[int, InjectedToolArg],
) -> dict:
    """Place an order for the signed-in customer and reduce the product's stock by
    the ordered quantity. The order will only succeed if the product exists and
    enough units are in stock; otherwise nothing is purchased.

    Use this only after the customer is signed in, has chosen a specific product,
    and has provided a delivery location. Confirm the product, quantity, and
    delivery location with the customer before calling. The signed-in customer's
    identity is handled automatically - you do not supply it.

    Args:
        product_id: The unique id of the product to order, as returned by
            `search_product_inventory` (e.g. "P002").
        quantity: How many units of the product to order. Must be at least 1.
        delivery_location: The general delivery location, e.g.
            "Shankhamul, Kathmandu".

    Returns:
        On success, the order's details: `order_id`, `product_id`,
        `product_name`, `quantity`, `unit_price` (USD), `delivery_location`,
        `remaining_stock`, and `created_at`. On failure, a dict with an `error`
        message, such as the product not existing or not having enough stock.
    """
    if quantity < 1:
        return {"error": "Quantity must be at least 1."}

    with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
        # Check if the product exists. If not return error.
        # FOR UPDATE locks the row until the transaction commits, so a
        # concurrent order cannot deplete the stock between this check and
        # the update below.
        cur.execute(
            "SELECT name, price, quantity FROM inventory WHERE id = %s FOR UPDATE;",
            (product_id,),
        )
        product = cur.fetchone()
        if product is None:
            return {"error": f"No product found with id '{product_id}'."}

        name, price, available = product

        # Check if there is enough quantity left.
        if available < quantity:
            return {
                "error": f"'{name}' does not have enough stock. Requested "
                f"{quantity}, but only {available} available."
            }

        # Place order
        cur.execute(
            """
            INSERT INTO orders (user_id, product_id, quantity, delivery_location)
            VALUES (%s, %s, %s, %s)
            RETURNING id, created_at;
            """,
            (user_id, product_id, quantity, delivery_location),
        )

        order_id, created_at = cur.fetchone()

        # Reduce inventory by order quantity
        cur.execute(
            """
            UPDATE inventory
            SET quantity = quantity - %s
            WHERE id = %s
            RETURNING quantity;
            """,
            (quantity, product_id),
        )
        remaining = cur.fetchone()[0]

    return {
        "order_id": order_id,
        "product_id": product_id,
        "product_name": name,
        "quantity": quantity,
        "unit_price": float(price),
        "delivery_location": delivery_location,
        "remaining_stock": remaining,
        "created_at": str(created_at),
    }


@tool
def get_orders(user_id: Annotated[int, InjectedToolArg]) -> dict | list[dict]:
    """Retrieve all existing orders placed by the signed-in customer, most recent first.

    Use this when a signed-in customer asks about their order history or the status
    of what they have ordered. The signed-in customer's identity is handled
    automatically - you do not supply it.

    Returns:
        A list of the customer's orders, each with `order_id`, `product_id`,
        `product_name`, `quantity`, `unit_price` (USD), `delivery_location`, and
        `created_at`. An empty list means the customer has not placed any orders.
    """
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                o.id,
                o.product_id,
                i.name,
                o.quantity,
                i.price,
                o.delivery_location,
                o.created_at
            FROM orders o
            JOIN inventory i ON i.id = o.product_id
            WHERE o.user_id = %s
            ORDER BY o.created_at DESC;
            """,
            (user_id,),
        )

        orders = cur.fetchall()

        return [
            {
                "order_id": order_id,
                "product_id": product_id,
                "product_name": name,
                "quantity": quantity,
                "unit_price": float(price),
                "delivery_location": delivery_location,
                "created_at": str(created_at),
            }
            for order_id, product_id, name, quantity, price, delivery_location, created_at in orders
        ]


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


tools = [
    search_business_faqs,
    search_product_inventory,
    get_current_user,
    place_order,
    get_orders,
]
tools_by_name = {tool.name: tool for tool in tools}

# Tools whose `user_id` is injected from the authenticated session.
AUTH_REQUIRED_TOOLS = {"get_current_user", "place_order", "get_orders"}
