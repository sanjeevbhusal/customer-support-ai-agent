"""Shared fixtures for the test suite.

Assumes Postgres is running and OPENAI_API_KEY is set. The only marker is
`slow`: tests that make real LLM calls, skipped unless `--run-slow` is passed.
"""

import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Repo Root path should be added so files in the project can be imported (makes `import tools` / `import server` work)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tracing  # noqa: E402  (needs sys.path above)
from auth import hash_password  # noqa: E402
from db import pool  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


TEST_USER = {
    "first_name": "Pytest",
    "last_name": "User",
    "email": "pytest-user@example.com",
    "password": "pytest-password",
}


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests that make real LLM calls",
    )


def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("slow test; pass --run-slow to run it")


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def test_user():
    """Setup a test user to use in tests.

    When the test finishes, cleans up the user and other created resources.
    """

    password_hash = hash_password(TEST_USER["password"])
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO users (first_name, last_name, email, password_hash)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (email) DO UPDATE
                SET password_hash = EXCLUDED.password_hash,
                    first_name = EXCLUDED.first_name,
                    last_name = EXCLUDED.last_name
            RETURNING id;
            """,
            (
                TEST_USER["first_name"],
                TEST_USER["last_name"],
                TEST_USER["email"],
                password_hash,
            ),
        )
        user_id = cur.fetchone()[0]

    yield {**TEST_USER, "id": user_id}

    # CLEANUP
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM orders WHERE user_id = %s;", (user_id,))
        cur.execute("DELETE FROM users WHERE id = %s;", (user_id,))


@pytest.fixture
def no_db(monkeypatch):
    """Capture persisted traces instead of writing to Postgres."""
    written: list = []
    monkeypatch.setattr(
        tracing, "_persist_trace", lambda trace, spans: written.append((trace, spans))
    )
    return written


@pytest.fixture
def a_product():
    """Setup a product to use in tests.

    When the test finishes, reset the product quantity.
    """

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, quantity FROM inventory "
            "WHERE quantity > 0 ORDER BY quantity DESC LIMIT 1;"
        )
        row = cur.fetchone()
    if row is None:
        pytest.skip("no in-stock product to order")

    product_id, name, original_qty = row
    yield {"id": product_id, "name": name, "quantity": original_qty}

    # CLEANUP
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE inventory SET quantity = %s WHERE id = %s;",
            (original_qty, product_id),
        )
