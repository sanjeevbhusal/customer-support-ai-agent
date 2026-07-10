"""Run the eval dataset through the agent.

Stage 1: execute each case and print its trajectory (tool calls + final reply)
and trace metrics. Scoring is added in later stages.

    uv run python -m evals.run_evals
"""

from typing import cast

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agent import SYSTEM_PROMPT, agent
from auth import hash_password
from db import pool
from evals.dataset import CASES, EvalCase
from tracing import TraceCollector

load_dotenv()


EVAL_USER = {
    "first_name": "Eval",
    "last_name": "User",
    "email": "eval-user@example.com",
    "password": "eval-password",
}


def ensure_eval_user() -> dict:
    """Upsert the dedicated eval user; return {id, first_name, last_name, email}."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO users (first_name, last_name, email, password_hash)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (email) DO UPDATE SET password_hash = EXCLUDED.password_hash
            RETURNING id, first_name, last_name, email;
            """,
            (
                EVAL_USER["first_name"],
                EVAL_USER["last_name"],
                EVAL_USER["email"],
                hash_password(EVAL_USER["password"]),
            ),
        )
        user_id, first_name, last_name, email = cur.fetchone()
    return {
        "id": user_id,
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
    }


def restore_after_orders(user_id: int) -> None:
    """Add back stock for the eval user's orders, then delete those orders, so
    order-placing cases don't drain inventory across runs."""
    with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute(
            """
            UPDATE inventory i
            SET quantity = i.quantity + o.qty
            FROM (
                SELECT product_id, sum(quantity) AS qty
                FROM orders WHERE user_id = %s GROUP BY product_id
            ) o
            WHERE i.id = o.product_id;
            """,
            (user_id,),
        )
        cur.execute("DELETE FROM orders WHERE user_id = %s;", (user_id,))


def build_messages(case: EvalCase) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    for role, text in case.history:
        messages.append(
            HumanMessage(content=text) if role == "user" else AIMessage(content=text)
        )
    messages.append(HumanMessage(content=case.user_message))
    return messages


def run_case(case: EvalCase, eval_user: dict) -> tuple[dict, dict]:
    """Invoke the agent for one case; return (result, trace)."""
    auth_user = eval_user if case.auth == "user" else None
    collector = TraceCollector(
        source="eval",
        thread_id=f"eval:{case.id}",
        user_id=auth_user["id"] if auth_user else None,
    )
    result = agent.invoke(
        {"messages": build_messages(case)},
        config={"configurable": {"auth_user": auth_user}, "callbacks": [collector]},
    )
    return result, collector.finish(result=result)


def trajectory(result: dict) -> tuple[list[str], str]:
    """Tool names in call order, and the final assistant reply."""
    tool_calls = [
        tc["name"]
        for m in result["messages"]
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
    ]
    reply = next(
        (
            cast(str, m.content)
            for m in reversed(result["messages"])
            if isinstance(m, AIMessage) and m.content
        ),
        "",
    )
    return tool_calls, reply


def main() -> None:
    eval_user = ensure_eval_user()
    try:
        for case in CASES:
            result, trace = run_case(case, eval_user)
            tools, reply = trajectory(result)
            signed = "signed-in" if case.auth == "user" else "signed-out"
            print(f"\n=== {case.id} ({signed}) ===")
            print(f"tools: {tools}")
            print(
                f"latency={trace['latency_ms']}ms tokens={trace['total_tokens']} "
                f"cost=${trace['cost_usd']:.4f} status={trace['status']}"
            )
            print(f"reply: {reply[:200]}")
    finally:
        restore_after_orders(eval_user["id"])


if __name__ == "__main__":
    main()
