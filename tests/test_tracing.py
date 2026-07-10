"""Tests for the tracing collector (tracing.py)."""

import logging
from uuid import uuid4

import pytest
from langchain.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from db import pool
from tracing import TraceCollector, _compute_cost_usd, _truncate


def _llm_result(
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    model: str = "gpt-4o-mini-2024-07-18",
    tool_calls: list | None = None,
) -> LLMResult:
    """A minimal LLMResult shaped like what ChatOpenAI returns."""
    message = AIMessage(
        content="hello",
        tool_calls=tool_calls or [],
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": cache_read},
        },
        response_metadata={"model_name": model},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def _run_llm(collector: TraceCollector, **kwargs) -> None:
    run_id = uuid4()
    collector.on_llm_start(serialized={}, prompts=[], run_id=run_id)
    collector.on_llm_end(_llm_result(**kwargs), run_id=run_id)


def _run_tool(collector: TraceCollector, name: str, inputs: dict, output) -> None:
    run_id = uuid4()
    collector.on_tool_start({"name": name}, "", run_id=run_id, inputs=inputs)
    collector.on_tool_end(output, run_id=run_id)


# --- span accumulation + rollups ----------------------------------------------


def test_llm_spans_roll_up_tokens(no_db):
    """Token totals across LLM spans are summed onto the trace."""
    collector = TraceCollector(source="eval")
    _run_llm(collector, input_tokens=100, output_tokens=10, cache_read=40)
    _run_llm(collector, input_tokens=50, output_tokens=5)

    trace = collector.finish(result={"messages": [AIMessage(content="hi")]})
    assert trace["model"] == "gpt-4o-mini-2024-07-18"
    assert trace["prompt_tokens"] == 150
    assert trace["completion_tokens"] == 15
    assert trace["total_tokens"] == 165
    assert trace["cached_input_tokens"] == 40
    assert trace["num_spans"] == 2
    assert trace["status"] == "ok"


def test_llm_span_records_requested_tool_calls(no_db):
    """The tools an LLM step asked to call are captured in the span."""
    collector = TraceCollector(source="eval")
    _run_llm(
        collector,
        input_tokens=10,
        output_tokens=1,
        tool_calls=[{"name": "get_orders", "args": {}, "id": "1", "type": "tool_call"}],
    )
    assert collector.spans[0]["attributes"]["tool_calls"] == ["get_orders"]


def test_tool_span_captures_args_and_result(no_db):
    """A tool span records its name, args, and result."""
    collector = TraceCollector(source="eval")
    _run_tool(collector, "place_order", {"product_id": "P1", "quantity": 2}, {"order_id": 9})

    span = collector.spans[0]
    assert span["type"] == "tool"
    assert span["name"] == "place_order"
    assert span["attributes"]["args"] == {"product_id": "P1", "quantity": 2}
    assert span["attributes"]["result"] == {"order_id": 9}


def test_finish_error_marks_status_and_zero_cost(no_db):
    """An errored turn is recorded with status=error, the message, and no cost."""
    collector = TraceCollector(source="app")
    trace = collector.finish(error=RuntimeError("boom"))
    assert trace["status"] == "error"
    assert trace["error"] == "boom"
    assert trace["cost_usd"] == 0.0
    assert trace["model"] is None


def test_user_and_reply_text_extracted(no_db):
    """input_text is the last human message; output_text the final reply."""
    collector = TraceCollector(source="app")
    result = {
        "messages": [
            HumanMessage(content="show my orders"),
            AIMessage(content="here they are"),
        ]
    }
    trace = collector.finish(result=result)
    assert trace["input_text"] == "show my orders"
    assert trace["output_text"] == "here they are"


# --- cost ---------------------------------------------------------------------


def test_cost_includes_cached_discount():
    cost = _compute_cost_usd("gpt-4o-mini", input_tokens=100, cached=40, output=10)
    expected = 60 * 0.15 / 1e6 + 40 * 0.075 / 1e6 + 10 * 0.60 / 1e6
    assert cost == pytest.approx(expected)


def test_cost_matches_dated_model_by_prefix():
    dated = _compute_cost_usd("gpt-4o-mini-2024-07-18", 100, 0, 10)
    base = _compute_cost_usd("gpt-4o-mini", 100, 0, 10)
    assert dated == base


def test_cost_unknown_model_logs_and_returns_zero(caplog):
    with caplog.at_level(logging.ERROR):
        cost = _compute_cost_usd("some-unpriced-model", 100, 0, 10)
    assert cost == 0.0
    assert "no pricing configured" in caplog.text


# --- size cap -----------------------------------------------------------------


def test_truncate_caps_long_strings_but_keeps_structure():
    long = "x" * 5000
    out = _truncate({"a": long, "b": [1, 2, "ok"]})
    assert out["a"].endswith("…[truncated]")
    assert len(out["a"]) < len(long)
    assert out["b"] == [1, 2, "ok"]


# --- real Postgres round-trip -------------------------------------------------


def test_finish_persists_to_db():
    """finish writes the trace and its spans, and ON DELETE CASCADE cleans up."""
    collector = TraceCollector(source="eval", thread_id="test-thread", user_id=None)
    _run_llm(collector, input_tokens=10, output_tokens=2)
    _run_tool(collector, "get_orders", {}, [])
    trace = collector.finish(
        result={"messages": [HumanMessage(content="hi"), AIMessage(content="there")]}
    )
    trace_id = trace["id"]

    try:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT source, num_spans, total_tokens, status FROM traces WHERE id = %s;",
                (trace_id,),
            )
            assert cur.fetchone() == ("eval", 2, 12, "ok")
            cur.execute("SELECT count(*) FROM spans WHERE trace_id = %s;", (trace_id,))
            assert cur.fetchone()[0] == 2
    finally:
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM traces WHERE id = %s;", (trace_id,))
