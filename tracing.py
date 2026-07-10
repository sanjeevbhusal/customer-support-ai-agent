"""Self-hosted tracing: a callback handler that records what an agent run did.

Attach one collector per turn via `config["callbacks"]`; LangChain fires the
hooks for every LLM and tool call automatically.
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from langchain.messages import AIMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGeneration, LLMResult
from psycopg.types.json import Json

from db import pool

logger = logging.getLogger(__name__)

# USD per token. Approximate list prices - update when they change.
MODEL_PRICING = {
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,
        "cached_input": 0.075 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
}


_EMPTY_USAGE = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}


def _ai_message(response: LLMResult) -> AIMessage | None:
    """The AIMessage from a chat LLM result, or None for a non-chat result."""
    generation = response.generations[0][0]
    if isinstance(generation, ChatGeneration) and isinstance(generation.message, AIMessage):
        return generation.message
    return None


def _parse_llm_result(response: LLMResult) -> tuple[str | None, dict[str, int]]:
    """Return `(model_name, token_usage)` from the chat message.

    Uses LangChain's provider-normalized fields (`usage_metadata`,
    `response_metadata`), not the raw provider `llm_output`, so the shape is
    stable across providers.
    """
    message = _ai_message(response)
    if message is None:
        return None, dict(_EMPTY_USAGE)

    model = message.response_metadata.get("model_name")

    metadata = message.usage_metadata
    if metadata:
        input_token_details = (
            metadata["input_token_details"]
            if "input_token_details" in metadata
            else None
        )
        cached_input_tokens = (
            input_token_details["cache_read"]
            if input_token_details and "cache_read" in input_token_details
            else 0
        )

        token_usage = {
            "input_tokens": metadata["input_tokens"],
            "output_tokens": metadata["output_tokens"],
            "cached_input_tokens": cached_input_tokens,
        }
    else:
        token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
        }

    return model, token_usage


def _llm_tool_calls(response: LLMResult) -> list[str]:
    """Names of the tools the LLM asked to call."""
    message = _ai_message(response)
    return [tc["name"] for tc in message.tool_calls] if message else []


_MAX_STR = 4096


def _truncate(value: Any) -> Any:
    """Cap large strings so a span row can't blow up; keep structure otherwise.

    Recurses into dicts/lists; anything not JSON-friendly is stringified. There is
    no secret material in tool args/results here, so this is a size guard, not
    redaction.
    """
    if isinstance(value, str):
        return value if len(value) <= _MAX_STR else value[:_MAX_STR] + "…[truncated]"
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v) for v in value]
    if isinstance(value, (int, float)) or value is None:
        return value
    return _truncate(str(value))


def _compute_cost_usd(model: str, input_tokens: int, cached: int, output: int) -> float:
    """Cost of a turn, billing cached input tokens at the discounted rate.

    Model names come back dated (e.g. "gpt-4o-mini-2024-07-18"), so match by
    prefix. A model with no configured pricing is a real gap - log it loudly and
    record the cost as 0 rather than silently guessing.
    """
    rates = MODEL_PRICING.get(model) or next(
        (r for k, r in MODEL_PRICING.items() if model.startswith(k)), None
    )
    if rates is None:
        logger.error("no pricing configured for model %r; recording cost_usd as 0", model)
        return 0.0
    uncached = input_tokens - cached
    return uncached * rates["input"] + cached * rates["cached_input"] + output * rates["output"]


def _user_and_reply_text(result: dict[str, Any]) -> tuple[str | None, str | None]:
    """The customer's message and the assistant's final reply from a run result."""
    messages = result.get("messages", [])
    user = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    reply = next(
        (m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
        None,
    )
    return user, reply


def _persist_trace(trace: dict[str, Any], spans: list[dict[str, Any]]) -> None:
    """Write a trace and its spans in one transaction."""
    with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO traces (
                id, source, thread_id, user_id, model, input_text, output_text,
                num_spans, prompt_tokens, completion_tokens, total_tokens,
                cached_input_tokens, cost_usd, latency_ms, status, error
            ) VALUES (
                %(id)s, %(source)s, %(thread_id)s, %(user_id)s, %(model)s,
                %(input_text)s, %(output_text)s, %(num_spans)s, %(prompt_tokens)s,
                %(completion_tokens)s, %(total_tokens)s, %(cached_input_tokens)s,
                %(cost_usd)s, %(latency_ms)s, %(status)s, %(error)s
            );
            """,
            trace,
        )
        cur.executemany(
            """
            INSERT INTO spans (
                id, trace_id, name, type, started_at, duration_ms, attributes, error
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """,
            [
                (
                    span["id"],
                    trace["id"],
                    span["name"],
                    span["type"],
                    span["started_at"],
                    span["duration_ms"],
                    Json(span["attributes"]),
                    span["error"],
                )
                for span in spans
            ],
        )


class TraceCollector(BaseCallbackHandler):
    """Accumulates spans for a single agent run.

    Usage:
        collector = TraceCollector(source="app", thread_id=tid, user_id=uid)
        agent.invoke(state, config={"callbacks": [collector]})
    """

    def __init__(
        self,
        source: str,
        thread_id: str | None = None,
        user_id: int | None = None,
    ) -> None:
        self.source = source
        self.thread_id = thread_id
        self.user_id = user_id
        self.model: str | None = None
        self.start_perf = time.perf_counter()

        self.spans: list[dict[str, Any]] = []
        self._open: dict[str, dict[str, Any]] = {}  # run_id -> partial span

        # Running token totals across the trace's LLM spans.
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_input_tokens = 0

    # --- LLM spans ---

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._start_span(run_id, name="llm", span_type="llm")

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        model, usage = _parse_llm_result(response)
        if model:
            self.model = model
        self.input_tokens += usage["input_tokens"]
        self.output_tokens += usage["output_tokens"]
        self.cached_input_tokens += usage["cached_input_tokens"]

        self._end_span(
            run_id,
            attributes={**usage, "tool_calls": _llm_tool_calls(response)},
        )

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._end_span(run_id, attributes={}, error=str(error))

    # --- tool spans ---

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # `inputs` is the model-supplied args; the injected user_id is not among them.
        self._start_span(
            run_id,
            name=serialized["name"],
            span_type="tool",
            attributes={"args": _truncate(inputs)},
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._end_span(run_id, attributes={"result": _truncate(output)})

    def on_tool_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._end_span(run_id, attributes={}, error=str(error))

    # --- finish ---

    def finish(
        self,
        result: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        """Roll the spans up into a trace and persist it.

        Call once after `agent.invoke` returns (pass `result`) or raises (pass
        `error`). Persistence is best-effort: a DB failure is logged, never
        raised, so tracing can't break the turn.
        """
        input_text, reply_text = _user_and_reply_text(result) if result else (None, None)
        # No model means no LLM ran (e.g. an early error) - genuinely no cost,
        # not a missing-pricing error.
        cost_usd = (
            _compute_cost_usd(
                self.model, self.input_tokens, self.cached_input_tokens, self.output_tokens
            )
            if self.model
            else 0.0
        )
        trace = {
            "id": str(uuid.uuid4()),
            "source": self.source,
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "model": self.model,
            "input_text": input_text,
            "output_text": reply_text,
            "num_spans": len(self.spans),
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cost_usd": cost_usd,
            "latency_ms": int((time.perf_counter() - self.start_perf) * 1000),
            "status": "error" if error or any(s["error"] for s in self.spans) else "ok",
            "error": str(error) if error else None,
        }
        try:
            _persist_trace(trace, self.spans)
        except Exception:
            logger.exception("failed to persist trace")
        return trace

    # --- span bookkeeping ---

    def _start_span(
        self,
        run_id: UUID,
        *,
        name: str,
        span_type: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self._open[str(run_id)] = {
            "name": name,
            "type": span_type,
            "started_at": datetime.now(timezone.utc),
            "start_perf": time.perf_counter(),
            "attributes": attributes or {},
        }

    def _end_span(
        self, run_id: UUID, *, attributes: dict[str, Any], error: str | None = None
    ) -> None:
        span = self._open.pop(str(run_id), None)
        if span is None:
            return
        self.spans.append(
            {
                "id": str(uuid.uuid4()),
                "name": span["name"],
                "type": span["type"],
                "started_at": span["started_at"],
                "duration_ms": int((time.perf_counter() - span["start_perf"]) * 1000),
                "attributes": {**span["attributes"], **attributes},
                "error": error,
            }
        )
