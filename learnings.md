# Learnings

A running list of production AI-agent topics to learn by building each one into
this project. We do one at a time: pick a topic, write a detailed plan in
`plan.md`, iterate on it, then build it stage by stage.

## Done
- **Tracing & observability** - a self-hosted trace/span collector (`tracing.py`):
  a LangChain callback handler that records per-turn latency, token usage
  (including cached), cost, and every LLM/tool step, persisted to Postgres.
  Skills: the callback system, the trace/span data model, token/cost accounting.
  (Enabler built along the way: decoupling the agent core from the transport -
  `agent.py`.)

## Now
- **Eval harness** - measure answer and behaviour quality so a prompt/model/tool
  change can be checked for regressions. Skills: eval dataset design,
  LLM-as-judge, tool-trajectory / faithfulness scoring, regression gating.

## Queued
- **Analytics over traces** - deflection rate, p50/p95 latency, cost per day and
  per user, most-failing tool, token trends (the metrics CS-agent products sell
  on). Builds on the traces we already store.
- **Guardrails & grounding** - input/output filtering (off-topic, jailbreak, PII)
  and verifying answers are grounded in retrieved context, with citations.
- **Human handoff / escalation** - detect low-confidence, frustration, or
  out-of-scope requests and escalate: create a ticket, hand off, notify.

## Maybe / smaller
- **Feedback capture** - 👍/👎 per turn stored against its trace; bridges into evals.
- **Streaming responses & latency** - token streaming, time-to-first-token.
- **Conversation persistence** - a LangGraph checkpointer so sessions resume.
- **Trace UI / OpenTelemetry** - emit OTel spans and view them in Phoenix/Langfuse.
