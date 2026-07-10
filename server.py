"""
# My first app
Here's our first attempt at using data to create a table:
"""

import json
import logging
import os
import uuid

import streamlit as st
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from streamlit_local_storage import LocalStorage

from agent import agent, initial_messages
from auth import (
    authenticate_user,
    clear_persisted_login,
    load_persisted_user,
    persist_login,
)
from tracing import TraceCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Customer Support Agent", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = initial_messages()

# The authenticated customer (or None).
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

# One trace thread id per browser session, so a conversation's turns group together.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# Browser localStorage keeps the login across full page refreshes, via the auth
# helpers. The component mounts a frontend widget and busy-waits for the browser,
# so under pytest (AppTest, no browser) it would hang - skip it there, which makes
# the auth helpers treat storage as empty.
local_storage = None if "PYTEST_CURRENT_TEST" in os.environ else LocalStorage()

# A localStorage write only reaches the browser when the script run finishes and
# flushes to the frontend; calling st.rerun() right after a write discards it. So
# login/logout record a pending write and we apply it here, at the top of the next
# run, which then completes normally.
pending_op = st.session_state.pop("pending_storage_op", None)
if pending_op == "persist" and st.session_state.auth_user is not None:
    persist_login(local_storage, st.session_state.auth_user)
elif pending_op == "clear":
    clear_persisted_login(local_storage)

# On a fresh page load (new session) auth_user is None; restore it from the token.
# Skip after an explicit logout so we don't immediately restore the cleared login.
if st.session_state.auth_user is None and not st.session_state.get("logged_out"):
    restored_user = load_persisted_user(local_storage)
    if restored_user is not None:
        st.session_state.auth_user = restored_user


# Clear chat button lives in the sidebar, out of the conversation flow
with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = initial_messages()
        st.rerun()

    st.divider()

    if st.session_state.auth_user is None:
        st.subheader("Log in")
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            user = authenticate_user(email, password)
            if user is None:
                st.error("Invalid email or password.")
            else:
                st.session_state.auth_user = user
                st.session_state.logged_out = False
                st.session_state.pending_storage_op = "persist"
                st.rerun()
    else:
        user = st.session_state.auth_user
        st.subheader("Account")
        st.markdown(f"**{user['first_name']} {user['last_name']}**  \n{user['email']}")
        if st.button("Log out"):
            # Reset the conversation too, so it doesn't carry into the next login.
            st.session_state.auth_user = None
            st.session_state.logged_out = True
            st.session_state.pending_storage_op = "clear"
            st.session_state.messages = initial_messages()
            st.rerun()

    # Metrics for the most recent turn, from its trace.
    trace = st.session_state.get("last_trace")
    if trace is not None:
        st.divider()
        with st.expander("⏱ Last turn"):
            st.caption(f"{trace['model']} · {trace['num_spans']} steps · {trace['status']}")
            latency, tokens, cost = st.columns(3)
            latency.metric("Latency", f"{trace['latency_ms']} ms")
            tokens.metric("Tokens", trace["total_tokens"])
            cost.metric("Cost", f"${trace['cost_usd']:.4f}")


# chat_input is at the top level of the app, so Streamlit pins it to the
# bottom of the page. Read it first, then render history above it.
user_input = st.chat_input("Ask Question...")

if user_input:
    # session_state is the only state that survives a Streamlit rerun, so it
    # is our source of truth for the whole conversation.
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Pass the full history in, plus the signed-in customer via config so the
    # tools can act on their behalf. The collector records a trace of the turn.
    auth_user = st.session_state.auth_user
    collector = TraceCollector(
        source="app",
        thread_id=st.session_state.thread_id,
        user_id=auth_user["id"] if auth_user else None,
    )
    try:
        result = agent.invoke(
            {"messages": st.session_state.messages},
            config={
                "configurable": {"auth_user": auth_user},
                "callbacks": [collector],
            },
        )
        st.session_state.last_trace = collector.finish(result=result)
    except Exception as exc:
        collector.finish(error=exc)
        raise

    # MessagesState's add_messages reducer appends each node's output, so
    # result["messages"] is the complete updated list; write it back to persist.
    st.session_state.messages = result["messages"]

    # Rerun so the sidebar (rendered above, before this turn ran) picks up the
    # new messages and the just-recorded last_trace metrics.
    st.rerun()


# Render messages in natural order (oldest -> newest) so the newest sits
# just above the pinned input.
messages = st.session_state.messages

# A ToolMessage holds a tool's result; index them by tool_call_id so each tool
# call can show its result inline, regardless of where the ToolMessage sits.
tool_results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}


def render_tool_call(container, tool_call):
    with container.expander(f"🔧 {tool_call['name']}"):
        st.caption("Arguments")
        st.json(tool_call["args"])

        tool_message = tool_results.get(tool_call["id"])
        if tool_message is not None:
            st.caption("Result")
            content = tool_message.content
            if isinstance(content, str):
                try:
                    st.json(json.loads(content))
                except json.JSONDecodeError:
                    st.markdown(content)
            else:
                st.json(content)


# Walk the conversation in order. All the assistant activity for one turn (its
# tool calls and text, possibly across several AIMessages) goes in a single chat
# message box; a HumanMessage starts a fresh one.
ai_box = None
for message in messages:
    if isinstance(message, HumanMessage):
        ai_box = None
        st.chat_message("human").markdown(message.content)
    elif isinstance(message, AIMessage):
        if ai_box is None:
            ai_box = st.chat_message("ai")
        if message.content:
            ai_box.markdown(message.content)
        for tool_call in message.tool_calls:
            render_tool_call(ai_box, tool_call)

    # SystemMessage / ToolMessage aren't rendered on their own
