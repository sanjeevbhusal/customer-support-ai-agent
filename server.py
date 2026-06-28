"""
# My first app
Here's our first attempt at using data to create a table:
"""

import json
import logging

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool

from vector_store import search_faqs, search_inventory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


st.set_page_config(page_title="Customer Support Agent", layout="wide")

SYSTEM_PROMPT = """You are the customer support assistant for Bloom & Petal, an \
online flower shop that sells bouquets and arrangements and delivers them locally.

Your role is to help customers with:
- Questions about the business (delivery, payment, returns, hours, policies)
- Finding flowers and arrangements that fit their needs, budget, and occasion
- Checking product availability and pricing
- Helping them decide what to order

Tools available to you:
- `search_business_faqs`: use this for any question about business policies, \
delivery, payments, returns, or how the shop operates. Do not guess at policy.
- `search_product_inventory`: use this to find products, check availability, and \
get current prices. Always rely on this for what is in stock and how much it costs.

Guidelines:
- Always ground answers in the information returned by the tools. Never invent \
products, prices, availability, or policies.
- If the tools return nothing relevant, say you don't have that information or \
that the item isn't available, and suggest alternatives when you can.
- Prices are in USD.
- Be warm, concise, and helpful. Confirm key order details (item, quantity, \
delivery address, timing) before treating an order as ready.
- If a request is outside what a flower shop assistant can do, politely say so."""

GREETING = "Hi! I'm the Bloom & Petal assistant 🌸 — I can help you find the perfect arrangement, check what's in stock, or answer questions about delivery and orders. How can I help?"

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        AIMessage(content=GREETING),
    ]


@tool(description="Search business FAQ's")
def search_business_faqs(query: str):
    """Fetch top 3 FAQ results for user query"""
    return search_faqs(query)


@tool(description="Search product inventory")
def search_product_inventory(query: str):
    """Fetch top 5 products for user query"""
    return search_inventory(query)


tools = [search_business_faqs, search_product_inventory]
tools_by_name = {tool.name: tool for tool in tools}

# Augment the LLM with tools
model = init_chat_model("gpt-4o-mini", temperature=0).bind_tools(tools)


def invoke_agent():
    end = False
    while not end:
        response = model.invoke(st.session_state.messages)
        st.session_state.messages.append(response)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                logger.info(f"Calling tool {tool_call['name']} ")
                tool = tools_by_name[tool_call["name"]]
                observation = tool.invoke(tool_call["args"])
                if not isinstance(observation, str):
                    observation = json.dumps(observation, default=str)
                tool_message = ToolMessage(
                    content=observation, tool_call_id=tool_call["id"]
                )
                st.session_state.messages.append(tool_message)
        else:
            end = True

    return st.session_state.messages


# Clear chat button lives in the sidebar, out of the conversation flow
with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            AIMessage(content=GREETING),
        ]
        st.rerun()


# chat_input is at the top level of the app, so Streamlit pins it to the
# bottom of the page. Read it first, then render history above it.
user_input = st.chat_input("Ask Question...")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    invoke_agent()


# Render messages in natural order (oldest -> newest) so the newest sits
# just above the pinned input.
messages = st.session_state.messages
for index, message in enumerate(messages):
    if isinstance(message, (SystemMessage, ToolMessage)):
        continue

    if isinstance(message, AIMessage) and message.tool_calls:
        continue

    message_box = st.chat_message(message.type)
    message_box.markdown(message.content)

    # Look BACKWARD for the tool-call turn that produced this message:
    # ... AIMessage(tool_calls) -> ToolMessage(s) -> this final AIMessage
    tools_called_message: AIMessage | None = None
    tool_messages: list[ToolMessage] = []

    for i in range(index - 1, -1, -1):
        prev = messages[i]
        if isinstance(prev, ToolMessage):
            tool_messages.append(prev)
        elif isinstance(prev, AIMessage) and prev.tool_calls:
            tools_called_message = prev
            break
        else:
            break

    if tools_called_message:
        for tool_call in tools_called_message.tool_calls:
            with message_box.expander(f"🔧 {tool_call['name']}"):
                st.caption("Arguments")
                st.json(tool_call["args"])

                tool_message = None
                for tm in tool_messages:
                    if tm.tool_call_id == tool_call["id"]:
                        tool_message = tm

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
