"""The support agent graph"""

import json
from typing import Literal, cast

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph

from auth import resolve_tool_args
from tools import tools, tools_by_name

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
- `get_current_user`: use this to see who is currently signed in, for example to \
greet them by name or confirm whose account an order will go under.
- `place_order`: use this to place an order for the signed-in customer. Requires \
the product id, the quantity, and a delivery location.
- `get_orders`: use this to look up the signed-in customer's existing orders.

Accounts and ordering:
- Customers sign in using the login form in the sidebar, not through chat. Never \
ask for, collect, or handle an email, password, or user id in the conversation.
- The signed-in customer's identity is handled automatically. You do not pass a \
user id to any tool; if you need to know who you are talking to, call \
`get_current_user`.
- A customer must be signed in before you can place an order or look up their \
orders. If they want to do either but are not signed in (a tool will tell you so \
with an error, or `get_current_user` will report no one is signed in), politely \
ask them to log in using the login form in the sidebar on the left.
- Only bring up signing in when the customer actually wants to place an order or \
check their orders. While they are just browsing, asking questions, or comparing \
products, do not mention accounts - let them inquire freely.
- To place an order you need the product (use `search_product_inventory` to get \
its `id`), the quantity, and a delivery location (e.g. "Shankhamul, Kathmandu"). \
Confirm these details with the customer, then call `place_order`.
- If a tool returns an `error` (e.g. not signed in, or not enough stock), explain \
it to the customer and guide them to the right next step (logging in via the \
sidebar, or choosing a different product or quantity).

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


def initial_messages() -> list[BaseMessage]:
    """A fresh conversation: the system prompt and the greeting."""
    return [SystemMessage(content=SYSTEM_PROMPT), AIMessage(content=GREETING)]


# Augment the LLM with tools
model = init_chat_model("gpt-4o-mini", temperature=0).bind_tools(tools)


def llm_call(state: MessagesState) -> MessagesState:
    """LLM decides whether to call a tool or not"""
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    last_message = cast(AIMessage, state["messages"][-1])

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


def tool_node(state: MessagesState, config: RunnableConfig):
    """Run the tool calls, injecting the signed-in customer's id from config."""
    auth_user = (config.get("configurable") or {}).get("auth_user")
    last_message = cast(AIMessage, state["messages"][-1])

    result = []
    for tool_call in last_message.tool_calls:
        # error is set when an auth tool is called with no one signed in.
        args, error = resolve_tool_args(tool_call["name"], tool_call["args"], auth_user)
        if error is not None:
            observation = error
        else:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(args)

        # ToolMessage.content must be a string. A list/dict gets interpreted as
        # OpenAI "content blocks" (each needing a valid `type`), which fails.
        if not isinstance(observation, str):
            observation = json.dumps(observation, default=str)
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    return {"messages": result}


def _build_agent():
    graph = StateGraph(MessagesState)
    graph.add_node(llm_call)
    graph.add_node(tool_node)
    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    graph.add_edge("tool_node", "llm_call")
    return graph.compile()


agent = _build_agent()
