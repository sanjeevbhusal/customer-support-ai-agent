"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st

from vector_store import search_faqs, search_inventory

st.set_page_config(page_title="Customer Support Agent", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "type": "assistant",
            "content": "Hi, I'm the flower shop chatbot. How can i help? ",
        }
    ]

messages = st.session_state.messages


col1, col2, col3 = st.columns(3)


# Clear chat button
with col1:
    if st.button("Clear Chat"):
        st.session_state.messages = []

    collection = st.radio("Which collection?", ["faqs", "inventory"])


# Chat history
with col2:
    user_input = st.chat_input("Ask Question...")

    if user_input:
        if collection == "faqs":
            data = search_faqs(user_input)
        else:
            data = search_inventory(user_input)

        st.session_state.messages.append({"type": "assistant", "content": str(data)})
        st.session_state.messages.append({"type": "user", "content": user_input})

    for message in reversed(messages):
        message_box = st.chat_message(message["type"])
        message_box.markdown(message["content"])

# Display message history
with col3:
    st.text(st.session_state.messages)
