"""Streamlit chat for the AWS + Strands research assistant (see research_assistant.py).

SPDX-License-Identifier: Apache-2.0
"""
import asyncio
import os
import uuid

import streamlit as st

from research_assistant import DEFAULT_MODELS, Assistant, build_assistant

st.set_page_config(page_title="AWS + Strands Research Assistant", page_icon="🧶", layout="wide")
st.title("🧶 AWS + Strands Research Assistant")
st.caption(
    "A Strands harness agent that delegates to two specialists, each reading the official "
    "docs through its own MCP server: AWS Knowledge and Strands."
)

EXAMPLES = [
    "How does Amazon Bedrock AgentCore Runtime isolate user sessions?",
    "How do I add a custom tool to a Strands agent?",
    "How do I deploy a Strands agent to Amazon Bedrock AgentCore Runtime?",
]


def init_state() -> None:
    st.session_state.setdefault("session_id", f"chat-{uuid.uuid4().hex[:8]}")
    st.session_state.setdefault("history", [])  # [(role, text, trail_or_None)]
    st.session_state.setdefault("assistant", None)
    st.session_state.setdefault("assistant_key", None)
    st.session_state.setdefault("queued", None)


def close_assistant() -> None:
    """Close the agent (flushes memory) and its MCP connections."""
    assistant: Assistant | None = st.session_state.assistant
    if assistant is not None:
        assistant.close()
    st.session_state.assistant = None
    st.session_state.assistant_key = None


def new_conversation() -> None:
    close_assistant()
    st.session_state.session_id = f"chat-{uuid.uuid4().hex[:8]}"
    st.session_state.history = []


def queue_example(text: str) -> None:
    st.session_state.queued = text


def get_assistant(model: str, memory: bool) -> Assistant:
    """Reuse the agent while the settings are unchanged; rebuild it when they change."""
    key = (model, memory, st.session_state.session_id)
    if st.session_state.assistant_key != key:
        close_assistant()
        st.session_state.assistant = build_assistant(
            model,
            st.session_state.session_id,
            memory=memory,
            callback_handler=None,  # the UI does the streaming; keep the terminal quiet
        )
        st.session_state.assistant_key = key
    return st.session_state.assistant


async def stream_reply(assistant: Assistant, prompt: str, status, box) -> str:
    """Stream the harness agent's answer into the page, and show which tool it is using."""
    text = ""
    async for event in assistant.agent.stream_async(prompt):
        tool = event.get("current_tool_use")
        if tool and tool.get("name"):
            status.update(label=f"Using {tool['name']}...")
        if "data" in event:
            text += event["data"]
            box.markdown(text)
    return text


def show_trail(trail: dict) -> None:
    with st.expander("How it was answered"):
        st.markdown("**Agent calls:** " + (", ".join(f"`{s}`" for s in trail["specialists"]) or "none"))
        for call in trail["mcp_calls"]:
            st.markdown(f"- `{call['agent']}` used `{call['tool']}`")


init_state()

with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("Model provider", list(DEFAULT_MODELS), key="provider")
    if provider == "OpenAI":
        openai_key = st.text_input("OpenAI API key", type="password", key="openai_key")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
    else:
        st.caption("Uses your AWS credentials and the region from your environment.")
    model = st.text_input("Model (provider/name)", value=DEFAULT_MODELS[provider], key=f"model_{provider}")
    use_memory = st.checkbox(
        "Long-term memory",
        value=False,
        help="The harness distills durable facts into files under .agent/memory and recalls them "
        "in later conversations. Off by default: a distilled documentation fact can go stale "
        "and would then be recalled without checking the docs. Good for user preferences.",
    )
    st.button("New conversation", on_click=new_conversation, use_container_width=True)
    st.caption(f"Session: `{st.session_state.session_id}`")

for role, text, trail in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)
        if trail:
            show_trail(trail)

if not st.session_state.history:
    st.markdown("### Try one")
    for example in EXAMPLES:
        st.button(example, on_click=queue_example, args=(example,), key=f"ex_{example}")

prompt = st.chat_input("Ask about an AWS service or the Strands Agents SDK") or st.session_state.queued
st.session_state.queued = None

if prompt:
    st.session_state.history.append(("user", prompt, None))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if provider == "OpenAI" and not os.environ.get("OPENAI_API_KEY"):
            st.error("Enter your OpenAI API key in the sidebar.")
            st.stop()
        status = st.status("Thinking...", expanded=False)
        box = st.empty()
        try:
            assistant = get_assistant(model, use_memory)
            assistant.trail.clear()
            answer = asyncio.run(stream_reply(assistant, prompt, status, box))
            status.update(label="Done", state="complete")
            trail = {"specialists": list(assistant.trail.specialists), "mcp_calls": list(assistant.trail.mcp_calls)}
            show_trail(trail)
            st.session_state.history.append(("assistant", answer, trail))
        except Exception as exc:  # show setup problems (credentials, model name) instead of a stack trace
            status.update(label="Failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
