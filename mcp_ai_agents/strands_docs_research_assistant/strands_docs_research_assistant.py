"""Docs research assistant built on the Strands harness, with a Streamlit chat.

`create_harness()` returns a plain Strands `Agent` that already ships with a tuned system
prompt, context management, sessions, long-term memory, Agent Skills and a todo tracker.
This file only adds what is specific to the app:

- specialist agents, each wired to its own MCP server and handed to the harness as tools
  with `Agent.as_tool()`:
    aws_specialist     -> AWS Knowledge MCP server (hosted, no credentials)
    strands_specialist -> Strands docs MCP server (local, started with uvx)
    mcp_specialist     -> Model Context Protocol docs MCP server (hosted, no credentials)
- a `grounded-research` Agent Skill that tells the harness how to delegate and cite
- a session id, so a conversation survives restarts, and a memory folder
- no shell, file or web tools: the answer has to come from the specialists

The agent code lives at the top and the Streamlit chat in `main()`, so `evals.py` can
import the agent without starting the UI.

SPDX-License-Identifier: Apache-2.0
"""
import asyncio
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import streamlit as st
from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.hooks import AfterToolCallEvent
from strands.tools.mcp import MCPClient
from strands_harness import create_harness
from strands_harness.models import resolve_model

HERE = Path(__file__).parent
SKILLS_DIR = HERE / "skills"
STATE_DIR = HERE / ".agent"  # sessions and memory; git-ignored

DEFAULT_MODELS = {
    "OpenAI": "openai/gpt-5.4-mini",
    "Amazon Bedrock": "bedrock/global.anthropic.claude-sonnet-5",
}

INSTRUCTIONS = (
    "You are a documentation research assistant for AWS, the Strands Agents SDK and the "
    "Model Context Protocol (MCP). You have no tools of your own for facts: specialist "
    "agents read the official documentation for you. For every question, load the "
    "grounded-research skill and follow it."
)

# --------------------------------------------------------------------------------------
# Specialists: one documentation source each. Add a source by adding a SpecialistSpec.
# --------------------------------------------------------------------------------------

GROUNDING_RULES = (
    "Grounding rules:\n"
    "- Every fact, name and parameter in your answer must appear in the tool output. "
    "Never add details from memory, even if you believe they are true.\n"
    "- Include the URL of every page you rely on, and a short code snippet when the "
    "documentation has one.\n"
    "- If two pages disagree, give each page's value and its URL instead of choosing one.\n"
    "- If the documentation does not cover the question, say so plainly, but only after you "
    "have tried at least three differently worded searches.\n"
    "- Keep the answer under 200 words.\n"
)

AWS_SPECIALIST_PROMPT = (
    "You are an AWS documentation specialist. Use your tools to search the official AWS "
    "documentation, read pages and check regional availability.\n"
    + GROUNDING_RULES
    + "For \"which regions\" questions, and any question that needs a complete list or table, "
    "follow these steps in order:\n"
    "1. Search the documentation.\n"
    "2. Call the read tool on the best developer-guide page from the results. This step is "
    "required: search results are excerpts and can cut a table or list short.\n"
    "3. Answer from the full page. If it has a separate row or column per feature, report "
    "the one the question is about and say which row you used.\n"
    "The availability tool checks at most 10 regions per call. If you use it, say exactly "
    "which regions you checked, and never present a partial check as the full list."
)

STRANDS_SPECIALIST_PROMPT = (
    "You are a Strands Agents documentation specialist. Use your tools to search the "
    "Strands user guide and API reference and read the relevant sections.\n"
    + GROUNDING_RULES
    + "How to search: search_docs ranks mostly by page title, so use short, title-like queries "
    "(for example \"custom tools\", \"deploy AgentCore\", \"session manager\") instead of full "
    "sentences. Prefer user-guide pages over API reference pages, and ignore pages about other "
    "Strands products such as Box or Shell unless the question is about them. If the first "
    "results are not the how-to guide for the question, search again with other words."
)

MCP_SPECIALIST_PROMPT = (
    "You are a Model Context Protocol documentation specialist. Use your tools to search the "
    "official MCP documentation (specification, concepts, SDK guides) and read the relevant "
    "pages.\n"
    + GROUNDING_RULES
    + "How to search: start with the search tool, using the protocol's own vocabulary (for "
    "example \"tools\", \"resources\", \"transports\"). If a result is too short to answer, read "
    "the full page with the documentation filesystem tool. Prefer the latest specification "
    "version and say which version a statement comes from."
)


@dataclass(frozen=True)
class SpecialistSpec:
    """One documentation source: a specialist agent plus the MCP server it reads.

    To add a source, append a SpecialistSpec to SPECIALISTS, mention it in
    skills/grounded-research/SKILL.md so the harness knows when to call it, and add a
    case for it in evals.py.
    """

    name: str  # becomes the tool name the harness agent calls
    description: str  # tells the harness agent when to call this specialist
    system_prompt: str
    mcp_client: Callable[[], MCPClient]  # a factory: every assistant opens its own connection


def _aws_knowledge_client() -> MCPClient:
    # Hosted by AWS, no credentials. Only the documentation tools are allowed.
    return MCPClient(
        url="https://knowledge-mcp.global.api.aws",
        tool_filters={
            "allowed": [
                "aws___search_documentation",
                "aws___read_documentation",
                "aws___list_regions",
                "aws___get_regional_availability",
            ]
        },
    )


def _strands_docs_client() -> MCPClient:
    # Started locally with uvx over stdio.
    return MCPClient(lambda: stdio_client(StdioServerParameters(command="uvx", args=["strands-agents-mcp-server"])))


def _mcp_docs_client() -> MCPClient:
    # Hosted by the MCP docs site, no credentials. The server also exposes `submit_feedback`,
    # which sends a report to the docs team, so only the two read-only tools are allowed.
    return MCPClient(
        url="https://modelcontextprotocol.io/mcp",
        tool_filters={
            "allowed": ["search_model_context_protocol", "query_docs_filesystem_model_context_protocol"]
        },
    )


SPECIALISTS = (
    SpecialistSpec(
        name="aws_specialist",
        description=(
            "Reads the official AWS documentation. Ask it about AWS services, features, "
            "limits, and which regions support a service. Send one focused, self-contained question."
        ),
        system_prompt=AWS_SPECIALIST_PROMPT,
        mcp_client=_aws_knowledge_client,
    ),
    SpecialistSpec(
        name="strands_specialist",
        description=(
            "Reads the Strands Agents documentation. Ask it how to build, configure, deploy "
            "and extend agents with the Strands SDK. Send one focused, self-contained question."
        ),
        system_prompt=STRANDS_SPECIALIST_PROMPT,
        mcp_client=_strands_docs_client,
    ),
    SpecialistSpec(
        name="mcp_specialist",
        description=(
            "Reads the official Model Context Protocol documentation. Ask it about the MCP "
            "specification and concepts such as tools, resources, prompts and transports. "
            "Send one focused, self-contained question."
        ),
        system_prompt=MCP_SPECIALIST_PROMPT,
        mcp_client=_mcp_docs_client,
    ),
)
SPECIALIST_NAMES = frozenset(spec.name for spec in SPECIALISTS)

# --------------------------------------------------------------------------------------
# The harness agent
# --------------------------------------------------------------------------------------


@dataclass
class Trail:
    """What happened during a question: which specialists ran and which MCP tools they used."""

    specialists: list[str] = field(default_factory=list)
    mcp_calls: list[dict[str, Any]] = field(default_factory=list)

    def clear(self) -> None:
        self.specialists.clear()
        self.mcp_calls.clear()


def _watch_tools(agent: Agent, record) -> None:
    """Call `record(tool_name, tool_input)` after every tool the agent runs (a Strands hook)."""

    def on_tool_call(event: AfterToolCallEvent) -> None:
        record(event.tool_use["name"], event.tool_use.get("input", {}))

    agent.add_hook(on_tool_call, AfterToolCallEvent)


def _specialist(spec: SpecialistSpec, model: Any, trail: Trail, trace_attributes: Optional[dict[str, Any]]) -> Agent:
    """A specialist is a normal Agent. Passing the MCPClient in `tools` lets Strands
    open and close the MCP connection for us."""
    agent = Agent(
        name=spec.name,
        description=spec.description,
        system_prompt=spec.system_prompt,
        tools=[spec.mcp_client()],
        model=model,
        callback_handler=None,
        trace_attributes=trace_attributes,
    )
    _watch_tools(agent, lambda tool, args: trail.mcp_calls.append({"agent": spec.name, "tool": tool, "input": args}))
    return agent


@dataclass
class Assistant:
    """The harness agent plus everything that has to be closed with it.

    Use it as a context manager. Closing runs `shutdown()` on the harness agent, which
    flushes pending long-term memory, and `cleanup()` on each specialist, which closes
    its MCP connection (and stops the local Strands docs server).
    """

    agent: Agent
    trail: Trail
    specialists: list[Agent] = field(default_factory=list)

    def __enter__(self) -> "Assistant":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.agent.shutdown()
        finally:  # always close the MCP connections, even if the shutdown above fails
            for specialist in self.specialists:
                specialist.cleanup()


def build_assistant(
    model: str,
    session_id: str,
    *,
    effort: str = "low",
    state_dir: Path = STATE_DIR,
    memory: bool = True,
    **agent_kwargs: Any,
) -> Assistant:
    """Build the harness agent.

    model       "provider/name", for example "openai/gpt-5.4-mini" or a Bedrock model id.
    session_id  reuse an id to resume that conversation, even after a restart.
    memory      long-term memory across conversations (files under `state_dir`).
    agent_kwargs go straight to the underlying Agent (the harness forwards unknown keywords).
    """
    trail = Trail()
    # Specialists share the session/trace attributes, so evals see their spans too.
    trace_attributes = agent_kwargs.get("trace_attributes")

    # The specialists only read documentation, so they run on the same provider without reasoning.
    specialist_model = resolve_model(model, default=DEFAULT_MODELS["Amazon Bedrock"], effort="off")
    specialists = [_specialist(spec, specialist_model, trail, trace_attributes) for spec in SPECIALISTS]

    agent = create_harness(
        model=model,
        effort=effort,
        instructions=INSTRUCTIONS,
        tools=[specialist.as_tool() for specialist in specialists],
        builtin_tools=[],  # no shell, file or web tools: facts come from the specialists
        skills=str(SKILLS_DIR),
        session={"id": session_id, "dir": str(state_dir / "sessions")},
        memory={"dir": str(state_dir / "memory")} if memory else False,
        **agent_kwargs,
    )

    # The harness agent is a plain Agent, so hooks work on it too.
    _watch_tools(
        agent,
        lambda tool, args: trail.specialists.append(tool) if tool in SPECIALIST_NAMES else None,
    )
    return Assistant(agent=agent, trail=trail, specialists=specialists)


# --------------------------------------------------------------------------------------
# Streamlit chat
# --------------------------------------------------------------------------------------

EXAMPLES = [
    "How does Amazon Bedrock AgentCore Runtime isolate user sessions?",
    "How do I add a custom tool to a Strands agent?",
    "What is the difference between MCP tools and MCP resources?",
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
    st.session_state.assistant = None
    st.session_state.assistant_key = None
    if assistant is not None:
        assistant.close()


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


def main() -> None:
    st.set_page_config(page_title="Docs Research Assistant", page_icon="📚", layout="wide")
    st.title("📚 Docs Research Assistant")
    st.caption(
        "A Strands harness agent that delegates to specialists, each reading official docs "
        "through its own MCP server: AWS, Strands Agents and the Model Context Protocol."
    )
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
        st.caption(
            "The harness also accepts anthropic/..., google/..., ollama/... and other providers. "
            "Export that provider's credentials before starting the app."
        )
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

    prompt = st.chat_input("Ask about AWS, Strands Agents or MCP") or st.session_state.queued
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


if __name__ == "__main__":
    main()
