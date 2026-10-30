"""AWS + Strands research assistant, built on the Strands harness.

`create_harness()` returns a plain Strands `Agent` that already ships with a tuned system
prompt, context management, sessions, long-term memory, Agent Skills and a todo tracker.
This module only adds what is specific to this app:

- two specialist agents, each wired to its own MCP server and handed to the harness as
  tools with `Agent.as_tool()`:
    aws_specialist     -> AWS Knowledge MCP server (hosted by AWS, no credentials)
    strands_specialist -> Strands docs MCP server (local, started with uvx)
- a `grounded-research` Agent Skill that tells the harness how to delegate and cite
- a session id, so a conversation survives restarts, and a memory folder
- no shell, file or web tools: the answer has to come from the specialists

SPDX-License-Identifier: Apache-2.0
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.hooks import AfterToolCallEvent
from strands.tools.mcp import MCPClient
from strands_harness import create_harness
from strands_harness.models import resolve_model

HERE = Path(__file__).parent
SKILLS_DIR = HERE / "skills"
STATE_DIR = HERE / ".agent"  # sessions and memory; git-ignored

# AWS Knowledge MCP server: hosted by AWS, no credentials needed.
AWS_KNOWLEDGE_MCP_URL = "https://knowledge-mcp.global.api.aws"
AWS_DOC_TOOLS = [
    "aws___search_documentation",
    "aws___read_documentation",
    "aws___list_regions",
    "aws___get_regional_availability",
]

DEFAULT_MODELS = {
    "OpenAI": "openai/gpt-5.4-mini",
    "Amazon Bedrock": "bedrock/global.anthropic.claude-sonnet-5",
}

INSTRUCTIONS = (
    "You are an AWS and Strands Agents research assistant. You answer questions about "
    "AWS services and about building agents with the Strands Agents SDK. You have no "
    "tools of your own for facts: two specialist agents read the official documentation "
    "for you. For every question, load the grounded-research skill and follow it."
)

AWS_SPECIALIST_PROMPT = (
    "You are an AWS documentation specialist. Use your tools to search the official AWS "
    "documentation, read pages and check regional availability.\n"
    "Grounding rules:\n"
    "- Every fact, name and region code in your answer must appear in the tool output. "
    "Never add items from memory, even if you believe they are true.\n"
    "- Include the URL of every page you rely on.\n"
    "- If the documentation does not cover the question, say so plainly.\n"
    "- If two pages disagree, give each page's value and its URL instead of choosing one.\n"
    "- Keep the answer under 200 words.\n"
    "For \"which regions\" questions, and any question that needs a complete list or table, "
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
    "Grounding rules:\n"
    "- Every fact, class name and parameter in your answer must appear in the tool output. "
    "Never add API details from memory.\n"
    "- Include the URL of every page you rely on, and a short code snippet when the "
    "documentation has one.\n"
    "- If the documentation does not cover the question, say so plainly, but only after you "
    "have tried at least three differently worded searches.\n"
    "- Keep the answer under 200 words.\n"
    "How to search: search_docs ranks mostly by page title, so use short, title-like queries "
    "(for example \"custom tools\", \"deploy AgentCore\", \"session manager\") instead of full "
    "sentences. Prefer user-guide pages over API reference pages, and ignore pages about other "
    "Strands products such as Box or Shell unless the question is about them. If the first "
    "results are not the how-to guide for the question, search again with other words."
)


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


def _specialist(
    name: str,
    description: str,
    system_prompt: str,
    mcp_client: MCPClient,
    model: Any,
    trail: Trail,
    trace_attributes: Optional[dict[str, Any]] = None,
) -> Agent:
    """A specialist is a normal Agent. Passing the MCPClient in `tools` lets Strands
    open and close the MCP connection for us."""
    agent = Agent(
        name=name,
        description=description,
        system_prompt=system_prompt,
        tools=[mcp_client],
        model=model,
        callback_handler=None,
        trace_attributes=trace_attributes,
    )
    _watch_tools(agent, lambda tool, args: trail.mcp_calls.append({"agent": name, "tool": tool, "input": args}))
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
        self.agent.shutdown()
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

    aws = _specialist(
        name="aws_specialist",
        description=(
            "Reads the official AWS documentation. Ask it about AWS services, features, "
            "limits, and which regions support a service. Send one focused, self-contained question."
        ),
        system_prompt=AWS_SPECIALIST_PROMPT,
        mcp_client=MCPClient(url=AWS_KNOWLEDGE_MCP_URL, tool_filters={"allowed": AWS_DOC_TOOLS}),
        model=specialist_model,
        trail=trail,
        trace_attributes=trace_attributes,
    )
    strands = _specialist(
        name="strands_specialist",
        description=(
            "Reads the Strands Agents documentation. Ask it how to build, configure, deploy "
            "and extend agents with the Strands SDK. Send one focused, self-contained question."
        ),
        system_prompt=STRANDS_SPECIALIST_PROMPT,
        mcp_client=MCPClient(
            lambda: stdio_client(StdioServerParameters(command="uvx", args=["strands-agents-mcp-server"]))
        ),
        model=specialist_model,
        trail=trail,
        trace_attributes=trace_attributes,
    )

    agent = create_harness(
        model=model,
        effort=effort,
        instructions=INSTRUCTIONS,
        tools=[aws.as_tool(), strands.as_tool()],
        builtin_tools=[],  # no shell, file or web tools: facts come from the specialists
        skills=str(SKILLS_DIR),
        session={"id": session_id, "dir": str(state_dir / "sessions")},
        memory={"dir": str(state_dir / "memory")} if memory else False,
        **agent_kwargs,
    )

    # The harness agent is a plain Agent, so hooks work on it too.
    _watch_tools(
        agent,
        lambda tool, args: trail.specialists.append(tool) if tool in ("aws_specialist", "strands_specialist") else None,
    )
    return Assistant(agent=agent, trail=trail, specialists=[aws, strands])
