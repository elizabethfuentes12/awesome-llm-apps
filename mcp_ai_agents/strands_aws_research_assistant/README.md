# 🧶 AWS + Strands Research Assistant

A chat app that answers questions about AWS services and the Strands Agents SDK **from the official documentation**, with sources. It is built on the [Strands harness](https://strandsagents.com/docs/user-guide/harness/): `create_harness()` returns a ready-to-work Strands `Agent` with a tuned system prompt, context management, sessions, memory and Agent Skills, and this example adds only what is specific to the app.

The agent does not know the facts itself. It delegates to two specialist agents, each reading one documentation source through its own MCP server, and then writes a short, sourced answer.

```
You ── Streamlit chat ──► harness agent  (create_harness, skill: grounded-research)
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
       aws_specialist                  strands_specialist
       (Agent.as_tool)                 (Agent.as_tool)
              │                                │
   AWS Knowledge MCP server        Strands docs MCP server
   (hosted by AWS, HTTP)           (local, uvx, stdio)
```

## What it shows

| Strands harness feature | How this example uses it | Where |
|---|---|---|
| `create_harness()` | One call builds the agent: model, tuned prompt, defaults | `research_assistant.py` |
| Subagents as tools | Two named `Agent`s passed with `Agent.as_tool()`; each call starts from a clean conversation | `research_assistant.py` |
| MCP with managed lifecycle | `MCPClient` goes straight into each specialist's `tools`; Strands opens and closes the connection. `tool_filters` limits the AWS server to its documentation tools | `research_assistant.py` |
| Agent Skills | `skills/grounded-research/SKILL.md` holds the delegation workflow and the answer format; the agent loads it on demand | `skills/` |
| Sessions | `session={"id": ...}` persists the conversation; a new agent with the same id resumes it | `research_assistant.py` |
| Long-term memory | Optional (see the note below); `memory={"dir": ...}` | `research_assistant.py` |
| Least privilege | `builtin_tools=[]` removes the shell, file and web tools, so facts can only come from the specialists | `research_assistant.py` |
| Model providers | `provider/name` strings (`openai/...`, `bedrock/...`), plus `effort` | `research_assistant.py` |
| Hooks | An `AfterToolCallEvent` hook records which specialists and MCP tools were used | `research_assistant.py` |
| Streaming | `agent.stream_async()` streams the answer into the page | `strands_aws_research_assistant.py` |
| Defaults left on | Context management (`auto`), prompt caching, and the `todos` and `environment` plugins | |
| Observability | Set `OTEL_TRACES_EXPORTER=console` (or `otlp`) and the harness wires up tracing | |
| Strands Evals | Routing, faithfulness and quality checks over real traces | `evals.py` |

## Setup

Requirements:

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (the Strands docs MCP server is started with `uvx`)
- One model provider:
  - **OpenAI**: an API key
  - **Amazon Bedrock**: AWS credentials, a region, and access to the model

```bash
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git
cd awesome-llm-apps/mcp_ai_agents/strands_aws_research_assistant
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run strands_aws_research_assistant.py
```

1. Pick a provider in the sidebar. For OpenAI, enter your API key (or export `OPENAI_API_KEY`). For Bedrock, export your AWS credentials and region first.
2. Ask a question or click an example. The first question starts the local Strands docs server, which `uvx` downloads once.
3. Open **How it was answered** under a reply to see which specialists ran and which MCP tools they called.

Example questions:

- How does Amazon Bedrock AgentCore Runtime isolate user sessions?
- How do I add a custom tool to a Strands agent?
- How do I deploy a Strands agent to Amazon Bedrock AgentCore Runtime?

## Evaluate it

```bash
python evals.py --model openai/gpt-5.4-mini
```

Four cases run in sequence: an AWS question, a Strands question, one that needs both specialists, and one that is out of scope. Each is scored three ways:

- **routing**: did the harness agent call the right specialists? Deterministic, read from the traces.
- **faithfulness**: is the answer grounded in what the specialists and MCP tools returned? An LLM judge, with a stricter pass mark than the built-in one (0.75 instead of 0.5), because the built-in mark still passes an answer with one invented detail.
- **quality**: does the answer meet a rubric written for that case? An LLM judge.

The model you pass is also the judge. The script exits with a non-zero code when any check fails, so it can run in CI.

## Notes

- **Answers are only as good as the documentation and its search.** The assistant is told to say so when the docs do not cover a question, and to show each value with its source when two pages disagree. Check anything important against the cited page.
- **Long-term memory is off by default in the app.** The harness distills facts from conversations into files and recalls them later. For a documentation assistant that is a risk: a distilled fact can go stale and would then be recalled without checking the docs. Memory suits user preferences better. Turn it on in the sidebar to try it, and delete `.agent/memory` to reset it.
- **Sessions and memory are written to `.agent/`** next to the app (git-ignored).
- **MCP connections open the first time a specialist needs them** and stay open for the conversation. Starting a new conversation, changing the settings or restarting the app closes them.

## Files

- `research_assistant.py`: builds the harness agent and its two specialists
- `strands_aws_research_assistant.py`: the Streamlit chat
- `evals.py`: the Strands Evals suite
- `skills/grounded-research/SKILL.md`: the Agent Skill
- `requirements.txt`
