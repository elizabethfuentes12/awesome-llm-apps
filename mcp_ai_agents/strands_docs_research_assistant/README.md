# 📚 Docs Research Assistant with the Strands Harness

A chat app that answers questions about **AWS**, the **Strands Agents SDK** and the **Model Context Protocol** from their official documentation, with sources. It is built on the [Strands harness](https://strandsagents.com/docs/user-guide/harness/): `create_harness()` returns a ready-to-work Strands `Agent` with a tuned system prompt, context management, sessions, memory and Agent Skills, and this example adds only what is specific to the app.

The agent does not know the facts itself. It delegates to specialist agents, each reading one documentation source through its own MCP server, and then writes a short, sourced answer.

> **Disclosure:** the author works at AWS. Strands Agents and the AWS Knowledge MCP server are AWS projects. All three documentation servers are free and need no account, the model provider is your choice, and the specialists are declared in one list, so you can point the same pattern at any other documentation MCP server (see [Adapt it](#adapt-it)).

```
You ── Streamlit chat ──► harness agent  (create_harness, skill: grounded-research)
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   aws_specialist     strands_specialist    mcp_specialist
   (Agent.as_tool)    (Agent.as_tool)       (Agent.as_tool)
          │                   │                   │
   AWS Knowledge       Strands docs         MCP docs
   MCP server          MCP server           MCP server
   (hosted, HTTP)      (local, uvx, stdio)  (hosted, HTTP)
```

## How it differs from the other MCP examples

`multi_mcp_agent_router` also gives specialist agents their own MCP servers. This example differs in the parts that are the point of a tutorial:

- **The agent loop is the Strands harness**, not hand-written tool dispatch, and the model decides which specialists to call instead of keyword routing.
- **It needs no tokens or paid accounts.** All three MCP servers are public documentation servers.
- **Answers are grounded and cited**, with an Agent Skill that sets the answer format and a rule against filling gaps from memory.
- **It ships an evaluation suite and a check that runs offline**, which no other MCP example here has.

## What it shows

| Strands harness feature | How this example uses it |
|---|---|
| `create_harness()` | One call builds the agent: model, tuned prompt, defaults |
| Subagents as tools | Named `Agent`s passed with `Agent.as_tool()`; each call starts from a clean conversation |
| MCP with managed lifecycle | `MCPClient` goes straight into each specialist's `tools`; Strands opens and closes the connection. `tool_filters` allows only read-only documentation tools |
| Agent Skills | `skills/grounded-research/SKILL.md` holds the delegation workflow and the answer format; the agent loads it on demand |
| Sessions | `session={"id": ...}` persists the conversation; a new agent with the same id resumes it |
| Long-term memory | Optional (see the notes below); `memory={"dir": ...}` |
| Least privilege | `builtin_tools=[]` removes the shell, file and web tools, so facts can only come from the specialists |
| Model providers | `provider/name` strings (`openai/...`, `bedrock/...`), plus `effort` |
| Hooks | An `AfterToolCallEvent` hook records which specialists and MCP tools were used |
| Streaming | `agent.stream_async()` streams the answer into the page |
| Defaults left on | Context management (`auto`), prompt caching, and the `todos` and `environment` plugins |
| Observability | Set `OTEL_TRACES_EXPORTER=console` (or `otlp`) and the harness wires up tracing |
| Strands Evals | Routing, faithfulness and quality checks over real traces |

## Setup

Requirements:

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (the Strands docs MCP server is started with `uvx`)
- One model provider:
  - **OpenAI**: an API key
  - **Amazon Bedrock**: AWS credentials, a region, and access to the model

```bash
git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git
cd awesome-llm-apps/mcp_ai_agents/strands_docs_research_assistant
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run strands_docs_research_assistant.py
```

1. Pick a provider in the sidebar. For OpenAI, enter your API key (or export `OPENAI_API_KEY`). For Bedrock, export your AWS credentials and region first.
2. Ask a question or click an example. The first question that needs the Strands docs starts the local server, which `uvx` downloads once.
3. Open **How it was answered** under a reply to see which specialists ran and which MCP tools they called.

Example questions:

- How does Amazon Bedrock AgentCore Runtime isolate user sessions?
- How do I add a custom tool to a Strands agent?
- What is the difference between MCP tools and MCP resources?
- How do I deploy a Strands agent to Amazon Bedrock AgentCore Runtime?

## Check and evaluate it

```bash
python evals.py --offline
```

Runs in seconds with no network and no API key. It checks that every specialist and the skill are registered, that the shell, file and web tools are off, that the skill mentions every specialist, and that every specialist has an eval case. Use it after you change the wiring.

```bash
python evals.py --model openai/gpt-5.4-mini
```

Runs five cases in sequence: one per specialist, one that needs two of them, and one that is out of scope. Each is scored three ways:

- **routing**: did the harness agent call the right specialists? Deterministic, read from the traces.
- **faithfulness**: is the answer grounded in what the specialists and MCP tools returned? An LLM judge, with a stricter pass mark than the built-in one (0.75 instead of 0.5), because the built-in mark still passes an answer with one invented detail.
- **quality**: does the answer meet a rubric written for that case? An LLM judge.

The model you pass is also the judge. The script exits with a non-zero code when any check fails, so it can run in CI.

## Adapt it

The specialists are plain data in `strands_docs_research_assistant.py`. To read another documentation source, add one `SpecialistSpec` to `SPECIALISTS`: a name, a description that tells the harness agent when to call it, a system prompt, and a factory that returns an `MCPClient`. Allow only the read-only tools of the server with `tool_filters`: the MCP docs server, for example, also exposes a tool that sends feedback to the docs team. Then mention the new specialist in `skills/grounded-research/SKILL.md` and add a case for it in `evals.py`. `python evals.py --offline` tells you if you forgot one.

To use another model provider, type any `provider/name` the harness supports in the sidebar (`anthropic/...`, `google/...`, `ollama/...`) and export that provider's credentials first. This example was run on OpenAI and Amazon Bedrock.

## Notes

- **Answers are only as good as the documentation and its search.** The assistant is told to say so when the docs do not cover a question, and to show each value with its source when two pages disagree. Check anything important against the cited page.
- **Long-term memory is off by default in the app.** The harness distills facts from conversations into files and recalls them later. For a documentation assistant that is a risk: a distilled fact can go stale and would then be recalled without checking the docs. Memory suits user preferences better. Turn it on in the sidebar to try it, and delete `.agent/memory` to reset it.
- **This is a local demo.** The OpenAI key you type in the sidebar is set in the app process's environment, so do not host one shared instance for several users.
- **Sessions and memory are written to `.agent/`** next to the app (git-ignored).
- **MCP connections open the first time a specialist needs them** and stay open for the conversation. Starting a new conversation, changing the settings or restarting the app closes them.

## Files

- `strands_docs_research_assistant.py`: the harness agent and its specialists, and the Streamlit chat in `main()`
- `evals.py`: the Strands Evals suite and the offline checks
- `skills/grounded-research/SKILL.md`: the Agent Skill
- `requirements.txt`
