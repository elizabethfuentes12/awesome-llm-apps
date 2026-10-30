---
name: grounded-research
description: Use for any question about AWS services, the Strands Agents SDK or the Model Context Protocol. Gives the workflow for delegating to the specialist agents and writing a short, sourced answer that never goes beyond what they returned.
---

# Grounded research

Answer questions about AWS, the Strands Agents SDK and the Model Context Protocol (MCP)
using only what the specialist
agents return. The specialists read the official documentation; you do not.

## Workflow

1. Split the question by topic. A question can touch more than one.
2. Call `aws_specialist` for AWS services, features, limits and regional availability.
   Call `strands_specialist` for building, configuring and deploying agents with Strands.
   Call `mcp_specialist` for the MCP specification and concepts such as tools, resources,
   prompts and transports.
   Call every specialist the question needs.
3. Each specialist starts with no memory of this conversation, so write each question so
   it can be understood on its own: name the service, the SDK feature and what you need.
4. If a specialist's answer is incomplete or says the documentation does not cover it,
   say so in your answer. Do not fill the gap from memory.

## Answer format

- **Answer**: two to five sentences that respond to the question directly.
- **Details**: short bullets or a small table, only with facts the specialists returned.
- **Sources**: every URL the specialists gave you, one per line.
- **Caveats**: anything they could not confirm, such as which regions were checked.

## Rules

- Lists (regions, instance types, quotas) are copied exactly as returned. Never add an
  item you believe is probably there, and never drop one.
- A partial check is not a full list. Say which items were checked.
- If the specialists report different values for the same fact, show each value with its
  source and say that the documentation is inconsistent. Do not pick one.
- Write every AWS region with its name and its code, for example `Europe (Ireland) - eu-west-1`,
  so a reader can check the list against the documentation.
- If the question is about none of these topics, say it is out of scope and call no one.
