"""Evaluate the docs research assistant with Strands Evals.

Three checks on every case:
- routing:      did the harness agent call the right specialists? (deterministic, read from the traces)
- faithfulness: is the answer grounded in what the specialists and MCP tools returned? (LLM judge, strict)
- quality:      does the answer meet a rubric? (LLM judge)

Run it with `python evals.py --model openai/gpt-5.4-mini`.
`python evals.py --offline` runs only the wiring checks: no network, no API keys.

SPDX-License-Identifier: Apache-2.0
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator, FaithfulnessEvaluator, OutputEvaluator
from strands_evals.mappers import StrandsInMemorySessionMapper
from strands_evals.telemetry import StrandsEvalsTelemetry
from strands_evals.types.evaluation import EvaluationData, EvaluationOutput
from strands_evals.types.trace import ToolExecutionSpan
from strands_harness.models import resolve_model

from strands_docs_research_assistant import DEFAULT_MODELS, SKILLS_DIR, SPECIALIST_NAMES, build_assistant

CASES = [
    Case[str, str](
        name="aws-concept",
        input="How does Amazon Bedrock AgentCore Runtime isolate user sessions?",
        expected_trajectory=["aws_specialist"],
        metadata={"rubric": "Pass if it explains how sessions are isolated and cites an AWS documentation URL."},
    ),
    Case[str, str](
        name="strands-howto",
        input="How do I add a custom tool to a Strands agent?",
        expected_trajectory=["strands_specialist"],
        expected_output="Decorate a Python function with @tool, give it type hints and a docstring, and pass it in tools=[...] to the Agent.",
        metadata={"rubric": "Pass if it names the @tool decorator and shows how to pass the tool to an Agent."},
    ),
    Case[str, str](
        name="mcp-concept",
        input="What is the difference between MCP tools and MCP resources?",
        expected_trajectory=["mcp_specialist"],
        metadata={"rubric": "Pass if it explains both concepts, says who controls each, and cites a modelcontextprotocol.io URL."},
    ),
    Case[str, str](
        name="mixed",
        input="How do I deploy a Strands agent to Amazon Bedrock AgentCore Runtime?",
        expected_trajectory=["aws_specialist", "strands_specialist"],
        metadata={"rubric": "Pass if it gives deployment steps and cites both Strands and AWS documentation URLs."},
    ),
    Case[str, str](
        name="out-of-scope",
        input="What is the weather in Seattle today?",
        expected_trajectory=[],
        metadata={"rubric": "Pass if it says the question is out of scope or that it cannot answer, and gives no weather data."},
    ),
]


class RoutingEvaluator(Evaluator[str, str]):
    """Passes when the specialists called match `expected_trajectory` (order does not matter)."""

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        session = evaluation_case.actual_trajectory
        called = {
            span.tool_call.name
            for trace in getattr(session, "traces", [])
            for span in trace.spans
            if isinstance(span, ToolExecutionSpan) and span.tool_call.name in SPECIALIST_NAMES
        }
        expected = set(evaluation_case.expected_trajectory or [])
        ok = called == expected
        return [
            EvaluationOutput(
                score=1.0 if ok else 0.0,
                test_pass=ok,
                reason=f"called {sorted(called) or 'no specialist'}, expected {sorted(expected) or 'no specialist'}",
            )
        ]

    async def evaluate_async(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        return self.evaluate(evaluation_case)


class StrictFaithfulness(FaithfulnessEvaluator):
    """FaithfulnessEvaluator that passes only at `min_score` or above.

    The built-in evaluator passes at 0.5 ("Neutral/Mixed"), which still lets an answer with an
    invented detail through. A research assistant should not make things up, so require
    "Generally Yes" (0.75) or better.
    """

    def __init__(self, model, min_score: float = 0.75, name: str = "faithfulness"):
        super().__init__(model=model, name=name)
        self.min_score = min_score

    def _apply_threshold(self, outputs: list[EvaluationOutput]) -> list[EvaluationOutput]:
        for output in outputs:
            output.test_pass = output.score >= self.min_score
        return outputs

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        return self._apply_threshold(super().evaluate(evaluation_case))

    async def evaluate_async(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        return self._apply_threshold(await super().evaluate_async(evaluation_case))


class RubricEvaluator(OutputEvaluator):
    """OutputEvaluator whose rubric comes from each case, so one evaluator covers every case."""

    def __init__(self, judge_model, name: str = "quality"):
        super().__init__(rubric="(set per case)", model=judge_model, name=name)

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        self.rubric = (evaluation_case.metadata or {}).get("rubric", self.rubric)
        return super().evaluate(evaluation_case)

    async def evaluate_async(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        self.rubric = (evaluation_case.metadata or {}).get("rubric", self.rubric)
        return await super().evaluate_async(evaluation_case)


# Built-in harness tools that must stay off: the answers have to come from the specialists.
FORBIDDEN_TOOLS = {"shell", "read", "write", "edit", "web_fetch", "web_search", "programmatic_tool_caller", "subagent"}


def offline_checks() -> int:
    """Wiring checks that need no network and no API key. Returns the number of problems."""
    # Building the model object needs a key to exist; no request is made.
    os.environ.setdefault("OPENAI_API_KEY", "offline-check")
    problems: list[str] = []

    with build_assistant(
        DEFAULT_MODELS["OpenAI"], "offline-check", state_dir=Path(tempfile.mkdtemp()), memory=False, callback_handler=None
    ) as assistant:
        tools = set(assistant.agent.tool_names)
    problems += [f"specialist tool missing: {name}" for name in sorted(SPECIALIST_NAMES - tools)]
    if "skills" not in tools:
        problems.append("the Agent Skills tool is not registered (is skills/ found?)")
    problems += [f"built-in tool should be off: {name}" for name in sorted(tools & FORBIDDEN_TOOLS)]

    skill = (SKILLS_DIR / "grounded-research" / "SKILL.md").read_text()
    problems += [f"SKILL.md does not mention {name}" for name in sorted(SPECIALIST_NAMES) if name not in skill]

    expected = {name for case in CASES for name in (case.expected_trajectory or [])}
    problems += [f"eval case expects an unknown specialist: {name}" for name in sorted(expected - SPECIALIST_NAMES)]
    problems += [f"no eval case covers {name}" for name in sorted(SPECIALIST_NAMES - expected)]

    for problem in problems:
        print(f"FAIL  {problem}")
    print(f"offline checks: {len(problems)} problem(s); tools registered: {sorted(tools)}")
    return len(problems)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODELS["OpenAI"], help="provider/name, also used as the judge")
    parser.add_argument("--offline", action="store_true", help="only run the wiring checks (no network, no API keys)")
    args = parser.parse_args()
    if args.offline:
        return 1 if offline_checks() else 0

    judge = resolve_model(args.model, default=DEFAULT_MODELS["Amazon Bedrock"], effort="off")
    telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
    state_dir = Path(tempfile.mkdtemp(prefix="research-assistant-evals-"))

    def task(case: Case) -> dict:
        telemetry.in_memory_exporter.clear()
        ids = {"gen_ai.conversation.id": case.session_id, "session.id": case.session_id}
        # memory=False keeps every case independent of the others.
        with build_assistant(
            args.model, case.session_id, state_dir=state_dir, memory=False, trace_attributes=ids, callback_handler=None
        ) as assistant:
            answer = str(assistant.agent(case.input))
        session = StrandsInMemorySessionMapper().map_to_session(
            telemetry.in_memory_exporter.get_finished_spans(), session_id=case.session_id
        )
        return {"output": answer, "trajectory": session}

    experiment = Experiment[str, str](
        cases=CASES,
        evaluators=[
            RoutingEvaluator(name="routing"),
            StrictFaithfulness(judge),
            RubricEvaluator(judge),
        ],
    )
    report = experiment.run_evaluations(task)  # sequential: the span exporter is shared
    report.display()
    return 0 if all(report.test_passes) else 1


if __name__ == "__main__":
    sys.exit(main())
