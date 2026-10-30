"""Evaluate the research assistant with Strands Evals.

Three checks on every case:
- routing:      did the harness agent call the right specialists? (deterministic, read from the traces)
- faithfulness: is the answer grounded in what the specialists and MCP tools returned? (LLM judge, strict)
- quality:      does the answer meet a rubric? (LLM judge)

Run it with `python evals.py --model openai/gpt-5.4-mini`.

SPDX-License-Identifier: Apache-2.0
"""
import argparse
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

from research_assistant import DEFAULT_MODELS, build_assistant

SPECIALISTS = {"aws_specialist", "strands_specialist"}

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
            if isinstance(span, ToolExecutionSpan) and span.tool_call.name in SPECIALISTS
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODELS["OpenAI"], help="provider/name, also used as the judge")
    args = parser.parse_args()

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
