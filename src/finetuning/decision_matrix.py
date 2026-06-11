# src/finetuning/decision_matrix.py

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel


FailureType = Literal[
    "missing_knowledge",
    "stale_knowledge",
    "bad_format",
    "bad_style",
    "tool_not_called",
    "tool_args_wrong",
    "hallucination_without_evidence",
    "preference_misalignment",
    "high_cost",
    "high_latency",
]


class OptimizationDecision(BaseModel):
    failure_type: FailureType
    primary_solution: str
    secondary_solution: str
    should_finetune: bool
    reasoning: str


def decide_optimization(failure_type: FailureType) -> OptimizationDecision:
    mapping = {
        "missing_knowledge": OptimizationDecision(
            failure_type=failure_type,
            primary_solution="RAG or Tool",
            secondary_solution="Improve retrieval / data connector",
            should_finetune=False,
            reasoning="The model lacks external or private knowledge; parameters should not be used as a database.",
        ),
        "stale_knowledge": OptimizationDecision(
            failure_type=failure_type,
            primary_solution="RAG or Tool",
            secondary_solution="Freshness-aware retrieval",
            should_finetune=False,
            reasoning="Knowledge changes over time; updating the knowledge source is better than retraining.",
        ),
        "bad_format": OptimizationDecision(
            failure_type=failure_type,
            primary_solution="Structured Output / JSON Schema",
            secondary_solution="Retry / repair / SFT if persistent",
            should_finetune=False,
            reasoning="Try schema-constrained output first; SFT is useful only if format remains unstable at scale.",
        ),
        "bad_style": OptimizationDecision(
            failure_type=failure_type,
            primary_solution="Prompt examples",
            secondary_solution="SFT",
            should_finetune=True,
            reasoning="Stable domain style across many examples is a good SFT use case.",
        ),
        "tool_not_called": OptimizationDecision(
            failure_type=failure_type,
            primary_solution="Tool schema / description / routing eval",
            secondary_solution="Tool-use SFT",
            should_finetune=False,
            reasoning="Most tool-use failures should first be fixed with schema, prompt, and trajectory eval.",
        ),
        "tool_args_wrong": OptimizationDecision(
            failure_type=failure_type,
            primary_solution="Argument schema validation / repair",
            secondary_solution="Tool-use SFT",
            should_finetune=False,
            reasoning="Validate and repair tool arguments before training.",
        ),
        "hallucination_without_evidence": OptimizationDecision(
            failure_type=failure_type,
            primary_solution="Answerability check + citation validation",
            secondary_solution="DPO on safe vs unsafe answers",
            should_finetune=True,
            reasoning="If hallucination persists after grounding and validation, preference tuning can teach safer behavior.",
        ),
        "preference_misalignment": OptimizationDecision(
            failure_type=failure_type,
            primary_solution="DPO / preference optimization",
            secondary_solution="SFT with better examples",
            should_finetune=True,
            reasoning="Preference data directly fits DPO: chosen vs rejected responses.",
        ),
        "high_cost": OptimizationDecision(
            failure_type=failure_type,
            primary_solution="Use smaller model / distillation",
            secondary_solution="Caching / routing",
            should_finetune=True,
            reasoning="Distillation or SFT on a smaller model can reduce cost if task is stable.",
        ),
        "high_latency": OptimizationDecision(
            failure_type=failure_type,
            primary_solution="Serving optimization / smaller model",
            secondary_solution="Distillation",
            should_finetune=True,
            reasoning="Fine-tuning a smaller model may reduce latency if quality remains acceptable.",
        ),
    }

    return mapping[failure_type]


def main():
    for failure_type in [
        "missing_knowledge",
        "bad_format",
        "bad_style",
        "preference_misalignment",
        "high_cost",
    ]:
        decision = decide_optimization(failure_type)
        print(decision.model_dump_json(indent=2))


if __name__ == "__main__":
    main()