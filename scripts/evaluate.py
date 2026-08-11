#!/usr/bin/env python
"""
Offline evaluation against the golden set.

Scores a classifier on ``data/tickets/golden_set.json`` — 30 hand-labelled
Russian tickets, five per category — and prints per-class precision/recall/F1
plus macro-F1 and the confusion matrix.

Two things this is for:

1. **A real baseline number.** The rule classifier is the cheap end of the
   ladder and the fallback the whole degradation story rests on. Knowing its
   macro-F1 is what makes "rules first, LLM where they are not enough" an
   argument rather than a preference.
2. **The yardstick for distillation.** When there is enough logged and labelled
   production traffic to train a cheap model, this is the harness that says
   whether it may replace the LLM on the triage path. A distilled model has to
   beat the LLM here before it is allowed to be faster than it.

Usage::

    uv run python scripts/evaluate.py            # rules only, no network
    uv run python scripts/evaluate.py --llm      # also scores the Gemini classifier
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
# `src` is not an installed package, so a script run from anywhere needs the
# repo root on the path before it can import it.
sys.path.insert(0, str(REPO_ROOT))

from src.ml.classifier import (  # noqa: E402 - must follow the sys.path insert
    LlmTopicClassifier,
    RuleTopicClassifier,
    TopicClassifier,
)
from src.ml.pii import mask_pii  # noqa: E402 - must follow the sys.path insert


GOLDEN_SET_PATH = REPO_ROOT / "data" / "tickets" / "golden_set.json"

LABELS = [
    "billing/faq",
    "billing/payment_issue",
    "billing/refund",
    "account_recovery",
    "technical_issue",
    "other",
]


async def _score(classifier: TopicClassifier, cases: list[dict]) -> tuple[list[str], list[str], float]:
    expected: list[str] = []
    predicted: list[str] = []
    started = time.perf_counter()
    for case in cases:
        result = await classifier.classify(mask_pii(case["text"]).text)
        expected.append(case["category"])
        predicted.append(result.category)
    elapsed_ms = (time.perf_counter() - started) * 1000 / max(len(cases), 1)
    return expected, predicted, elapsed_ms


def _report(name: str, expected: list[str], predicted: list[str], elapsed_ms: float) -> None:
    print(f"\n=== {name} ===")
    print(f"mean latency: {elapsed_ms:.1f} ms/ticket")
    print(
        classification_report(
            expected,
            predicted,
            labels=LABELS,
            zero_division=0,
            digits=3,
        )
    )
    print("confusion matrix (rows = true, cols = predicted)")
    matrix = confusion_matrix(expected, predicted, labels=LABELS)
    width = max(len(label) for label in LABELS)
    print(" " * (width + 2) + " ".join(f"{i:>4}" for i in range(len(LABELS))))
    for index, (label, row) in enumerate(zip(LABELS, matrix)):
        print(f"{index} {label:<{width}} " + " ".join(f"{value:>4}" for value in row))


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm",
        action="store_true",
        help="also score the Gemini classifier (makes one API call per ticket)",
    )
    args = parser.parse_args()

    cases = json.loads(GOLDEN_SET_PATH.read_text("utf-8"))
    print(f"golden set: {len(cases)} tickets, {len(LABELS)} categories")

    expected, predicted, elapsed = await _score(RuleTopicClassifier(), cases)
    _report("rules (fallback classifier)", expected, predicted, elapsed)

    if args.llm:
        from src.dependencies import get_llm_client

        client = get_llm_client()
        if type(client).__name__ == "NullLLMClient":
            print("\nGEMINI_API_KEY is not set — skipping the LLM run.", file=sys.stderr)
            return 1
        expected, predicted, elapsed = await _score(LlmTopicClassifier(client), cases)
        _report("LLM (production classifier)", expected, predicted, elapsed)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
