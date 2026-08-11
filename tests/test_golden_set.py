"""The golden set, and the rule classifier measured against it.

``data/tickets/golden_set.json`` is the yardstick: 30 hand-labelled Russian
tickets, five per category. ``scripts/evaluate.py`` prints the full report; this
file pins the properties that must not silently change, so a rule edit that
looks harmless cannot quietly cost accuracy.

Scored exactly the way the script scores: mask PII first, then classify, then
let sklearn do the arithmetic. The rule classifier only — the LLM path needs a
key and a network, and neither belongs in a unit suite.

Two of these tests are not about accuracy at all, they are about the safety
argument the fallback rests on:

* the rules are *high-precision*: when they commit to a category they are right,
  which is what makes them usable as a risk floor;
* every classification they produce lands below ``min_topic_confidence``, so a
  degraded ticket can never be auto-sent no matter what the rules decided.
"""

import json
from collections import Counter
from typing import Any, get_args

import pytest

from sklearn.metrics import f1_score, precision_score

from src.config import get_settings
from src.ml.classifier import RuleTopicClassifier
from src.ml.pii import mask_pii
from src.models.domain import Category, RiskLevel


#: Label order is the one ``scripts/evaluate.py`` reports in.
CATEGORIES: list[str] = list(get_args(Category))
RISK_LEVELS: set[str] = set(get_args(RiskLevel))

EXPECTED_TICKETS = 30
EXPECTED_PER_CATEGORY = 5

#: Measured today: **0.598**. Recall is what caps it — the rules answer "other"
#: for 18 of 30 tickets — while precision on the categories they do commit to is
#: 1.0. The floor sits below the measurement so an honest refactor passes and a
#: real regression does not; the assertion prints the number it actually got, so
#: whoever moves it has the new figure in front of them.
MACRO_F1_FLOOR = 0.55


@pytest.fixture(scope="module")
def golden_set() -> list[dict[str, Any]]:
    return json.loads(get_settings().golden_set_path.read_text(encoding="utf-8"))


@pytest.fixture
async def rule_scores(
    golden_set: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[float]]:
    """``(expected, predicted, confidence)`` over the whole set, rules only."""
    classifier = RuleTopicClassifier()
    expected: list[str] = []
    predicted: list[str] = []
    confidences: list[float] = []
    for case in golden_set:
        result = await classifier.classify(mask_pii(case["text"]).text)
        expected.append(case["category"])
        predicted.append(result.category)
        confidences.append(result.confidence)
    return expected, predicted, confidences


# --- the dataset itself --------------------------------------------------


def test_the_golden_set_holds_thirty_tickets(golden_set: list[dict[str, Any]]) -> None:
    assert len(golden_set) == EXPECTED_TICKETS


def test_every_ticket_id_is_unique(golden_set: list[dict[str, Any]]) -> None:
    ids = [case["id"] for case in golden_set]

    duplicates = [ticket_id for ticket_id, count in Counter(ids).items() if count > 1]

    assert duplicates == []
    assert len(ids) == EXPECTED_TICKETS


def test_the_taxonomy_is_covered_evenly(golden_set: list[dict[str, Any]]) -> None:
    """Macro-F1 over an unbalanced set would not mean what the report says it means."""
    counts = Counter(case["category"] for case in golden_set)

    assert dict(counts) == {category: EXPECTED_PER_CATEGORY for category in CATEGORIES}


def test_every_category_label_is_in_the_taxonomy(
    golden_set: list[dict[str, Any]],
) -> None:
    unknown = sorted({case["category"] for case in golden_set} - set(CATEGORIES))

    assert unknown == [], f"labels outside src.models.domain.Category: {unknown}"


def test_every_risk_label_is_a_valid_risk_level(
    golden_set: list[dict[str, Any]],
) -> None:
    unknown = sorted({case["risk"] for case in golden_set} - RISK_LEVELS)

    assert unknown == [], f"labels outside src.models.domain.RiskLevel: {unknown}"


def test_no_ticket_body_is_empty(golden_set: list[dict[str, Any]]) -> None:
    blank = [case["id"] for case in golden_set if not case["text"].strip()]

    assert blank == []


# --- the rule classifier's score ----------------------------------------


async def test_rule_macro_f1_does_not_regress(
    rule_scores: tuple[list[str], list[str], list[float]],
) -> None:
    expected, predicted, _ = rule_scores

    macro_f1 = f1_score(
        expected, predicted, labels=CATEGORIES, average="macro", zero_division=0
    )

    assert macro_f1 >= MACRO_F1_FLOOR, (
        f"rule macro-F1 fell to {macro_f1:.3f}, floor is {MACRO_F1_FLOOR}"
    )


async def test_rules_are_never_wrong_about_a_category_they_commit_to(
    rule_scores: tuple[list[str], list[str], list[float]],
) -> None:
    """Precision 1.0 on every predicted category is what justifies the risk floor.

    ``other`` is excluded: it is the rules saying "no idea", not a commitment.
    """
    expected, predicted, _ = rule_scores
    per_label = dict(
        zip(
            CATEGORIES,
            precision_score(
                expected, predicted, labels=CATEGORIES, average=None, zero_division=0
            ),
        )
    )
    committed = sorted({label for label in predicted if label != "other"})

    imprecise = {
        label: round(float(per_label[label]), 3)
        for label in committed
        if per_label[label] < 1.0
    }

    assert committed, "the rules committed to no category at all"
    assert imprecise == {}, f"rules mislabelled tickets: precision {imprecise}"


async def test_no_rule_classification_could_ever_be_auto_sent(
    rule_scores: tuple[list[str], list[str], list[float]],
) -> None:
    """The fallback is a safety net: every answer it gives must reach a human."""
    threshold = get_settings().min_topic_confidence
    _, _, confidences = rule_scores

    highest = max(confidences)

    assert highest < threshold, (
        f"a rule classification scored {highest}, at or above the "
        f"auto-send threshold {threshold}"
    )
