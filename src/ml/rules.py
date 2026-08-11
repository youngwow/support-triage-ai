"""
Deterministic rules: the part of the problem that does not need a model.

Two jobs:

* **Risk floor.** A rule that fires sets a minimum risk level the model is not
  allowed to talk the system out of. Money disputes, legal threats and suspected
  account takeover are recognised by their vocabulary reliably enough that
  spending an LLM call — or accepting an LLM's disagreement — would be a
  regression, not an improvement.
* **Prompt-injection screen.** A flagged ticket is never handed to the
  generator, so the attack does not reach the provider at all.

The same table doubles as the fallback classifier's brain (see
``RuleTopicClassifier``), which is what keeps the system useful when the LLM API
is down.
"""

import re
from dataclasses import dataclass
from typing import NamedTuple, Optional

from src.models.domain import Category, RiskLevel, max_risk


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    risk: RiskLevel
    #: Set when firing this rule is also strong evidence of the topic.
    category: Optional[Category] = None


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


#: Order is priority order: the first rule with a category wins the category
#: vote in the fallback classifier.
RULES: tuple[Rule, ...] = (
    Rule(
        name="legal_threat",
        pattern=_rx(
            r"\b(в\s+суд|подам\s+в\s+суд|полици|прокуратур|роспотребнадзор"
            r"|центробанк|заявление\s+в|жалоб\w*\s+в)\b"
        ),
        risk="high",
    ),
    Rule(
        name="refund_demand",
        pattern=_rx(
            r"(верн\w+\s+(мои\s+)?деньги|возврат\w*\s+(средств|денег)|рефанд"
            r"|chargeback|оспор\w+\s+(платеж|списан)|двойн\w+\s+списан"
            r"|списал\w*\s+дважды|сняли\s+.{0,20}(\d+|дважды|два|три|четыре)\s*раз)"
        ),
        risk="high",
        category="billing/refund",
    ),
    Rule(
        name="account_takeover",
        pattern=_rx(
            r"(взлом\w*|украл\w+\s+(аккаунт|доступ)|мошенник|фишинг"
            r"|заход\w+\s+не\s+я|чужой\s+вход)"
        ),
        risk="high",
        category="account_recovery",
    ),
    Rule(
        name="abuse",
        pattern=_rx(r"\b(уроды|идиот\w*|дебил\w*|ворьё|воры|обманщик\w*)\b"),
        risk="high",
    ),
    Rule(
        name="account_recovery",
        pattern=_rx(
            r"(восстанов\w+\s+(аккаунт|доступ|пароль)|не\s+могу\s+(войти|зайти)"
            r"|забыл\s+пароль|потерял\s+доступ|сброс\w*\s+парол)"
        ),
        risk="medium",
        category="account_recovery",
    ),
    Rule(
        name="payment_issue",
        pattern=_rx(
            r"(оплат\w+\s+.{0,30}(не\s+работает|не\s+пришл|не\s+активиров)"
            r"|списал\w+\s+но|подписк\w+\s+не\s+активн|статус\w*\s+заказа"
            r"|платеж\w*\s+(вис|не\s+прош)"
            # "купил подписку ... но ничего не работает" — paid-but-no-access is
            # a payment problem, not a technical one, and the two rules would
            # otherwise fight over it.
            r"|(куп\w+|оплат\w+|приобрел\w*)\s+(подписк|премиум|тариф|доступ))"
        ),
        risk="low",
        category="billing/payment_issue",
    ),
    Rule(
        name="technical_issue",
        pattern=_rx(
            r"(50[0234]\b|bad\s+gateway|не\s+открывается|сайт\s+(лежит|не\s+работает)"
            r"|ошибк\w+\s+сервера|недоступен)"
        ),
        risk="low",
        category="technical_issue",
    ),
    Rule(
        name="billing_faq",
        pattern=_rx(
            r"(как\s+(мне\s+)?(отвязать|привязать|сменить|поменять|изменить)"
            r"|способ\w*\s+оплаты|привяз\w+\s+карт)"
        ),
        risk="low",
        category="billing/faq",
    ),
)


#: Instruction-shaped text inside a ticket body. Matching does not sanitise the
#: text — it marks the ticket so the draft path refuses to generate for it.
INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    _rx(r"(игнорир\w+|забудь|отмени)\s+(все\s+)?(предыдущ\w+\s+)?(инструкц|правил|указан)"),
    _rx(r"(ignore|disregard|forget)\s+(all\s+)?(previous\s+|prior\s+)?(instructions|rules|prompt)"),
    _rx(r"(системн\w+\s+промпт|system\s+prompt|покажи\s+свой\s+промпт|выведи\s+инструкц)"),
    _rx(r"(ты\s+теперь|теперь\s+ты)\s+\w+|you\s+are\s+now\b|act\s+as\s+(a|an)\b"),
    _rx(r"(ответь|напиши|скажи),?\s+что\s+(деньги|средства)\s+\w*\s*(возвращен|вернул)"),
)


class RuleVerdict(NamedTuple):
    hits: tuple[str, ...]
    risk: RiskLevel
    category: Optional[Category]
    injection: bool


def evaluate(text: str) -> RuleVerdict:
    """Run every rule and the injection screen over a ticket body."""
    hits: list[str] = []
    risk: RiskLevel = "low"
    category: Optional[Category] = None

    for rule in RULES:
        if rule.pattern.search(text):
            hits.append(rule.name)
            risk = max_risk(risk, rule.risk)
            if category is None and rule.category is not None:
                category = rule.category

    injection = any(pattern.search(text) for pattern in INJECTION_PATTERNS)
    if injection:
        hits.append("prompt_injection")
        risk = max_risk(risk, "high")

    return RuleVerdict(
        hits=tuple(hits),
        risk=risk,
        category=category,
        injection=injection,
    )
