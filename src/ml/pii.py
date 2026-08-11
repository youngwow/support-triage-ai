"""
PII detection and masking.

This runs *first*, before anything else touches a ticket. Everything downstream
— storage, classification, retrieval, the prompt sent to Gemini — sees only the
masked text, so "personal data must not reach the external LLM API" is enforced
by the shape of the pipeline rather than by remembering to sanitise at each call
site.

The masks are deliberately conservative in the other direction too: an order
number is not personal data, and masking it would destroy the very identifier an
operator needs. Every pattern is anchored so that bare digit runs survive.
"""

import re
from typing import NamedTuple


class MaskResult(NamedTuple):
    text: str
    #: Marker names found, each at most once, in the order the patterns are
    #: applied below — not alphabetical. Compare as a set if order would matter.
    types: tuple[str, ...]


#: Applied in order, each on the output of the previous one. Order matters:
#: e-mails are removed before digit patterns can eat parts of them, and card
#: numbers are removed before the phone pattern, which is the looser of the two.
_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "EMAIL",
        re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]*\w"),
        "[EMAIL]",
    ),
    (
        # 16 digits in 4 groups. The lookarounds stop it from biting a chunk out
        # of a longer digit run (an order id, a transaction id).
        "CARD",
        re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?!\d)"),
        "[CARD]",
    ),
    (
        # RU passport: 4-digit series + 6-digit number. Context-anchored on the
        # words around it, because "1234 567890" on its own is not identifiable.
        "PASSPORT",
        re.compile(
            r"(?:серия\s*)?\d{4}\s*,?\s*(?:номер|№)\s*\d{6}"
            r"|паспорт[а-яё]*\s*(?:рф\s*)?:?\s*\d{4}\s+\d{6}",
            re.IGNORECASE,
        ),
        "[PASSPORT]",
    ),
    (
        # RU phone. The +7/8 prefix is required — without it any 10-digit
        # identifier would be swallowed.
        "PHONE",
        re.compile(r"(?<!\d)(?:\+7|8)[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"),
        "[PHONE]",
    ),
)


def mask_pii(text: str) -> MaskResult:
    """Replace personal data with type markers and report what was found."""
    found: list[str] = []
    masked = text
    for name, pattern, replacement in _PATTERNS:
        masked, hits = pattern.subn(replacement, masked)
        if hits:
            found.append(name)
    return MaskResult(text=masked, types=tuple(found))


def contains_pii(text: str) -> bool:
    return bool(mask_pii(text).types)
