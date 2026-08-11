"""PII masking — the compliance surface of the synchronous path.

Two failure modes matter and they pull in opposite directions: personal data
leaking through to storage/the provider, and an identifier an operator needs
(an order number, a transaction id) being destroyed by an over-eager pattern.
Both are covered here.
"""

import pytest

from src.ml.pii import contains_pii, mask_pii


#: T-1004 from ``data/tickets/historical_tickets.json``, verbatim. The passport
#: is written the way a real user wrote it, which is why this exact string is
#: the multi-type case rather than a synthetic one.
T_1004 = (
    "Помогите восстановить аккаунт. "
    "Мой логин test@test.com, паспорт серия 1234 номер 567890."
)
T_1004_MASKED = "Помогите восстановить аккаунт. Мой логин [EMAIL], паспорт [PASSPORT]."


@pytest.mark.parametrize(
    ("raw", "expected_text", "expected_types"),
    [
        pytest.param(
            "Пишите на ivan.petrov+support@example.co.uk, жду ответа",
            "Пишите на [EMAIL], жду ответа",
            ("EMAIL",),
            id="email-with-plus-and-multilevel-domain",
        ),
        pytest.param(
            "мой телефон +7 916 123-45-67, перезвоните",
            "мой телефон [PHONE], перезвоните",
            ("PHONE",),
            id="phone-plus7-spaced",
        ),
        pytest.param(
            "телефон 8 (916) 123-45-67",
            "телефон [PHONE]",
            ("PHONE",),
            id="phone-8-with-brackets",
        ),
        pytest.param(
            "звоните 89161234567",
            "звоните [PHONE]",
            ("PHONE",),
            id="phone-8-unformatted",
        ),
        pytest.param(
            "карта 4276 1234 5678 9012 списала лишнее",
            "карта [CARD] списала лишнее",
            ("CARD",),
            id="card-in-four-groups",
        ),
        pytest.param(
            "карта 4276123456789012",
            "карта [CARD]",
            ("CARD",),
            id="card-16-digits-unspaced",
        ),
        pytest.param(
            "паспорт серия 1234 номер 567890",
            "паспорт [PASSPORT]",
            ("PASSPORT",),
            id="passport-seriya-nomer-t1004-form",
        ),
        pytest.param(
            "паспорт 4509 123456",
            "[PASSPORT]",
            ("PASSPORT",),
            id="passport-word-then-digits",
        ),
        pytest.param(
            "паспорт РФ: 4509 123456",
            "[PASSPORT]",
            ("PASSPORT",),
            id="passport-rf-with-colon",
        ),
    ],
)
def test_masks_personal_data(raw: str, expected_text: str, expected_types: tuple[str, ...]) -> None:
    result = mask_pii(raw)

    assert result.text == expected_text
    assert result.types == expected_types


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("Я купил подписку полчаса назад, заказ 77712, но ничего не работает!", id="t1002-order-77712"),
        pytest.param("заказ 77712", id="bare-order-number"),
        pytest.param("номер заказа 1002003004 не обновляется", id="bare-10-digit-id"),
        pytest.param("телефон 9161234567", id="10-digits-without-the-7-or-8-prefix"),
        pytest.param("id 123456789012345", id="15-digit-run-is-not-a-card"),
        pytest.param("транзакция 1234567890123456789", id="19-digit-run-is-not-a-card"),
    ],
)
def test_order_and_transaction_numbers_are_not_masked(raw: str) -> None:
    """An order id is not personal data — masking it destroys what the operator needs.

    Regression guard for the anchors in ``_PATTERNS``: every digit-based pattern
    must refuse to bite a chunk out of a bare identifier.
    """
    result = mask_pii(raw)

    assert result.text == raw
    assert result.types == ()


def test_clean_text_reports_no_pii_types() -> None:
    raw = "Здравствуйте! Подскажите, пожалуйста, когда обновится статус? Спасибо за помощь."

    result = mask_pii(raw)

    assert result.text == raw
    assert result.types == ()


def test_masks_every_pii_type_in_one_ticket() -> None:
    """T-1004 carries an e-mail and a passport in a single sentence."""
    result = mask_pii(T_1004)

    assert result.text == T_1004_MASKED
    assert result.types == ("EMAIL", "PASSPORT")


def test_masks_a_card_and_an_email_in_the_same_ticket() -> None:
    result = mask_pii("карта 4276 1234 5678 9012 и почта a@b.ru")

    assert result.text == "карта [CARD] и почта [EMAIL]"
    assert set(result.types) == {"CARD", "EMAIL"}


def test_repeated_occurrences_are_all_masked_but_reported_once() -> None:
    result = mask_pii("почта a@b.ru и вторая почта c@d.ru")

    assert result.text == "почта [EMAIL] и вторая почта [EMAIL]"
    assert result.types == ("EMAIL",)


def test_masking_is_idempotent() -> None:
    """Masking already-masked text must be a no-op — markers are not re-matched."""
    once = mask_pii(T_1004)

    twice = mask_pii(once.text)

    assert twice.text == once.text
    assert twice.types == ()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(T_1004, True, id="t1004-has-pii"),
        pytest.param("заказ 77712", False, id="order-number-is-not-pii"),
        pytest.param("", False, id="empty-string"),
    ],
)
def test_contains_pii_matches_the_mask_result(raw: str, expected: bool) -> None:
    assert contains_pii(raw) is expected
