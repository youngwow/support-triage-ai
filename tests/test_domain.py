"""Coverage for ``src/models/domain.py`` and the time helper it depends on."""

from datetime import UTC
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.models.domain import Item
from src.utils import utc_now


def test_utc_now_is_timezone_aware() -> None:
    """A naive datetime here would silently corrupt every stored timestamp."""
    now = utc_now()

    assert now.tzinfo is not None
    assert now.utcoffset() == UTC.utcoffset(None)


def test_item_defaults_to_active_with_generated_timestamps() -> None:
    item = Item(id=uuid4(), name="widget")

    assert item.is_active is True
    assert item.description is None
    assert item.created_at.tzinfo is not None
    assert item.updated_at.tzinfo is not None


def test_item_is_frozen() -> None:
    """Entities are replaced, never mutated in place."""
    item = Item(id=uuid4(), name="widget")

    with pytest.raises(ValidationError):
        item.name = "mutated"


def test_item_rejects_an_empty_name() -> None:
    with pytest.raises(ValidationError):
        Item(id=uuid4(), name="")


def test_item_rejects_an_over_long_name() -> None:
    with pytest.raises(ValidationError):
        Item(id=uuid4(), name="x" * 201)


def test_deactivate_returns_a_new_inactive_copy() -> None:
    item = Item(id=uuid4(), name="widget")

    deactivated = item.deactivate()

    assert deactivated is not item
    assert deactivated.is_active is False
    assert item.is_active is True, "the original must be untouched"
    assert deactivated.id == item.id
    assert deactivated.created_at == item.created_at
    assert deactivated.updated_at >= item.updated_at
