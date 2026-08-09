"""Tests for AgentTools — the assignment's tool-calling smoke requirements."""

from typing import Optional

import pytest

from src.agent.tools import AgentTools
from src.exceptions import EntityNotFoundError
from src.models.domain import Employee
from src.repositories.hr_system import AbstractHRSystem, StubHRSystem


class _EmptyHRSystem(AbstractHRSystem):
    """An HR system that knows nobody — forces the not-found branch."""

    async def get_employee(self, user_id: str) -> Optional[Employee]:
        return None

    async def ping(self) -> bool:
        return True


@pytest.fixture
def tools() -> AgentTools:
    return AgentTools(StubHRSystem())


@pytest.mark.parametrize(
    ("user_id", "expected"),
    [
        ("100001", 21),
        ("100002", 3),
        ("999999", 12),
    ],
    ids=["anna", "petr", "default-profile"],
)
async def test_get_vacation_balance_returns_exact_int(tools, user_id, expected):
    balance = await tools.get_vacation_balance(user_id)

    assert isinstance(balance, int)
    assert balance == expected


@pytest.mark.parametrize(
    ("user_id", "expected"),
    [
        ("100001", 4),
        ("100002", 1),
        ("999999", 2),
    ],
    ids=["anna", "petr", "default-profile"],
)
async def test_get_user_grade_returns_exact_int(tools, user_id, expected):
    grade = await tools.get_user_grade(user_id)

    assert isinstance(grade, int)
    assert grade == expected


@pytest.mark.parametrize(
    "tool_name",
    ["get_vacation_balance", "get_user_grade"],
)
async def test_unknown_employee_raises_entity_not_found(tool_name):
    tools = AgentTools(_EmptyHRSystem())

    with pytest.raises(EntityNotFoundError, match="unknown to the HR system") as exc_info:
        await getattr(tools, tool_name)("31337")

    assert "31337" in str(exc_info.value)
    # The AppError contract the API handler relies on.
    assert exc_info.value.status_code == 404
    assert exc_info.value.code == "not_found"
