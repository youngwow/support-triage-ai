"""Tests for the deterministic StubHRSystem."""

import pytest

from src.models.domain import Employee
from src.repositories.hr_system import StubHRSystem


@pytest.fixture
def hr_system() -> StubHRSystem:
    return StubHRSystem()


@pytest.mark.parametrize(
    ("user_id", "full_name", "grade", "vacation_balance_days"),
    [
        ("100001", "Анна Смирнова", 4, 21),
        ("100002", "Пётр Иванов", 1, 3),
    ],
    ids=["anna", "petr"],
)
async def test_known_employee_profile(
    hr_system, user_id, full_name, grade, vacation_balance_days
):
    employee = await hr_system.get_employee(user_id)

    assert isinstance(employee, Employee)
    assert employee.user_id == user_id
    assert employee.full_name == full_name
    assert employee.grade == grade
    assert employee.vacation_balance_days == vacation_balance_days


async def test_unknown_id_resolves_to_default_profile_echoing_the_id(hr_system):
    employee = await hr_system.get_employee("424242")

    assert employee is not None
    assert employee.user_id == "424242"
    assert employee.full_name == "Сотрудник компании"
    assert employee.grade == 2
    assert employee.vacation_balance_days == 12


async def test_each_unknown_id_gets_its_own_echo(hr_system):
    first = await hr_system.get_employee("111")
    second = await hr_system.get_employee("222")

    assert first.user_id == "111"
    assert second.user_id == "222"


async def test_ping_is_true(hr_system):
    assert await hr_system.ping() is True
