from abc import ABC, abstractmethod
from typing import Optional

from src.models.domain import Employee


class AbstractHRSystem(ABC):
    """
    The 'external' HR system the agent asks about balances and grades.
    """

    @abstractmethod
    async def get_employee(self, user_id: str) -> Optional[Employee]:
        """The employee profile, or ``None`` when the id is unknown."""

    @abstractmethod
    async def ping(self) -> bool:
        """True when the system can be reached."""

    async def load(self) -> None:
        """Optional warm-up hook awaited once from the app lifespan."""
        return None


class StubHRSystem(AbstractHRSystem):
    """
    Deterministic stand-in so the PoC needs no real integration.

    Unknown ids resolve to a default grade-2 profile with a 12-day balance,
    so any Telegram account can run the demo scenarios (grade 2 → $50 per
    diem, no business class; balance 12 ≥ 5 → trips allowed).
    """

    _DEFAULT = Employee(
        user_id="default",
        full_name="Сотрудник компании",
        grade=2,
        vacation_balance_days=12,
    )

    def __init__(self) -> None:
        self._employees = {
            "100001": Employee(
                user_id="100001",
                full_name="Анна Смирнова",
                grade=4,
                vacation_balance_days=21,
            ),
            "100002": Employee(
                user_id="100002",
                full_name="Пётр Иванов",
                grade=1,
                vacation_balance_days=3,
            ),
        }

    async def get_employee(self, user_id: str) -> Optional[Employee]:
        employee = self._employees.get(user_id)
        if employee is None:
            employee = self._DEFAULT.model_copy(update={"user_id": user_id})
        return employee

    async def ping(self) -> bool:
        return True
