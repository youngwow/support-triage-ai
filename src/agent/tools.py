from src.exceptions import EntityNotFoundError
from src.models.domain import Employee
from src.repositories.hr_system import AbstractHRSystem


class AgentTools:
    def __init__(self, hr_system: AbstractHRSystem) -> None:
        self._hr_system = hr_system

    async def get_vacation_balance(self, user_id: str) -> int:
        """Remaining vacation days of the user."""
        employee = await self._employee(user_id)
        return employee.vacation_balance_days

    async def get_user_grade(self, user_id: str) -> int:
        """The user's grade in the external HR system."""
        employee = await self._employee(user_id)
        return employee.grade

    async def _employee(self, user_id: str) -> Employee:
        employee = await self._hr_system.get_employee(user_id)
        if employee is None:
            raise EntityNotFoundError(f"Employee '{user_id}' is unknown to the HR system")
        return employee
