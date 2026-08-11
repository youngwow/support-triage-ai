from uuid import UUID

from fastapi import APIRouter, Response, status

from src.dependencies import TriageServiceDep
from src.exceptions import EntityNotFoundError
from src.models.requests import TicketCreateRequest
from src.models.responses import AuditRecordResponse, AuditResponse, TicketResponse


router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.post(
    "",
    response_model=TicketResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a ticket for triage",
)
async def create_ticket(
    request: TicketCreateRequest,
    response: Response,
    service: TriageServiceDep,
) -> TicketResponse:
    """
    Synchronous path: mask, classify, route, log.

    Returns 201 for a new ticket and 200 when ``external_id`` matched one that
    was already triaged — a redelivered message costs nothing and cannot be
    classified twice.
    """
    ticket, created = await service.triage(
        channel=request.channel,
        text=request.text,
        external_id=request.external_id,
    )
    if not created:
        response.status_code = status.HTTP_200_OK
    return TicketResponse.from_domain(ticket)


@router.get(
    "/{ticket_id}",
    response_model=TicketResponse,
    summary="Read a ticket and its draft",
)
async def get_ticket(ticket_id: UUID, service: TriageServiceDep) -> TicketResponse:
    ticket = await service.get(ticket_id)
    if ticket is None:
        raise EntityNotFoundError(f"Ticket {ticket_id} does not exist")
    return TicketResponse.from_domain(ticket)


@router.get(
    "/{ticket_id}/audit",
    response_model=AuditResponse,
    summary="Read the decision trail for a ticket",
)
async def get_ticket_audit(ticket_id: UUID, service: TriageServiceDep) -> AuditResponse:
    """Every automatic decision taken for this ticket, oldest first."""
    ticket = await service.get(ticket_id)
    if ticket is None:
        raise EntityNotFoundError(f"Ticket {ticket_id} does not exist")
    records = await service.audit_trail(ticket_id)
    return AuditResponse(
        ticket_id=ticket_id,
        records=[AuditRecordResponse.from_domain(record) for record in records],
    )
