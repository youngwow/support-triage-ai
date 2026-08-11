"""Liveness and readiness.

The asymmetry between the two is the design decision under test: liveness does
no I/O at all, and readiness reports *every* dependency but only lets
``CRITICAL_PROBES`` flip the status. A missing Gemini key or an unloaded vector
index degrades the answer quality, not the instance's ability to serve — pulling
it out of the load balancer for either would trade a degraded service for no
service.
"""

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import __version__
from src.config import get_settings
from src.dependencies import get_audit_log, get_knowledge_base, get_ticket_repository


LIVENESS_URL = "/api/v1/health"
READINESS_URL = "/api/v1/health/ready"

#: ``(probe name, the provider that supplies it)`` for the two probes whose
#: failure is allowed to take the instance out of rotation.
CRITICAL_PROBE_PROVIDERS = (
    ("tickets", get_ticket_repository),
    ("audit", get_audit_log),
)


class StubProbe:
    """A ``SupportsPing`` stand-in that records whether it was pinged."""

    def __init__(self, *, result: bool = True, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.calls = 0

    async def ping(self) -> bool:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._result


@pytest.fixture
def override(app: FastAPI) -> Iterator[Callable[[Callable[..., Any], object], None]]:
    """Swap a dependency for one test and undo it afterwards.

    Only the keys this test added are removed: a leaked override is the classic
    source of order-dependent failures.
    """
    replaced: list[Callable[..., Any]] = []

    def _override(dependency: Callable[..., Any], value: object) -> None:
        app.dependency_overrides[dependency] = lambda: value
        replaced.append(dependency)

    yield _override

    for dependency in replaced:
        app.dependency_overrides.pop(dependency, None)


@pytest.fixture
def configured_gemini_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Put a key in the environment *before* the app is built.

    ``create_app()`` calls the ``lru_cache``d ``get_settings()``, so a test that
    wants a different configuration must request this fixture ahead of
    ``client`` in its signature — setting the variable inside the test body is
    already too late.
    """
    key = "test-key-not-a-real-credential"
    monkeypatch.setenv("GEMINI_API_KEY", key)
    return key


# --- liveness ------------------------------------------------------------


def test_liveness_reports_ok_with_no_checks(client: TestClient) -> None:
    response = client.get(LIVENESS_URL)

    settings = get_settings()
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": settings.app_name,
        "version": __version__,
        "environment": settings.environment,
        "checks": {},
    }


def test_liveness_never_pings_a_dependency(
    override: Callable[[Callable[..., Any], object], None],
    client: TestClient,
) -> None:
    probes = {
        get_ticket_repository: StubProbe(),
        get_audit_log: StubProbe(),
        get_knowledge_base: StubProbe(),
    }
    for provider, probe in probes.items():
        override(provider, probe)

    response = client.get(LIVENESS_URL)

    assert response.status_code == 200
    assert [probe.calls for probe in probes.values()] == [0, 0, 0]


def test_liveness_stays_ok_while_every_dependency_is_down(
    override: Callable[[Callable[..., Any], object], None],
    client: TestClient,
) -> None:
    for provider in (get_ticket_repository, get_audit_log, get_knowledge_base):
        override(provider, StubProbe(result=False))

    response = client.get(LIVENESS_URL)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# --- readiness: the default deployment -----------------------------------


def test_readiness_is_ok_although_retrieval_and_the_llm_are_unavailable(
    client: TestClient,
) -> None:
    """The regression test for "a missing API key keeps serving traffic".

    Warm-up is off in tests, so the FAISS index is genuinely unloaded and the
    key is genuinely blank — exactly the state a credential-less deployment
    boots into. Both are reported, neither is fatal.
    """
    response = client.get(READINESS_URL)

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["checks"] == {
        "tickets": "ok",
        "audit": "ok",
        "knowledge_base": "unavailable",
        "llm": "unconfigured",
    }


def test_readiness_body_carries_the_app_identity(client: TestClient) -> None:
    response = client.get(READINESS_URL)

    settings = get_settings()
    body = response.json()
    assert (body["app"], body["version"], body["environment"]) == (
        settings.app_name,
        __version__,
        settings.environment,
    )


# --- readiness: non-critical probes --------------------------------------


def test_an_unavailable_knowledge_base_does_not_take_the_instance_out_of_rotation(
    override: Callable[[Callable[..., Any], object], None],
    client: TestClient,
) -> None:
    override(get_knowledge_base, StubProbe(result=False))

    response = client.get(READINESS_URL)

    body = response.json()
    assert response.status_code == 200
    assert (body["status"], body["checks"]["knowledge_base"]) == ("ok", "unavailable")


def test_a_raising_probe_is_reported_as_error_and_not_propagated(
    override: Callable[[Callable[..., Any], object], None],
    client: TestClient,
) -> None:
    override(get_knowledge_base, StubProbe(error=RuntimeError("index file is corrupt")))

    response = client.get(READINESS_URL)

    body = response.json()
    assert response.status_code == 200
    assert (body["status"], body["checks"]["knowledge_base"]) == ("ok", "error")


# --- readiness: the llm check --------------------------------------------


def test_llm_check_is_unconfigured_without_a_key(client: TestClient) -> None:
    response = client.get(READINESS_URL)

    body = response.json()
    assert body["checks"]["llm"] == "unconfigured"
    assert (body["status"], response.status_code) == ("ok", 200)


def test_llm_check_is_ok_when_a_key_is_configured(
    configured_gemini_key: str,
    client: TestClient,
) -> None:
    response = client.get(READINESS_URL)

    body = response.json()
    assert body["checks"]["llm"] == "ok"
    assert (body["status"], response.status_code) == ("ok", 200)


# --- readiness: critical probes ------------------------------------------


@pytest.mark.parametrize(
    ("name", "provider"),
    CRITICAL_PROBE_PROVIDERS,
    ids=[name for name, _ in CRITICAL_PROBE_PROVIDERS],
)
def test_a_failing_critical_probe_degrades_readiness_to_503(
    name: str,
    provider: Callable[..., Any],
    override: Callable[[Callable[..., Any], object], None],
    client: TestClient,
) -> None:
    override(provider, StubProbe(result=False))

    response = client.get(READINESS_URL)

    body = response.json()
    assert response.status_code == 503
    assert body["status"] == "degraded"
    assert body["checks"][name] == "unavailable"


@pytest.mark.parametrize(
    ("name", "provider"),
    CRITICAL_PROBE_PROVIDERS,
    ids=[name for name, _ in CRITICAL_PROBE_PROVIDERS],
)
def test_a_raising_critical_probe_is_reported_as_error_and_degrades(
    name: str,
    provider: Callable[..., Any],
    override: Callable[[Callable[..., Any], object], None],
    client: TestClient,
) -> None:
    override(provider, StubProbe(error=ConnectionError("connection refused")))

    response = client.get(READINESS_URL)

    body = response.json()
    assert response.status_code == 503
    assert body["checks"][name] == "error"


def test_a_degraded_readiness_keeps_the_same_body_shape(
    override: Callable[[Callable[..., Any], object], None],
    client: TestClient,
) -> None:
    """503 changes the status code and the status field, nothing else.

    An orchestrator gets the signal from the code and the reason from the same
    body it already parses.
    """
    healthy = client.get(READINESS_URL).json()
    override(get_ticket_repository, StubProbe(result=False))

    degraded = client.get(READINESS_URL).json()

    assert degraded == healthy | {
        "status": "degraded",
        "checks": healthy["checks"] | {"tickets": "unavailable"},
    }
