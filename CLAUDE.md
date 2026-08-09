# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Coursework for the AITH "System Design" assignment (see `README.md`, in Russian). The FastAPI app under `src/` is a **template**: `Item` is a placeholder aggregate meant to be replaced by the real domain. The declared dependencies (torch, transformers, faiss, langchain/langgraph, google-genai, aiogram) are the intended stack for the actual NLP case and are not yet imported anywhere in `src/`.

`AI_LOG.md` is a graded deliverable — it records how AI tooling was used, in Russian. Append to it when a session produces a meaningful chunk of the work.

## Commands

Everything runs through **uv**; never invoke bare `pytest`/`python`. Python >= 3.14.

```bash
uv sync                                     # install (incl. dev group)
uv run pytest -q                            # full suite (73 tests, <1s)
uv run pytest tests/test_items_api.py -q    # one file
uv run pytest tests/test_items_api.py::test_create_defaults_the_description_to_null
uv run pytest -k "readiness" -q             # by name
uv run pytest -m "not slow" -q              # by marker
uv run python -m src.main                   # dev server (auto-reload when environment=local)
docker compose up --build                   # containerised, mounts ./src read-only with --reload
```

No linter or formatter is configured — don't invent a `ruff`/`black` step.

## Architecture

Model-Service-Repository. A request flows `src/api/routes/*` → `src/services/*` → `src/repositories/*`, and each layer only knows the one below it.

**Wiring lives in `src/dependencies.py`, nowhere else.** It exports `Annotated` type aliases (`SettingsDep`, `ItemRepositoryDep`, `ItemServiceDep`, `HealthServiceDep`); routes declare `service: ItemServiceDep` and never write `Depends(...)` inline or construct a service themselves. New service → add its provider and alias there.

**Errors are raised, not returned.** Services raise `AppError` subclasses from `src/exceptions.py`, each carrying its own `status_code` and machine-readable `code`. The single handler registered in `create_app()` turns any of them into an `ErrorResponse` body. Routes contain no `try`/`except` and no `HTTPException` — a new failure mode means a new `AppError` subclass.

**Models are split by direction, not by entity:**
- `models/domain.py` — frozen Pydantic entities, no FastAPI and no storage imports. Domain behaviour lives here as methods returning new instances (`Item.deactivate()`), not in services.
- `models/requests.py` — inbound, all `extra="forbid"` so a typo'd field is a 422 rather than a silent no-op.
- `models/responses.py` — outbound. `ItemResponse.from_domain()` is the *only* place a domain entity becomes wire format.

**Repositories** implement the async `AbstractRepository` contract. Two hooks matter beyond CRUD: `load()` is an optional warm-up awaited once from the app lifespan, and `ping()` backs the readiness probe. Swapping the in-memory store for a real one means adding an implementation and changing `get_item_repository()` — nothing else. The repository signals "duplicate id" with a bare `KeyError`, which `ItemService.create_item` translates to `EntityAlreadyExistsError`.

**Config**: `Settings` in `src/config.py` is the only code allowed to read the environment; reach it via the `lru_cache`d `get_settings()`. `.env` at the repo root is loaded automatically and currently holds `GEMINI_API_KEY`.

**App construction**: `create_app()` is a factory; the module-level `app = create_app()` exists only for uvicorn. Tests build their own instance so they never share state through the import system.

**Health**: `/api/v1/health` is liveness (always ok, no I/O). `/api/v1/health/ready` runs the dependency checks and flips the status code to 503 while keeping the same `HealthResponse` body, so an orchestrator gets both the signal and the reason.

## Testing

`get_settings` and `get_item_repository` are `lru_cache`d singletons, so `tests/conftest.py` has an autouse fixture clearing both around every test. Without it the in-memory store leaks rows between tests and the suite becomes order-dependent — keep that invariant when adding cached providers.

Fixtures already available: `app`, `client` (sync `TestClient`, **runs the lifespan**), `async_client` (ASGITransport, **does not** run the lifespan — use `client` if the test needs anything lifespan sets up), `item_repository` (the same instance the app resolves — seed through it rather than driving the API to build preconditions).

Settings enforced by `pyproject.toml` that change how tests must be written:
- `asyncio_mode = "auto"` — write plain `async def test_...`, no `@pytest.mark.asyncio`.
- `filterwarnings = ["error"]` — any warning fails the run. Fix the cause; don't add an ignore.
- `--strict-markers` — only `slow` and `integration` are registered; register a new one before using it.
- `pythonpath = ["."]` — `src` is not an installed package, so imports are `from src...`.

Use **`httpx2`**, not `httpx`: Starlette deprecates `httpx` for `TestClient`, and a DeprecationWarning is a failure here.

A `test-writer` subagent is configured in `.claude/agents/` for adding or repairing tests. Note `.claude/` and `data/` are gitignored.
