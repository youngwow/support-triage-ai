# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Coursework for the AITH "System Design" assignment. `task_for_ai.md` is the brief (Russian): design an AI/ML system that automates support-ticket handling for a service with ~5M users and ~200k tickets/day, and back the design with a **minimal PoC** — not a production system.

The deliverables are the prose (`README.md`, `docs/*.md`) as much as the code. The assignment explicitly grades *against* volume: "не писать много кода ради объёма". Prefer deleting to adding.

**Never touch `AI_USAGE.md` or `SELF_REVIEW.md`** — the user writes those personally. **Never make git commits**; the graded commit history is the user's.

## Commands

Everything runs through **uv**; never invoke bare `pytest`/`python`. Python >= 3.14.

```bash
uv sync                                     # install (incl. dev group)
uv run pytest -q                            # full suite
uv run pytest tests/ml -q                   # one directory
uv run pytest -k "pii" -q                   # by name
uv run pytest -m "not slow" -q              # by marker
RUN_INTEGRATION=1 uv run pytest -m integration -q   # the real-embedder test

uv run python -m src.main                   # dev server (auto-reload when environment=local)
uv run python scripts/demo.py               # end-to-end demo against a running instance
uv run python scripts/evaluate.py           # rules baseline on the golden set
uv run python scripts/evaluate.py --llm     # also scores Gemini (30 API calls)
docker compose up --build                   # containerised; needs a GPU, or set WARMUP_ON_STARTUP=false
```

No linter or formatter is configured — don't invent a `ruff`/`black` step.

## The two paths

This is the whole design, and every file belongs to one of them.

**Triage — synchronous, `POST /api/v1/tickets`.** `src/services/triage_service.py`: mask PII → deterministic rules → LLM classification → risk → route → store → audit → enqueue. Order is load-bearing:

1. **PII is masked first** (`src/ml/pii.py`). Everything downstream — the stored ticket, the classifier prompt, the audit record — sees masked text only, so "no personal data reaches the external LLM" holds by construction rather than by discipline. The patterns are anchored so an order number survives; `tests/ml/test_pii.py` pins that.
2. **Rules run before the model** (`src/ml/rules.py`) and set a risk *floor*. Risk is combined with `max_risk()`, so a model can never talk the system out of a rule that fired. Rules also screen for prompt injection; a flagged ticket never reaches the provider.
3. **Classification is an LLM call** (`src/ml/classifier.py`). `LLMUnavailableError` is caught and `RuleTopicClassifier` takes over, with confidence capped below `min_topic_confidence` so a degraded decision can never be auto-sent.

**Draft — asynchronous.** A bounded `asyncio.Queue` (`src/repositories/draft_queue.py`) feeds one worker started in the lifespan. `src/agent/graph.py` is a 3-node LangGraph: `retrieve` → `generate` → `END | escalate`. One generation call per ticket is the cost ceiling — groundedness and confidence come back in the *same* structured call as the answer, not a second one. `senior_escalation` tickets are never enqueued, so risky tickets cost nothing to generate.

Nothing is ever sent to a user. `auto_send_allowed` means "would have been eligible"; a human is always in the loop.

## Deliberate compromises

Do not "fix" these without asking — they are documented decisions, and the docs explain them:

- **The <500 ms triage budget is not met.** An LLM call is ~2.3 s. Measured, reported through `GET /api/v1/metrics` (`over_budget`, `over_budget_ratio`), and the distillation path is described in `docs/ml.md`. The `TopicClassifier` Protocol is the seam a distilled model slots into.
- **The queue is not durable.** Restarting loses whatever is in it; the lifespan logs how many. RabbitMQ is the named production replacement.
- **Everything is in-memory** — tickets, audit log, FAISS index.

## Conventions

**Wiring lives in `src/dependencies.py`, nowhere else.** It exports `Annotated` aliases (`TriageServiceDep`, `KnowledgeBaseDep`, …); routes declare `service: TriageServiceDep` and never write `Depends(...)` inline. Two tiers: `@lru_cache` singletons for anything holding state, plain functions for per-request services. **Every new cached provider must be added to `_CACHED_PROVIDERS` in `tests/conftest.py`** or state leaks between tests.

**Errors are raised, not returned.** Services raise `AppError` subclasses from `src/exceptions.py`, each carrying its own `status_code` and machine-readable `code`. One handler in `create_app()` renders them. Routes contain no `try`/`except` and no `HTTPException`. `LLMUnavailableError` and `KnowledgeBaseUnavailableError` are *routes*, not failures: callers catch them and degrade.

**`get_llm_client()` returns `NullLLMClient` when no key is set**, because `genai.Client(api_key="")` raises at construction. This is what makes the whole PoC runnable with no credentials — the degraded path is the default local experience.

**Models are split by direction:** `models/domain.py` (frozen entities, no FastAPI, no storage; behaviour lives here as methods returning new instances), `models/requests.py` (`extra="forbid"`), `models/responses.py` (`TicketResponse.from_domain()` is the *only* place a domain entity becomes wire format).

**Metrics are a projection, not counters.** `MetricsService` recomputes everything from the audit log, so a dashboard number can always be traced to the records behind it.

**Readiness is deliberately tolerant.** Only `CRITICAL_PROBES = ("tickets", "audit")` flip `/health/ready` to 503. A missing API key or an unloaded index is reported but does not pull the instance out of the load balancer — the service still triages without them.

**Config**: `Settings` in `src/config.py` is the only code allowed to read the environment; reach it via the `lru_cache`d `get_settings()`.

## Testing

`tests/conftest.py` has two autouse fixtures: `_reset_singletons` clears every cached provider, and `_fast_lifespan` sets `WARMUP_ON_STARTUP=false`, `DRAFT_WORKER_ENABLED=false` and blanks `GEMINI_API_KEY`. So tests never load the 3B embedding model, never start the worker, and never reach the network. Draft tests call `await draft_service.process_next()` explicitly instead of racing a background task.

Fixtures: `app`, `client` (sync `TestClient`, **runs the lifespan**), `async_client` (ASGITransport, **does not**), `fake_embedder` (numpy bag-of-stems, keeps a bias axis so no vector normalises to NaN), `ticket_repository`, `audit_log`, `draft_queue`.

Injection style: `app.dependency_overrides` (cleared in teardown) at the HTTP layer only; hand-written recording fakes via the constructor everywhere below. No mock library.

Settings from `pyproject.toml` that change how tests must be written:
- `asyncio_mode = "auto"` — plain `async def test_...`, no `@pytest.mark.asyncio`.
- `filterwarnings = ["error"]` — any warning fails. The two ignores are import-time Python-3.14 deprecations in google-genai/langsmith; don't add more without a reason in the comment.
- `--strict-markers` — only `slow` and `integration` are registered.
- `pythonpath = ["."]` — imports are `from src...`.

Use **`httpx2`**, not `httpx`, for test clients (Starlette deprecates `httpx` for `TestClient`, and a DeprecationWarning is a failure). Note `src/agent/llm.py` imports plain `httpx` for its retry predicate — that is correct.

`transformers` is pinned `>=4.56,<5`: Giga-Embeddings' `trust_remote_code` is written against 4.x, and `dtype=` in `from_pretrained` needs >= 4.56.

## Subagents

`.claude/agents/` holds two, both worth using rather than doing the work inline:

- **`test-writer`** — adds or repairs pytest tests. Give it the target and the behaviours to pin.
- **`docs-writer`** — the *only* thing that writes `docs/*.md` and `README.md`. It is restricted to those files and to the scope in each document's `## Что должно быть:` block, which is the assignment's checklist and must not be deleted.

`.claude/` and the HF cache are gitignored; `data/` is committed and ships in the image.
