# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Coursework for the AITH "System Design" assignment: an internal HR/IT assistant Telegram bot (PoC) — RAG over the six Russian policy docs in `data/`, a LangGraph agent with tool calling (`get_vacation_balance`, `get_user_grade`), and human-in-the-loop escalation. `README.md` (Russian) carries the graded system-design write-up; `task.md` (gitignored) is the assignment text.

**Standing user rules (override anything else):**
- `AI_LOG.md` is a graded deliverable the user writes personally — NEVER create, edit, or append to it. Report AI_LOG-worthy material in chat instead.
- NEVER run `git add`/`git commit` — the commit history is graded and belongs to the user. Report what's ready to commit.
- Use Context7 to verify library APIs before writing code against them; use the `test-writer` subagent for writing/repairing tests.

## Commands

Everything runs through **uv**; never invoke bare `pytest`/`python`. Python >= 3.14.

```bash
uv sync                                     # install (incl. dev group)
uv run pytest -q                            # full suite (118 tests, ~1s, offline)
RUN_INTEGRATION=1 uv run pytest -m integration -q  # + real embedder on GPU (~13 GB model)
uv run pytest tests/agent/test_graph.py -q  # one file
uv run pytest -k "readiness" -q             # by name
uv run python -m src.main                   # API server (webhook mode; warm-up loads the 3B model, ~80s)
uv run python -m src.telegram.polling       # bot via long polling — no public URL needed
docker compose up --build                   # containerised: GPU reservation, hf-cache volume, ./src + ./data mounted ro
```

No linter or formatter is configured — don't invent a `ruff`/`black` step.

## Hardware / model constraints

- Machine has one Tesla T4 (15.4 GB, Turing): **no bf16, no flash-attention** — the embedder runs fp16 + `attn_implementation="eager"` on CUDA, fp32 on CPU.
- Embeddings: `ai-sage/Giga-Embeddings-instruct` (3B, dim 2048, `trust_remote_code=True`, instruction prefix on queries only). Its remote code is why **`transformers` is pinned `>=4.51,<5`** (5.x removed `ROPE_INIT_FUNCTIONS['default']`) and why `einops` is a dependency. Do not "upgrade" transformers to 5.x without re-testing the model load.
- FAISS (`IndexFlatIP` over L2-normalized vectors) strictly needs **float32** — the embedder casts `tensor.cpu().to(torch.float32).numpy()` before normalizing, and batches (`embedding_batch_size`) to avoid T4 OOM.
- Thresholds in `Settings` are calibrated on the real corpus: legit questions score >= 0.58 cosine, off-topic/injections <= 0.44 → `min_retrieval_score=0.5`. Re-calibrate if the corpus or model changes.
- Docker runtime image needs `gcc` (torch's triton JIT compiles a C stub at startup) — don't remove it from `src/Dockerfile`.

## Architecture

Model-Service-Repository. A request flows `src/api/routes/*` → `src/services/*` → `src/repositories/*`, and each layer only knows the one below it. The agent layer (`src/agent/`) sits beside services and is pure logic over injected abstractions.

**Wiring lives in `src/dependencies.py`, nowhere else.** `lru_cache`d singleton providers (`get_embedder`, `get_knowledge_base`, `get_hr_system`, `get_dialog_memory`, `get_llm_client`, `get_agent_graph`, `get_bot`, `get_dispatcher`, `get_update_deduplicator`) call `get_settings()` and each other directly; per-request providers (`get_assistant_service`, `get_health_service`) plus exported `Annotated` aliases feed the routes. Routes never write `Depends(...)` inline. New provider → add it to `_CACHED_PROVIDERS` in `tests/conftest.py` too. `get_bot()` returns `None` when the token is empty — **never construct `Bot("")`** (aiogram raises).

**The agent is a deterministic LangGraph** (`src/agent/graph.py`): `classify` (Gemini structured call №1 → `RouteDecision`) → optional `call_tools` → `retrieve` (FAISS top-k, drops `restricted_sources` chunks *before* the generator) → `generate` (Gemini call №2 → `GroundedAnswer`) → `escalate` (terminal, no LLM; prints the assignment-mandated `[ESCALATION] ...` line). All Gemini calls go through `GeminiAgentClient.generate_structured` (`response_schema`, tenacity retries on 408/429/5xx, everything else → `LLMUnavailableError`). `AssistantService` catches `LLMUnavailableError` and degrades to raw retrieval output. PII defense is three-layered (classifier, retrieval filter, generation post-check) — keep all three when touching any of them.

**Telegram**: webhook route (`src/api/routes/telegram.py`) is fast-ACK only — timing-safe secret check, `update_id` dedupe, `BackgroundTasks.add_task(feed_update)`, immediate `{"ok": true}`. Never run the pipeline inside the webhook request (Telegram re-sends slow updates → duplicate replies). Handlers live in `src/telegram/handlers.py`; `Dispatcher(assistant=...)` injects the service into handlers by parameter name.

**Errors are raised, not returned.** Services raise `AppError` subclasses from `src/exceptions.py` (`status_code` + machine-readable `code`); the single handler in `create_app()` renders `ErrorResponse`. Routes contain no `try`/`except` and no `HTTPException` — a new failure mode means a new `AppError` subclass.

**Models are split by direction, not by entity:** `models/domain.py` (frozen Pydantic, no FastAPI/storage imports), `models/requests.py` (`extra="forbid"`), `models/responses.py` (`ChatResponse.from_domain()` is the only domain→wire conversion). LLM output contracts live separately in `src/agent/schemas.py`.

**Repositories** are one small ABC per file (`AbstractKnowledgeBase`, `AbstractHRSystem`, `AbstractDialogMemory`) — async methods + `ping()` (backs readiness) + non-abstract `load()` warm-up awaited from the lifespan. The KB returns restricted chunks too: filtering is agent policy, not storage policy.

**Config**: `Settings` in `src/config.py` is the only code allowed to read the environment; reach it via the `lru_cache`d `get_settings()`. `.env` holds `GEMINI_API_KEY`, `TELEGRAM_BOT_API_KEY`, `HF_TOKEN` (+optional `TELEGRAM_WEBHOOK_SECRET`/`TELEGRAM_WEBHOOK_URL`). `warmup_on_startup=False` skips the model load (tests use this).

**Health**: `/api/v1/health` is liveness (no I/O). `/api/v1/health/ready` pings kb/hr/dialog-memory probes plus config checks; anything not-ok except `telegram_config` flips it to 503 with the same body shape.

## Testing

`tests/conftest.py` has two autouse fixtures — keep both invariants:
- `_reset_singletons` clears every provider in `_CACHED_PROVIDERS` around each test; add any new `lru_cache` provider there or the suite becomes order-dependent.
- `_fast_lifespan` sets `WARMUP_ON_STARTUP=false` and blanks the Telegram env vars, so `client` (which runs the lifespan) never loads the 3B model or touches the network.

Fixtures: `app`, `client` (runs lifespan), `async_client` (doesn't), `fake_embedder` (deterministic bag-of-Russian-stems `FakeEmbedder`). Patterns: scripted fake LLM (queue of `RouteDecision`/`GroundedAnswer`), preset-chunk fake KBs, fake bot/dispatcher via `dependency_overrides` (cleared in teardown). Never load torch/transformers or hit the network in default tests — only the `RUN_INTEGRATION=1`-gated test touches the real embedder.

Settings enforced by `pyproject.toml`:
- `asyncio_mode = "auto"` — plain `async def test_...`, no marker.
- `filterwarnings = ["error"]` — any warning fails. Two sanctioned narrow ignores exist (google-genai/langsmith import-time py3.14 deprecations); don't add more without the same justification.
- `--strict-markers` — only `slow` and `integration` are registered.
- `pythonpath = ["."]` — imports are `from src...`.

Use **`httpx2`**, not `httpx`, for test clients (Starlette deprecates `httpx` for `TestClient`). Runtime code may import `httpx` (google-genai's transport — used for retry exception types in `src/agent/llm.py`).

A `test-writer` subagent is configured in `.claude/agents/` for adding or repairing tests. Note `.claude/`, `data/`, `task.md`, and `CLAUDE.md` itself are gitignored.
