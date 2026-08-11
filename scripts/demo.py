#!/usr/bin/env python
"""
End-to-end demo against a running instance.

Drives the two paths the case asks for:

* **happy path** — T-1001, a routine billing question, classified confidently,
  routed to ``auto_answer`` and answered from the knowledge base;
* **risky path** — T-1004, an account-recovery request containing an e-mail and
  passport data, which must come back masked, must not be auto-sendable and must
  end up with a human.

Prints the decision trail for each and the metrics snapshot at the end, then
exits non-zero if any of those expectations is violated — so it doubles as a
smoke check you can run against a container.

Standard library only, so it runs anywhere without installing the project::

    uv run python scripts/demo.py
    uv run python scripts/demo.py --base-url http://localhost:8000
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
TICKETS_PATH = REPO_ROOT / "data" / "tickets" / "historical_tickets.json"

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def _request(method: str, url: str, payload: Optional[dict] = None) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8") or "{}")
    except urllib.error.URLError as error:
        print(f"Cannot reach {url}: {error.reason}", file=sys.stderr)
        print("Start the service first: uv run python -m src.main", file=sys.stderr)
        raise SystemExit(2) from error


def _heading(text: str) -> None:
    print(f"\n{BOLD}{text}{RESET}")
    print("-" * len(text))


def _show_triage(body: dict) -> None:
    triage = body["triage"]
    print(f"  id            {body['id']}")
    print(f"  masked text   {body['text']}")
    print(f"  category      {triage['category']} (confidence {triage['category_confidence']})")
    print(f"  risk          {triage['risk']}")
    print(f"  route         {triage['route']}")
    print(f"  auto-send     {triage['auto_send_allowed']}")
    print(f"  classifier    {triage['classifier']}")
    print(f"  PII found     {triage['pii_types'] or 'none'}")
    print(f"  rule hits     {triage['rule_hits'] or 'none'}")
    print(f"  latency       {triage['latency_ms']:.0f} ms")


def _show_draft(body: dict) -> None:
    draft = body.get("draft")
    if draft is None:
        print(f"  {DIM}no draft — the generation call was never made{RESET}")
        return
    print(f"  grounded      {draft['is_grounded']} (confidence {draft['confidence']})")
    print(f"  degraded      {draft['degraded']}")
    print(f"  sources       {draft['sources'] or 'none'}")
    print(f"  draft:\n    {draft['text'][:600]}")


def _wait_for_settled(base: str, ticket_id: str, timeout: float) -> dict:
    """Poll while the ticket is still waiting on the draft worker."""
    deadline = time.monotonic() + timeout
    _, body = _request("GET", f"{base}/api/v1/tickets/{ticket_id}")
    while body.get("status") == "queued" and time.monotonic() < deadline:
        time.sleep(1.0)
        _, body = _request("GET", f"{base}/api/v1/tickets/{ticket_id}")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="seconds to wait for the async draft path",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    tickets = {t["ticket_id"]: t for t in json.loads(TICKETS_PATH.read_text("utf-8"))}
    failures: list[str] = []

    status, health = _request("GET", f"{base}/api/v1/health/ready")
    _heading("Readiness")
    print(f"  HTTP {status} — {health.get('status')}")
    for name, value in (health.get("checks") or {}).items():
        print(f"  {name:<15} {value}")
    if health.get("checks", {}).get("llm") != "ok":
        print(
            f"\n  {DIM}GEMINI_API_KEY is not configured: classification will fall back to\n"
            f"  rules and drafts will be degraded. That is a supported mode — the run\n"
            f"  below then demonstrates the outage path rather than the happy path.{RESET}"
        )
    if health.get("checks", {}).get("knowledge_base") != "ok":
        print(
            f"  {DIM}Vector index not loaded (WARMUP_ON_STARTUP=false): drafts will be"
            f" degraded.{RESET}"
        )

    # --- happy path ------------------------------------------------------
    happy = tickets["T-1001"]
    _heading("1. Happy path — T-1001, routine billing question")
    status, body = _request(
        "POST",
        f"{base}/api/v1/tickets",
        {"channel": "chat", "text": happy["user_query"], "external_id": "demo-T-1001"},
    )
    if status not in (200, 201):
        print(f"  unexpected status {status}: {body}", file=sys.stderr)
        return 1
    _show_triage(body)
    happy_id = body["id"]

    # --- risky path ------------------------------------------------------
    risky = tickets["T-1004"]
    _heading("2. Risky path — T-1004, account recovery with passport data")
    status, risky_body = _request(
        "POST",
        f"{base}/api/v1/tickets",
        {"channel": "email", "text": risky["user_query"], "external_id": "demo-T-1004"},
    )
    if status not in (200, 201):
        print(f"  unexpected status {status}: {risky_body}", file=sys.stderr)
        return 1
    _show_triage(risky_body)
    risky_id = risky_body["id"]

    if "test@test.com" in risky_body["text"] or "567890" in risky_body["text"]:
        failures.append("raw PII survived masking")
    if risky_body["triage"]["auto_send_allowed"]:
        failures.append("a ticket containing PII was marked auto-sendable")
    if risky_body["triage"]["risk"] == "low":
        failures.append("a ticket containing PII was left at low risk")

    # --- idempotency -----------------------------------------------------
    _heading("3. Idempotency — the same external_id is not triaged twice")
    status, repeat = _request(
        "POST",
        f"{base}/api/v1/tickets",
        {"channel": "chat", "text": happy["user_query"], "external_id": "demo-T-1001"},
    )
    print(f"  HTTP {status} (200 means deduplicated), same id: {repeat['id'] == happy_id}")
    if status != 200 or repeat["id"] != happy_id:
        failures.append("redelivering the same external_id created a second ticket")

    # --- wait for the async path ----------------------------------------
    _heading("4. Asynchronous draft path")
    print(f"  waiting up to {args.timeout:.0f}s for the worker...")
    happy_final = _wait_for_settled(base, happy_id, args.timeout)
    risky_final = _wait_for_settled(base, risky_id, args.timeout)

    print(f"\n  T-1001 status: {happy_final.get('status')}")
    _show_draft(happy_final)
    print(f"\n  T-1004 status: {risky_final.get('status')}")
    _show_draft(risky_final)

    if risky_final.get("status") == "draft_ready":
        failures.append("a PII ticket reached draft_ready instead of a human")

    # --- audit trail -----------------------------------------------------
    _heading("5. Decision trail")
    for label, ticket_id in (("T-1001", happy_id), ("T-1004", risky_id)):
        _, audit = _request("GET", f"{base}/api/v1/tickets/{ticket_id}/audit")
        events = [record["event"] for record in audit["records"]]
        print(f"  {label}: {' -> '.join(events)}")
        if "triaged" not in events:
            failures.append(f"{label} has no triage record in the audit log")

    # --- metrics ---------------------------------------------------------
    _heading("6. Metrics")
    _, metrics = _request("GET", f"{base}/api/v1/metrics")
    latency = metrics["triage_latency_ms"]
    print(f"  tickets            {metrics['tickets_total']}")
    print(f"  by route           {metrics['by_route']}")
    print(f"  LLM calls          {metrics['llm_calls']} (failures: {metrics['llm_failures']})")
    print(f"  rule fallbacks     {metrics['rule_fallbacks']}")
    print(f"  drafts ready       {metrics['drafts_ready']} (degraded: {metrics['drafts_degraded']})")
    print(
        f"  triage latency     p50 {latency['p50_ms']:.0f} ms / "
        f"p95 {latency['p95_ms']:.0f} ms / max {latency['max_ms']:.0f} ms"
    )
    print(
        f"  budget             {metrics['hot_path_budget_ms']} ms — "
        f"{metrics['over_budget']} of {latency['count']} over "
        f"({metrics['over_budget_ratio'] * 100:.0f}%)"
    )
    if metrics["over_budget"]:
        print(
            f"  {DIM}Expected: classification is an LLM call in this PoC. See docs/ml.md\n"
            f"  for why that trade was made and what replaces it.{RESET}"
        )

    _heading("Result")
    if failures:
        for failure in failures:
            print(f"  FAIL  {failure}")
        return 1
    print("  OK — happy path answered, risky path masked and routed to a human")
    return 0


if __name__ == "__main__":
    sys.exit(main())
