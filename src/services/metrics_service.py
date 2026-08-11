"""
Operational metrics, computed as a projection of the audit log.

There is no counter registry: every number here is derived from the same records
that document the decisions, so a figure on a dashboard can always be traced
back to the tickets behind it and the two can never drift apart.

The endpoint returns JSON, not the Prometheus exposition format. In a real
deployment these same aggregates would be exported as Prometheus gauges and
histograms; the projection logic is the part worth showing in a PoC.
"""

from collections import Counter
from math import ceil, floor

from src.config import Settings
from src.models.responses import LatencyStats, MetricsResponse
from src.repositories.audit_log import AbstractAuditLog


def _percentile(sorted_values: list[float], quantile: float) -> float:
    """Linear-interpolated percentile over an already-sorted list."""
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * quantile
    lower, upper = floor(position), ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


class MetricsService:
    def __init__(self, *, audit: AbstractAuditLog, settings: Settings) -> None:
        self._audit = audit
        self._settings = settings

    async def snapshot(self) -> MetricsResponse:
        records = await self._audit.records()
        triaged = [record for record in records if record.event == "triaged"]
        drafted = [record for record in records if record.event == "drafted"]

        latencies = sorted(
            float(record.details.get("latency_ms", 0.0)) for record in triaged
        )
        budget = self._settings.hot_path_budget_ms
        over_budget = sum(1 for value in latencies if value > budget)

        llm_classifications = sum(
            1 for record in triaged if record.details.get("classifier") == "llm"
        )
        rule_fallbacks = sum(
            1 for record in triaged if record.details.get("classifier") == "rules"
        )
        drafts_degraded = sum(1 for record in drafted if record.details.get("degraded"))
        generation_calls = len(drafted) - drafts_degraded

        return MetricsResponse(
            tickets_total=len(triaged),
            by_category=dict(
                Counter(str(record.details.get("category", "")) for record in triaged)
            ),
            by_risk=dict(
                Counter(str(record.details.get("risk", "")) for record in triaged)
            ),
            by_route=dict(
                Counter(str(record.details.get("route", "")) for record in triaged)
            ),
            auto_send_allowed=sum(
                1 for record in triaged if record.details.get("auto_send_allowed")
            ),
            low_confidence=sum(
                1
                for record in triaged
                if float(record.details.get("category_confidence", 0.0))
                < self._settings.min_topic_confidence
            ),
            pii_detected=sum(1 for record in triaged if record.details.get("pii_types")),
            injection_flagged=sum(
                1 for record in triaged if record.details.get("injection_flagged")
            ),
            llm_classifications=llm_classifications,
            rule_fallbacks=rule_fallbacks,
            llm_calls=llm_classifications + generation_calls,
            llm_failures=(
                sum(1 for record in records if record.event == "llm_classification_failed")
                + drafts_degraded
            ),
            drafts_ready=sum(
                1 for record in drafted if record.details.get("status") == "draft_ready"
            ),
            drafts_degraded=drafts_degraded,
            queue_rejected=sum(1 for record in records if record.event == "queue_rejected"),
            triage_latency_ms=LatencyStats(
                count=len(latencies),
                p50_ms=round(_percentile(latencies, 0.50), 2),
                p95_ms=round(_percentile(latencies, 0.95), 2),
                max_ms=round(latencies[-1], 2) if latencies else 0.0,
            ),
            hot_path_budget_ms=budget,
            over_budget=over_budget,
            over_budget_ratio=round(over_budget / len(latencies), 4) if latencies else 0.0,
        )
