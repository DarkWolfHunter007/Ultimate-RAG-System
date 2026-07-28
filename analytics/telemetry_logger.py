import time
from typing import Any


class TelemetryLogger:
    """Tracks latency breakdown microsecond waterfall and query execution metrics."""

    def __init__(self, query_id: str = ""):
        self.query_id = query_id
        self.start_time = time.perf_counter()
        self.spans: list[dict[str, Any]] = []

    def log_span(self, name: str, duration_ms: float, details: str = ""):
        self.spans.append({
            "name": name,
            "duration_ms": round(duration_ms, 2),
            "details": details
        })

    def get_waterfall(self) -> dict[str, Any]:
        total_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        return {
            "query_id": self.query_id,
            "total_latency_ms": total_ms,
            "spans": self.spans
        }
