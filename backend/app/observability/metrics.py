from __future__ import annotations

import resource
import sys
from collections import Counter
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class RequestRecord:
    total_ms: float
    status: str
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    grounding_ms: float = 0.0
    route: str | None = None


def get_current_memory_mb() -> float:
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # Darwin ru_maxrss is in bytes; Linux is in kilobytes
        factor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
        return round(ru.ru_maxrss / factor, 2)
    except Exception:
        return 0.0


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = Lock()
        self._records: list[RequestRecord] = []
        self._statuses: Counter[str] = Counter()
        self._routes: Counter[str] = Counter()
        self._startup_time_ms: float | None = None
        self._first_request_latency_ms: float | None = None
        self._second_request_latency_ms: float | None = None

    def set_startup_time(self, startup_ms: float) -> None:
        with self._lock:
            self._startup_time_ms = round(startup_ms, 2)

    def record(self, record: RequestRecord) -> None:
        with self._lock:
            self._records.append(record)
            self._statuses[record.status] += 1
            if record.route:
                self._routes[record.route] += 1
            if len(self._records) == 1 and self._first_request_latency_ms is None:
                self._first_request_latency_ms = round(record.total_ms, 2)
            elif len(self._records) == 2 and self._second_request_latency_ms is None:
                self._second_request_latency_ms = round(record.total_ms, 2)

    def snapshot(self) -> dict:
        with self._lock:
            total_values = sorted(item.total_ms for item in self._records)
            retr_values = sorted(item.retrieval_ms for item in self._records if item.retrieval_ms > 0)
            rerank_values = sorted(item.rerank_ms for item in self._records if item.rerank_ms > 0)
            gen_values = sorted(item.generation_ms for item in self._records if item.generation_ms > 0)
            count = len(total_values)

            def calc_percentiles(vals: list[float]) -> dict[str, float]:
                if not vals:
                    return {"min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "p100": 0.0, "avg": 0.0}
                n = len(vals)
                pct = lambda p: round(vals[min(n - 1, int(n * p / 100))], 2)
                return {
                    "min": round(vals[0], 2),
                    "p50": pct(50),
                    "p90": pct(90),
                    "p95": pct(95),
                    "p99": pct(99),
                    "p100": round(vals[-1], 2),
                    "avg": round(sum(vals) / n, 2),
                }

            return {
                "requests": count,
                "statuses": dict(self._statuses),
                "routes": dict(self._routes),
                "startup_time_ms": self._startup_time_ms,
                "first_request_latency_ms": self._first_request_latency_ms,
                "second_request_latency_ms": self._second_request_latency_ms,
                "memory_usage_mb": get_current_memory_mb(),
                "latency_ms": calc_percentiles(total_values),
                "retrieval_latency_ms": calc_percentiles(retr_values),
                "rerank_latency_ms": calc_percentiles(rerank_values),
                "generation_latency_ms": calc_percentiles(gen_values),
            }


