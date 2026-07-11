
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetricsSnapshot:
    """Performance metrics snapshot for a single request or batch."""
    timestamp: float
    request_id: Optional[int] = None
    batch_size: int = 1
    ttft: float = 0.0  # Time to First Token (seconds)
    tpot: float = 0.0  # Time Per Output Token (seconds)
    latency: float = 0.0  # Total latency (seconds)
    tokens_generated: int = 0
    throughput: float = 0.0  # Tokens per second
    gpu_utilization: float = 0.0  # Simulated GPU utilization (0-1)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "batch_size": self.batch_size,
            "ttft": self.ttft,
            "tpot": self.tpot,
            "latency": self.latency,
            "tokens_generated": self.tokens_generated,
            "throughput": self.throughput,
            "gpu_utilization": self.gpu_utilization,
        }


class MetricsCollector:
    """Collect and aggregate performance metrics."""

    def __init__(self):
        self.snapshots: List[MetricsSnapshot] = []
        self._start_time: Optional[float] = None

    def start(self):
        """Mark benchmark start time."""
        self._start_time = time.time()

    def record(self, snapshot: MetricsSnapshot):
        """Record a metrics snapshot."""
        self.snapshots.append(snapshot)

    def calculate_aggregates(self) -> Dict:
        """Calculate aggregate metrics from all snapshots."""
        if not self.snapshots:
            return {}

        ttfts = [s.ttft for s in self.snapshots if s.ttft > 0]
        tpos = [s.tpot for s in self.snapshots if s.tpot > 0]
        latencies = [s.latency for s in self.snapshots if s.latency > 0]
        throughputs = [s.throughput for s in self.snapshots if s.throughput > 0]

        def percentile(data, p):
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]

        return {
            "count": len(self.snapshots),
            "avg_ttft": sum(ttfts) / len(ttfts) if ttfts else 0.0,
            "p50_ttft": percentile(ttfts, 50),
            "p95_ttft": percentile(ttfts, 95),
            "p99_ttft": percentile(ttfts, 99),
            "avg_tpot": sum(tpos) / len(tpos) if tpos else 0.0,
            "p50_tpot": percentile(tpos, 50),
            "p95_tpot": percentile(tpos, 95),
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0.0,
            "p50_latency": percentile(latencies, 50),
            "p95_latency": percentile(latencies, 95),
            "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0.0,
            "total_tokens": sum(s.tokens_generated for s in self.snapshots),
        }

    def export_csv(self, filepath: str):
        """Export snapshots to CSV file."""
        import csv
        if not self.snapshots:
            return

        fieldnames = list(self.snapshots[0].to_dict().keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for snapshot in self.snapshots:
                writer.writerow(snapshot.to_dict())
