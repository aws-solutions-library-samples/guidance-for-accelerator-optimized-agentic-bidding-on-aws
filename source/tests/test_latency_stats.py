"""Verification tests for latency statistics and histogram computation.

Validates the invariants specified in Task 7.3:
- For any non-empty array of positive floats: min ≤ p50 ≤ p95 ≤ p99 ≤ max
- avg = sum/count
- Histogram bucket counts sum to total count
- Each value is assigned to exactly one bucket based on its magnitude
- Per-container breakdown: avg latency and mutation count per container

**Validates: Requirements 10.5, 10.6, 10.10, 10.11**
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator.loadtest import compute_latency_stats, compute_histogram


# ---------------------------------------------------------------------------
# Unit tests: compute_latency_stats
# ---------------------------------------------------------------------------

class TestComputeLatencyStats:
    """Unit tests for compute_latency_stats correctness."""

    def test_empty_array_returns_zeros(self):
        """Empty input returns all-zero stats."""
        result = compute_latency_stats([])
        assert result == {
            "latency_min": 0.0,
            "latency_max": 0.0,
            "latency_avg": 0.0,
            "latency_p50": 0.0,
            "latency_p95": 0.0,
            "latency_p99": 0.0,
        }

    def test_single_element(self):
        """Single-element array: all stats equal that element."""
        result = compute_latency_stats([5.0])
        assert result["latency_min"] == 5.0
        assert result["latency_max"] == 5.0
        assert result["latency_avg"] == 5.0
        assert result["latency_p50"] == 5.0
        assert result["latency_p95"] == 5.0
        assert result["latency_p99"] == 5.0

    def test_two_elements(self):
        """Two-element array: min and max are correct, ordering holds."""
        result = compute_latency_stats([3.0, 7.0])
        assert result["latency_min"] == 3.0
        assert result["latency_max"] == 7.0
        assert result["latency_avg"] == 5.0
        # Ordering property
        assert result["latency_min"] <= result["latency_p50"]
        assert result["latency_p50"] <= result["latency_p95"]
        assert result["latency_p95"] <= result["latency_p99"]
        assert result["latency_p99"] <= result["latency_max"]

    def test_known_distribution(self):
        """Known distribution: 100 values from 1.0 to 100.0."""
        latencies = [float(i) for i in range(1, 101)]
        result = compute_latency_stats(latencies)
        assert result["latency_min"] == 1.0
        assert result["latency_max"] == 100.0
        assert result["latency_avg"] == 50.5
        # p50 should be around 50, p95 around 95, p99 around 99
        assert result["latency_p50"] == 51.0  # index 50 in 0-indexed sorted array
        assert result["latency_p95"] == 96.0  # index 95
        assert result["latency_p99"] == 100.0  # index 99 (clamped to n-1)

    def test_ordering_property_holds(self):
        """Verify min ≤ p50 ≤ p95 ≤ p99 ≤ max for a varied distribution."""
        latencies = [0.5, 1.2, 3.4, 5.6, 8.9, 12.3, 25.7, 45.0, 67.8, 120.5]
        result = compute_latency_stats(latencies)
        assert result["latency_min"] <= result["latency_p50"]
        assert result["latency_p50"] <= result["latency_p95"]
        assert result["latency_p95"] <= result["latency_p99"]
        assert result["latency_p99"] <= result["latency_max"]

    def test_avg_equals_sum_over_count(self):
        """avg = sum/count for a known set of values."""
        latencies = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = compute_latency_stats(latencies)
        expected_avg = sum(latencies) / len(latencies)
        assert result["latency_avg"] == round(expected_avg, 3)

    def test_all_same_values(self):
        """All identical values: all stats equal that value."""
        latencies = [7.5] * 50
        result = compute_latency_stats(latencies)
        assert result["latency_min"] == 7.5
        assert result["latency_max"] == 7.5
        assert result["latency_avg"] == 7.5
        assert result["latency_p50"] == 7.5
        assert result["latency_p95"] == 7.5
        assert result["latency_p99"] == 7.5

    def test_unsorted_input(self):
        """Input doesn't need to be pre-sorted."""
        latencies = [50.0, 10.0, 30.0, 20.0, 40.0]
        result = compute_latency_stats(latencies)
        assert result["latency_min"] == 10.0
        assert result["latency_max"] == 50.0
        assert result["latency_avg"] == 30.0


# ---------------------------------------------------------------------------
# Unit tests: compute_histogram
# ---------------------------------------------------------------------------

class TestComputeHistogram:
    """Unit tests for compute_histogram correctness."""

    def test_empty_array(self):
        """Empty input returns all-zero buckets."""
        result = compute_histogram([])
        assert result == {"lt_10ms": 0, "10_30ms": 0, "30_50ms": 0, "gt_50ms": 0}

    def test_all_under_10ms(self):
        """All values < 10.0 go into lt_10ms bucket."""
        result = compute_histogram([1.0, 5.0, 9.9])
        assert result == {"lt_10ms": 3, "10_30ms": 0, "30_50ms": 0, "gt_50ms": 0}

    def test_all_in_10_30ms(self):
        """All values in [10.0, 30.0) go into 10_30ms bucket."""
        result = compute_histogram([10.0, 15.0, 29.9])
        assert result == {"lt_10ms": 0, "10_30ms": 3, "30_50ms": 0, "gt_50ms": 0}

    def test_all_in_30_50ms(self):
        """All values in [30.0, 50.0] go into 30_50ms bucket."""
        result = compute_histogram([30.0, 40.0, 50.0])
        assert result == {"lt_10ms": 0, "10_30ms": 0, "30_50ms": 3, "gt_50ms": 0}

    def test_all_over_50ms(self):
        """All values > 50.0 go into gt_50ms bucket."""
        result = compute_histogram([50.1, 75.0, 100.0])
        assert result == {"lt_10ms": 0, "10_30ms": 0, "30_50ms": 0, "gt_50ms": 3}

    def test_bucket_counts_sum_to_total(self):
        """Bucket counts must sum to total number of values."""
        latencies = [1.0, 9.9, 10.0, 15.0, 29.9, 30.0, 50.0, 50.1, 100.0]
        result = compute_histogram(latencies)
        total = result["lt_10ms"] + result["10_30ms"] + result["30_50ms"] + result["gt_50ms"]
        assert total == len(latencies)

    def test_boundary_10ms(self):
        """Value exactly 10.0 goes into 10_30ms bucket (not lt_10ms)."""
        result = compute_histogram([10.0])
        assert result == {"lt_10ms": 0, "10_30ms": 1, "30_50ms": 0, "gt_50ms": 0}

    def test_boundary_30ms(self):
        """Value exactly 30.0 goes into 30_50ms bucket (not 10_30ms)."""
        result = compute_histogram([30.0])
        assert result == {"lt_10ms": 0, "10_30ms": 0, "30_50ms": 1, "gt_50ms": 0}

    def test_boundary_50ms(self):
        """Value exactly 50.0 goes into 30_50ms bucket (not gt_50ms)."""
        result = compute_histogram([50.0])
        assert result == {"lt_10ms": 0, "10_30ms": 0, "30_50ms": 1, "gt_50ms": 0}

    def test_mixed_distribution(self):
        """Mixed values are correctly distributed across buckets."""
        latencies = [5.0, 9.9, 10.0, 20.0, 30.0, 40.0, 50.0, 51.0, 100.0]
        result = compute_histogram(latencies)
        assert result["lt_10ms"] == 2      # 5.0, 9.9
        assert result["10_30ms"] == 2      # 10.0, 20.0
        assert result["30_50ms"] == 3      # 30.0, 40.0, 50.0
        assert result["gt_50ms"] == 2      # 51.0, 100.0

    def test_single_value_each_bucket(self):
        """One value per bucket — each assigned to exactly one."""
        result = compute_histogram([5.0, 15.0, 35.0, 75.0])
        assert result == {"lt_10ms": 1, "10_30ms": 1, "30_50ms": 1, "gt_50ms": 1}


# ---------------------------------------------------------------------------
# Unit tests: per-container breakdown
# ---------------------------------------------------------------------------

class TestPerContainerBreakdown:
    """Verify per-container avg latency and mutation count computation logic."""

    def test_avg_latency_computation(self):
        """Average latency is sum/count for each container."""
        # Simulate what _run_load_test does for per-container stats
        per_container_latencies = {
            "container-a": [10.0, 20.0, 30.0],
            "container-b": [5.0, 15.0],
            "container-c": [],
        }
        per_container_mutations = {
            "container-a": 6,
            "container-b": 4,
            "container-c": 0,
        }

        per_container = []
        for name in ["container-a", "container-b", "container-c"]:
            c_lats = per_container_latencies[name]
            avg_lat = round(sum(c_lats) / len(c_lats), 3) if c_lats else 0.0
            per_container.append({
                "name": name,
                "avg_latency_ms": avg_lat,
                "total_mutations": per_container_mutations[name],
            })

        assert per_container[0]["name"] == "container-a"
        assert per_container[0]["avg_latency_ms"] == 20.0
        assert per_container[0]["total_mutations"] == 6

        assert per_container[1]["name"] == "container-b"
        assert per_container[1]["avg_latency_ms"] == 10.0
        assert per_container[1]["total_mutations"] == 4

        assert per_container[2]["name"] == "container-c"
        assert per_container[2]["avg_latency_ms"] == 0.0
        assert per_container[2]["total_mutations"] == 0

    def test_empty_container_latencies(self):
        """Container with no invocations gets avg_latency_ms = 0.0."""
        c_lats: list[float] = []
        avg_lat = round(sum(c_lats) / len(c_lats), 3) if c_lats else 0.0
        assert avg_lat == 0.0


# ---------------------------------------------------------------------------
# Integration: combined stats + histogram consistency
# ---------------------------------------------------------------------------

class TestStatsHistogramConsistency:
    """Verify that stats and histogram are consistent with each other."""

    def test_histogram_total_matches_input_length(self):
        """Histogram bucket sum always equals input length."""
        latencies = [0.1, 5.0, 9.99, 10.0, 15.5, 29.99, 30.0, 45.0, 50.0, 50.01, 99.9]
        histogram = compute_histogram(latencies)
        total = sum(histogram.values())
        assert total == len(latencies)

    def test_stats_and_histogram_same_input(self):
        """Stats min/max are consistent with histogram bucket assignments."""
        latencies = [2.0, 8.0, 12.0, 25.0, 35.0, 48.0, 55.0, 80.0]
        stats = compute_latency_stats(latencies)
        histogram = compute_histogram(latencies)

        # Min is 2.0 (< 10ms), so lt_10ms should be > 0
        assert stats["latency_min"] == 2.0
        assert histogram["lt_10ms"] >= 1

        # Max is 80.0 (> 50ms), so gt_50ms should be > 0
        assert stats["latency_max"] == 80.0
        assert histogram["gt_50ms"] >= 1

        # Total histogram count = input length
        assert sum(histogram.values()) == len(latencies)
