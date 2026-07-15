"""
latency.py
Profiles end-to-end detection and response decision latency,
and compares it against industry baselines.
"""

import time
import numpy as np

def measure_pipeline_latency(model, orchestrator, X, n_sample=200):
    """
    Measures per-entity inference + SOAR playbook decision latency.
    """
    X_sample = X[:n_sample]
    latencies = []
    
    for i in range(len(X_sample)):
        start = time.perf_counter()
        
        # 1. Detection (Inference)
        row = X_sample[i:i+1]
        score = float(model.score(row)[0])
        
        # 2. SOAR decision logic
        # Mock entity ID and technique for testing
        orchestrator.handle(f"10.0.0.{i%255}", score, "T1071")
        
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
        
    return {
        "end_to_end_latency_ms": {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99))
        },
        "baseline_source": "IBM Cost of a Data Breach Report 2025",
        "baseline_mtti_days_global": 181,
        "baseline_mtti_days_india_range": (201, 211),
        "baseline_mttc_days": 60
    }

def format_comparison(result: dict) -> str:
    mtti_india = result['baseline_mtti_days_india_range']
    p95_ms = result['end_to_end_latency_ms']['p95']
    return (
        f"Latency Profiling Results (Inference + SOAR decision):\n"
        f"  Mean latency: {result['end_to_end_latency_ms']['mean']:.2f} ms\n"
        f"  p95 latency:  {p95_ms:.2f} ms\n\n"
        f"Baseline ({result['baseline_source']}):\n"
        f"  Global MTTI (Mean Time to Identify): {result['baseline_mtti_days_global']} days\n"
        f"  India Adjusted MTTI: {mtti_india[0]} - {mtti_india[1]} days\n"
        f"  Global MTTC (Mean Time to Contain): {result['baseline_mttc_days']} days\n\n"
        f"CAVEAT (Say this to the judges):\n"
        f"  This prototype shrinks the model-inference and analyst-decision latency down to\n"
        f"  ~{p95_ms:.0f}ms. It does NOT claim to eliminate the full 200-day dwell time, as\n"
        f"  log-ingestion, SIEM processing, and pipeline delays are out of scope here."
    )
