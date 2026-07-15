"""
calendar_features.py

Differentiator: conditions the anomaly baseline on institutional calendar
phase (exam period / admission season / fiscal year-end / normal), since
generic rolling-window UEBA flags legitimate bulk-traffic periods as
anomalies -- which is exactly why SOC analysts abandon these tools
(alert fatigue). This is the feature that the ablation study in
evaluate.py measures.

UNSW-NB15 has no real timestamps usable as wall-clock dates, so for the
synthetic smoke test we assign synthetic institutional-calendar phases
to rows and inject legitimate high-volume traffic during "exam_period"
to prove the conditioning actually suppresses false positives on
legitimate bursts. When you swap in real deployment data with real
timestamps, replace `assign_calendar_phase` with a lookup against your
institution's actual academic/fiscal calendar.
"""

import numpy as np
import pandas as pd

PHASES = ["normal", "exam_period", "admission_season", "fiscal_year_end"]

# One-hot phase weight multipliers used only for the synthetic injection
# in inject_legitimate_bursts(). Not used at inference time on real data.
_BURST_MULTIPLIER = {"exam_period": 6.0, "admission_season": 3.0,
                      "fiscal_year_end": 4.0, "normal": 1.0}


def assign_calendar_phase(n_rows: int, seed: int = 7) -> pd.Series:
    """Randomly assign calendar phases with realistic weighting (most days are 'normal')."""
    rng = np.random.default_rng(seed)
    weights = [0.75, 0.08, 0.10, 0.07]  # ~75% of the year is calendar-normal
    return pd.Series(rng.choice(PHASES, size=n_rows, p=weights), name="calendar_phase")


def add_calendar_features(X: pd.DataFrame, phase: pd.Series) -> pd.DataFrame:
    """One-hot encode calendar phase and append to feature matrix."""
    phase_dummies = pd.get_dummies(phase, prefix="phase")
    for col in [f"phase_{p}" for p in PHASES]:
        if col not in phase_dummies.columns:
            phase_dummies[col] = 0
    phase_dummies = phase_dummies.reindex(sorted(phase_dummies.columns), axis=1)
    return pd.concat([X.reset_index(drop=True), phase_dummies.reset_index(drop=True)], axis=1)


def inject_legitimate_bursts(X: pd.DataFrame, phase: pd.Series, volume_cols: list[str]) -> pd.DataFrame:
    """
    Simulates legitimate high-volume traffic during institutional peak periods
    (e.g. board-exam result uploads, EOFY bulk transfers). This is what a
    calendar-naive baseline will misclassify as anomalous, and what
    calendar-conditioning is supposed to correctly suppress.
    """
    X = X.copy()
    for p, mult in _BURST_MULTIPLIER.items():
        if mult == 1.0:
            continue
        mask = (phase == p).values
        for col in volume_cols:
            if col in X.columns:
                X.loc[mask, col] = X.loc[mask, col] * mult
    return X
