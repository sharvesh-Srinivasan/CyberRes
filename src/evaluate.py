"""
evaluate.py
Two things this reports, both needed for the deck/README/resume claim:
  1. Per-attack-category precision/recall/FPR (not one blended accuracy number)
  2. The calendar-conditioning ablation: FPR on legitimate high-volume
     "burst" traffic, WITH vs WITHOUT calendar-phase features. This is
     the differentiator metric.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def per_category_metrics(y_true_binary, y_pred_binary, categories) -> dict:
    """Overall + per-attack-category breakdown."""
    results = {}
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
    results["overall"] = {
        "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
        "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

    cats = np.array(categories)
    for cat in sorted(set(cats) - {"Normal"}):
        mask = cats == cat
        if mask.sum() == 0:
            continue
        yt = np.ones(mask.sum())  # all true attacks in this slice
        yp = y_pred_binary[mask]
        results[cat] = {
            "recall": recall_score(yt, yp, zero_division=0),  # detection rate for this category
            "n_samples": int(mask.sum()),
        }
    return results


def fpr_recall_tradeoff(detector, X_test: np.ndarray,
                         y_test, threshold_multipliers=None) -> list[dict]:
    """
    Compute precision/recall/FPR at multiple threshold calibration levels.
    Shows judges that recall is a calibration knob, not a hard ceiling.

    Args:
        detector: Fitted HybridAnomalyDetector.
        X_test: Feature matrix (same scale as training).
        y_test: Binary ground-truth labels.
        threshold_multipliers: List of floats to multiply detector.threshold_ by.
            < 1.0 = more sensitive (higher recall, higher FPR).
            > 1.0 = more specific (lower recall, lower FPR).

    Returns:
        List of dicts, one per operating point, sorted by FPR ascending.
    """
    if threshold_multipliers is None:
        threshold_multipliers = [0.7, 0.85, 1.0, 1.1, 1.25]

    scores = detector.score(X_test)
    y_true = np.array(y_test)
    results = []

    for mult in sorted(threshold_multipliers):
        t = detector.threshold_ * mult
        y_pred = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        results.append({
            "threshold_multiplier": round(mult, 2),
            "threshold": round(float(t), 4),
            "fpr":       round(fp / (fp + tn) if (fp + tn) else 0.0, 4),
            "recall":    round(tp / (tp + fn) if (tp + fn) else 0.0, 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        })

    return results



def calendar_ablation(detector_with_calendar, detector_without_calendar,
                       X_burst_with_cal, X_burst_no_cal, threshold=None) -> dict:
    """
    Both inputs represent the SAME legitimate high-volume traffic (e.g.
    exam-period result uploads). All of it is normal (label=0). A
    calendar-naive model should misflag much of it as anomalous;
    calendar-conditioning should suppress most of those false positives.
    Returns FPR for each, and the relative FPR reduction.
    """
    pred_with = detector_with_calendar.predict(X_burst_with_cal, threshold=threshold)
    pred_without = detector_without_calendar.predict(X_burst_no_cal, threshold=threshold)

    fpr_with = pred_with.mean()      # all samples are normal -> mean(pred==1) is the FPR
    fpr_without = pred_without.mean()
    reduction = (fpr_without - fpr_with) / fpr_without if fpr_without > 0 else 0.0

    return {
        "baseline_fpr": float(fpr_without),                        # canonical key
        "cal_fpr": float(fpr_with),                                 # canonical key
        "fpr_without_calendar_conditioning": float(fpr_without),    # verbose alias
        "fpr_with_calendar_conditioning": float(fpr_with),          # verbose alias
        "relative_fpr_reduction": float(reduction),
        "n_samples": len(X_burst_with_cal),
    }
