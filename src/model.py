"""
model.py
Hybrid anomaly detector: MLP-based Autoencoder (sklearn, no GPU deps) for
dimensionality reduction -> Isolation Forest on the latent representation.

Design rationale (state this in the deck):
  - Autoencoder trained ONLY on normal-labeled traffic -> preserves the
    "no signature matching" claim (no attack labels used at train time).
  - Feeding the AE's bottleneck (latent) representation into Isolation
    Forest, instead of running IF on raw features independently, avoids
    redundant computation and lets IF operate on a denser, more
    separable representation of "normal" structure.
  - Pure sklearn (no torch/tensorflow) keeps the dependency footprint
    small enough to install and run in minutes on any judge's laptop.

Fixes applied after testing against the REAL UNSW-NB15 dataset (v2 -- see
CONTEXT.md Section 8 for the full writeup of what broke and why):
  1. Auto log1p on skewed non-negative columns (byte/rate/load features
     span 5+ orders of magnitude; RobustScaler alone doesn't fix that).
  2. Score normalization constants (IF raw-score range, reconstruction-error
     range) are now FIXED at fit time from the training distribution, not
     recomputed per inference batch. The old version re-derived min/max from
     whatever batch was passed to .score() -- not reproducible, and
     meaningless for scoring a single entity in production. This was the
     main bug: it didn't crash, it silently produced a batch-relative
     ranking that looked fine on a balanced synthetic smoke test and
     collapsed on real, imbalanced data.
  3. threshold=0.5 was an arbitrary constant unrelated to the actual score
     distribution (real scores cluster in [0, 0.3], not [0, 1]) -- this
     alone cut real-data recall to ~0.1%. Replaced with
     calibrate_threshold(), which picks a threshold from the TRAINING
     (normal-only) score distribution at a target false-positive rate --
     the standard way to set an operating point for an unsupervised
     detector when you have no labeled attacks to tune against.
"""

from dataclasses import dataclass
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


@dataclass
class HybridAnomalyDetector:
    latent_dim: int = 16
    hidden_dims: tuple = (64, 32)
    contamination: float = 0.05
    random_state: int = 42
    iforest_weight: float = 0.6
    recon_weight: float = 0.4

    def __post_init__(self):
        self.scaler = RobustScaler()
        layer_sizes = self.hidden_dims + (self.latent_dim,) + self.hidden_dims[::-1]
        self._bottleneck_index = len(self.hidden_dims)
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=self.random_state,
        )
        self.iforest = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._log_cols = None
        self._raw_lo = self._raw_hi = None
        self._recon_lo = self._recon_hi = None
        self._train_scores = None
        self.threshold_ = 0.5

    def _apply_log(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        if self._log_cols is not None and len(self._log_cols):
            X[:, self._log_cols] = np.log1p(np.clip(X[:, self._log_cols], 0, None))
        return X

    def fit(self, X_normal: np.ndarray):
        X_normal = np.asarray(X_normal, dtype=float)

        col_min = X_normal.min(axis=0)
        nonneg = col_min >= 0
        with np.errstate(all="ignore"):
            mean = X_normal.mean(axis=0)
            std = X_normal.std(axis=0) + 1e-9
            skew = np.mean(((X_normal - mean) / std) ** 3, axis=0)
        self._log_cols = np.where(nonneg & (np.abs(skew) > 2))[0]

        Xl = self._apply_log(X_normal)
        Xs = self.scaler.fit_transform(Xl)
        self.autoencoder.fit(Xs, Xs)
        latent = self._encode(Xs)
        self.iforest.fit(latent)

        raw_train = self.iforest.decision_function(latent)
        recon_train = self.autoencoder.predict(Xs)
        recon_err_train = np.mean((Xs - recon_train) ** 2, axis=1)
        self._raw_lo, self._raw_hi = np.percentile(raw_train, [1, 99])
        self._recon_lo, self._recon_hi = np.percentile(recon_err_train, [1, 99])

        self._train_scores = self._combine(raw_train, recon_err_train)
        self.calibrate_threshold()
        return self

    def _encode(self, Xs: np.ndarray) -> np.ndarray:
        activations = Xs
        for i in range(self._bottleneck_index + 1):
            W, b = self.autoencoder.coefs_[i], self.autoencoder.intercepts_[i]
            activations = activations @ W + b
            if i < self._bottleneck_index:
                activations = np.maximum(activations, 0)
        return activations

    def _combine(self, raw: np.ndarray, recon_err: np.ndarray) -> np.ndarray:
        iforest_norm = 1 - (raw - self._raw_lo) / (self._raw_hi - self._raw_lo + 1e-9)
        recon_norm = (recon_err - self._recon_lo) / (self._recon_hi - self._recon_lo + 1e-9)
        combined = self.iforest_weight * iforest_norm + self.recon_weight * recon_norm
        return np.clip(combined, 0, 1)

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Xl = self._apply_log(X)
        Xs = self.scaler.transform(Xl)
        latent = self._encode(Xs)
        raw = self.iforest.decision_function(latent)
        recon = self.autoencoder.predict(Xs)
        recon_err = np.mean((Xs - recon) ** 2, axis=1)
        return self._combine(raw, recon_err)

    def calibrate_threshold(self, target_fpr: float = 0.05) -> float:
        """Set the operating threshold from the TRAINING (normal-only) score
        distribution at a target false-positive rate. Call again with a
        different target_fpr to change the recall/FPR trade-off without
        retraining."""
        self.threshold_ = float(np.percentile(self._train_scores, 100 * (1 - target_fpr)))
        return self.threshold_

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        t = self.threshold_ if threshold is None else threshold
        return (self.score(X) >= t).astype(int)

    def explain_anomaly(self, X_row: np.ndarray, feature_names: list[str],
                        top_k: int = 5) -> dict:
        """
        Return the top-k features that contributed most to the anomaly score
        for a single entity row. Uses per-feature reconstruction error
        (squared difference between input and autoencoder reconstruction)
        as the contribution metric.

        Args:
            X_row: shape (1, n_features) — a single entity's feature vector.
            feature_names: list of column names matching X_row columns.
            top_k: number of top contributing features to return.

        Returns:
            dict with keys: anomaly_score, threshold, is_anomaly,
            top_features (list of {feature, reconstruction_error, normalized_contribution}).
        """
        X_row = np.asarray(X_row, dtype=float)
        Xl = self._apply_log(X_row)
        Xs = self.scaler.transform(Xl)

        # Per-feature reconstruction error
        recon = self.autoencoder.predict(Xs)
        per_feature_err = (Xs - recon) ** 2  # shape (1, n_features)
        per_feature_err = per_feature_err[0]  # flatten to (n_features,)

        total_err = per_feature_err.sum() + 1e-9
        n_features = min(len(feature_names), len(per_feature_err))

        # Rank features by reconstruction error contribution
        top_indices = np.argsort(per_feature_err[:n_features])[::-1][:top_k]
        top_features = [
            {
                "feature": feature_names[i],
                "reconstruction_error": float(per_feature_err[i]),
                "normalized_contribution": float(per_feature_err[i] / total_err),
            }
            for i in top_indices
        ]

        score = float(self.score(X_row)[0])
        return {
            "anomaly_score": score,
            "threshold": float(self.threshold_),
            "is_anomaly": score >= self.threshold_,
            "top_features": top_features,
        }

