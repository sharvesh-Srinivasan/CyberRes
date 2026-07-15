"""
data_loader.py
Loads UNSW-NB15 training/testing CSVs and prepares feature matrices.

Dataset source (download before running):
  Kaggle: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
  Files needed: UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv
  Place both in the data/ directory.

UNSW-NB15 columns (relevant subset):
  dur, proto, service, state, spkts, dpkts, sbytes, dbytes, rate,
  sttl, dttl, sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit,
  swin, stcpb, dtcpb, dwin, tcprtt, synack, ackdat, smean, dmean,
  trans_depth, response_body_len, ct_srv_src, ct_state_ttl, ct_dst_ltm,
  ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, is_ftp_login,
  ct_ftp_cmd, ct_flw_http_mthd, ct_src_ltm, ct_srv_dst, is_sm_ips_ports,
  attack_cat, label   <- label: 0=normal, 1=attack. attack_cat: category name.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CATEGORICAL_COLS = ["proto", "service", "state"]
DROP_COLS = ["id", "attack_cat", "label"]

# The 9 UNSW-NB15 attack categories -> mapped to MITRE ATT&CK techniques
# in mitre_rag.py. Kept here so loader and RAG layer share one vocabulary.
ATTACK_CATEGORIES = [
    "Normal", "Fuzzers", "Analysis", "Backdoor", "DoS",
    "Exploits", "Generic", "Reconnaissance", "Shellcode", "Worms",
]


def load_raw(train_path: Path = None, test_path: Path = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = train_path or DATA_DIR / "UNSW_NB15_training-set.csv"
    test_path = test_path or DATA_DIR / "UNSW_NB15_testing-set.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Dataset not found. Download UNSW-NB15 training/testing CSVs and place them at:\n"
            f"  {train_path}\n  {test_path}\n"
            f"Source: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df.columns = [c.strip().lower() for c in train_df.columns]
    test_df.columns = [c.strip().lower() for c in test_df.columns]
    return train_df, test_df


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode proto/service/state, aligning columns between train/test."""
    combined = pd.concat([train_df, test_df], keys=["train", "test"])
    combined = pd.get_dummies(combined, columns=[c for c in CATEGORICAL_COLS if c in combined.columns])
    bool_cols = combined.select_dtypes(include="bool").columns
    combined[bool_cols] = combined[bool_cols].astype("int8")
    train_enc = combined.xs("train")
    test_enc = combined.xs("test")
    return train_enc, test_enc


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Returns X_train_normal (features for normal-only rows, for unsupervised training),
    X_test (all test features), y_test (binary labels), cat_test (attack_cat strings).
    """
    train_enc, test_enc = encode_categoricals(train_df.copy(), test_df.copy())

    y_train = train_df["label"].values if "label" in train_df.columns else None
    y_test = test_df["label"].values
    cat_test = test_df["attack_cat"].fillna("Normal").str.strip() if "attack_cat" in test_df.columns else None

    feature_cols = [c for c in train_enc.columns if c not in DROP_COLS]
    # keep only numeric columns (defensive against stray non-numeric leftovers)
    X_train_full = train_enc[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    X_test = test_enc[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    X_test = X_test.reindex(columns=X_train_full.columns, fill_value=0)  # align columns exactly

    if y_train is not None:
        X_train_normal = X_train_full[y_train == 0]
    else:
        X_train_normal = X_train_full

    return X_train_normal, X_train_full, X_test, y_test, cat_test


def make_synthetic_dataset(n_train=4000, n_test=1500, seed=42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synthetic UNSW-NB15-shaped dataset for smoke-testing the pipeline
    when the real dataset isn't available locally. NOT for reporting
    real benchmark numbers -- swap in the real CSVs for that.
    """
    rng = np.random.default_rng(seed)

    def gen(n, attack_frac):
        n_attack = int(n * attack_frac)
        n_normal = n - n_attack
        rows = []
        for _ in range(n_normal):
            rows.append(_normal_row(rng))
        cats = rng.choice(ATTACK_CATEGORIES[1:], size=n_attack)
        for cat in cats:
            rows.append(_attack_row(rng, cat))
        df = pd.DataFrame(rows)
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return gen(n_train, 0.35), gen(n_test, 0.35)


def _normal_row(rng):
    return {
        "dur": rng.exponential(0.5), "proto": rng.choice(["tcp", "udp"]),
        "service": rng.choice(["http", "dns", "-", "ftp"]), "state": rng.choice(["FIN", "CON"]),
        "spkts": rng.poisson(10), "dpkts": rng.poisson(10),
        "sbytes": rng.normal(500, 100), "dbytes": rng.normal(500, 100),
        "rate": rng.normal(20, 5), "sttl": 64, "dttl": 64,
        "sload": rng.normal(1000, 200), "dload": rng.normal(1000, 200),
        "ct_srv_src": rng.integers(1, 5), "ct_dst_ltm": rng.integers(1, 5),
        "ct_src_dport_ltm": rng.integers(1, 5), "attack_cat": "Normal", "label": 0,
    }


def _attack_row(rng, cat):
    # exaggerated feature shifts per category, just enough to be separable for a smoke test
    shift = {"DoS": 8, "Exploits": 4, "Reconnaissance": 6, "Generic": 3,
             "Fuzzers": 5, "Analysis": 3, "Backdoor": 4, "Shellcode": 5, "Worms": 6}.get(cat, 3)
    return {
        "dur": rng.exponential(0.1), "proto": rng.choice(["tcp", "udp"]),
        "service": rng.choice(["http", "-", "ssh"]), "state": rng.choice(["REQ", "INT"]),
        "spkts": rng.poisson(10 * shift), "dpkts": rng.poisson(2),
        "sbytes": rng.normal(500 * shift, 300), "dbytes": rng.normal(50, 20),
        "rate": rng.normal(20 * shift, 10), "sttl": rng.choice([32, 128, 255]), "dttl": 64,
        "sload": rng.normal(1000 * shift, 400), "dload": rng.normal(100, 50),
        "ct_srv_src": rng.integers(5, 20), "ct_dst_ltm": rng.integers(5, 20),
        "ct_src_dport_ltm": rng.integers(5, 20), "attack_cat": cat, "label": 1,
    }
