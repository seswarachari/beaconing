#!/usr/bin/env python3
"""
kaggle_baseline_loader.py
=========================
Load and preprocess the Kaggle CICIDS2017 dataset for benign traffic
baseline analysis. If the real dataset is unavailable, generates a
CICIDS2017-like synthetic fallback with realistic distributions.

PURPOSE:
    This module supplements benign traffic diversity for the C2 Beaconing
    Detection Engine. It provides a rich set of benign flow features to
    help the model learn what "normal" looks like.

    IMPORTANT: This is NOT a source of C2 ground truth. All data from
    this loader is benign (is_malicious=0).

Outputs:
    A DataFrame of benign flow features in unified format, suitable for
    feature engineering and ML pipeline ingestion.

Usage:
    python -m src.kaggle_baseline_loader
    python src/kaggle_baseline_loader.py

Author: C2 Beaconing Detection Engine Team
"""

import os
import glob
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "kaggle_reference"

# Columns expected in CICIDS2017 dataset (case-insensitive matching)
CICIDS_LABEL_COL = "Label"
CICIDS_BENIGN_LABEL = "BENIGN"

# Unified output schema
OUTPUT_COLUMNS = [
    "flow_duration",
    "fwd_packets",
    "bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "fwd_iat_mean",
    "fwd_iat_std",
    "bwd_iat_mean",
    "bwd_iat_std",
    "dst_port",
    "protocol",
]

# Derived feature columns appended to the output
DERIVED_COLUMNS = [
    "mean_interval",
    "std_interval",
    "interval_cov",
    "mean_bytes",
    "bytes_cov",
]

# Column mapping: CICIDS2017 column names → our unified names
# CICIDS2017 has several naming conventions depending on the file;
# we handle the most common variants.
CICIDS_COLUMN_MAP = {
    # Flow Duration
    "Flow Duration": "flow_duration",
    " Flow Duration": "flow_duration",
    # Packet counts
    "Total Fwd Packets": "fwd_packets",
    " Total Fwd Packets": "fwd_packets",
    "Total Backward Packets": "bwd_packets",
    " Total Backward Packets": "bwd_packets",
    # Byte counts
    "Total Length of Fwd Packets": "total_fwd_bytes",
    " Total Length of Fwd Packets": "total_fwd_bytes",
    "Total Length of Bwd Packets": "total_bwd_bytes",
    " Total Length of Bwd Packets": "total_bwd_bytes",
    # Inter-arrival times
    "Fwd IAT Mean": "fwd_iat_mean",
    " Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    " Fwd IAT Std": "fwd_iat_std",
    "Bwd IAT Mean": "bwd_iat_mean",
    " Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    " Bwd IAT Std": "bwd_iat_std",
    # Port and protocol
    "Destination Port": "dst_port",
    " Destination Port": "dst_port",
    "Protocol": "protocol",
    " Protocol": "protocol",
    # Label
    "Label": "label",
    " Label": "label",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Real CICIDS2017 loader
# ---------------------------------------------------------------------------

def _find_csv_files(data_dir: Path) -> list:
    """Find all CSV files in the given directory (non-recursive)."""
    csv_files = sorted(glob.glob(str(data_dir / "*.csv")))
    csv_files += sorted(glob.glob(str(data_dir / "*.CSV")))
    return csv_files


def _load_real_cicids(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Attempt to load real CICIDS2017 dataset from data_dir.

    Returns:
        DataFrame of benign flows in unified format, or None if no
        valid CSV files are found.
    """
    csv_files = _find_csv_files(data_dir)
    if not csv_files:
        logger.info("No CSV files found in %s", data_dir)
        return None

    logger.info("Found %d CSV file(s) in %s", len(csv_files), data_dir)
    all_frames = []

    for csv_path in csv_files:
        logger.info("Loading: %s", os.path.basename(csv_path))
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", csv_path, exc)
            continue

        # Rename columns using our mapping
        df = df.rename(columns=CICIDS_COLUMN_MAP)

        # Check if label column exists after rename
        if "label" not in df.columns:
            logger.warning("No 'Label' column found in %s — skipping",
                           csv_path)
            continue

        # Filter to BENIGN rows only
        benign_mask = df["label"].astype(str).str.strip().str.upper() == "BENIGN"
        benign_df = df[benign_mask].copy()
        logger.info("  Benign rows: %d / %d total", len(benign_df), len(df))

        if len(benign_df) == 0:
            continue

        # Select only the columns we need (that exist in this file)
        available_cols = [c for c in OUTPUT_COLUMNS if c in benign_df.columns]
        benign_df = benign_df[available_cols].copy()

        all_frames.append(benign_df)

    if not all_frames:
        logger.info("No benign data extracted from CSV files")
        return None

    combined = pd.concat(all_frames, ignore_index=True)
    logger.info("Total benign rows from real data: %d", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Synthetic fallback generator
# ---------------------------------------------------------------------------

def _generate_synthetic_fallback(num_rows: int = 500) -> pd.DataFrame:
    """
    Generate a synthetic CICIDS2017-like benign flow dataset.

    This fallback uses realistic statistical distributions modeled after
    actual CICIDS2017 benign traffic patterns. It supplements benign
    diversity when the real dataset is unavailable.

    NOTE: This is a synthetic approximation — NOT real captured traffic.
    All rows are benign (is_malicious=0).

    Args:
        num_rows: Number of synthetic benign flow records to generate.

    Returns:
        DataFrame with unified benign flow features.
    """
    rng = np.random.default_rng(SEED)
    logger.info("Generating %d synthetic benign flows (CICIDS2017-like)", num_rows)

    # --- Flow duration: lognormal(mean=5, sigma=2) seconds ---
    # Converted to microseconds (CICIDS2017 uses μs for flow duration)
    flow_duration_s = np.exp(rng.normal(5, 2, size=num_rows))
    flow_duration_s = np.clip(flow_duration_s, 0.001, 3600)  # 1ms to 1hr
    flow_duration = (flow_duration_s * 1e6).astype(int)  # microseconds

    # --- Packet counts: Poisson distribution ---
    fwd_packets = np.maximum(1, rng.poisson(lam=15, size=num_rows))
    bwd_packets = np.maximum(1, rng.poisson(lam=12, size=num_rows))

    # --- Byte counts: lognormal(mean=8, sigma=2) ---
    total_fwd_bytes = np.maximum(
        1, np.exp(rng.normal(8, 2, size=num_rows)).astype(int)
    )
    total_bwd_bytes = np.maximum(
        1, np.exp(rng.normal(7.5, 2.2, size=num_rows)).astype(int)
    )

    # --- Inter-arrival times (IAT) ---
    # Forward IAT mean: lognormal(mean=3, sigma=1.5) milliseconds
    fwd_iat_mean = np.maximum(
        0.01, np.exp(rng.normal(3, 1.5, size=num_rows))
    )

    # Forward IAT std: high relative to mean (CoV typically 0.5-3.0)
    # This ensures benign traffic shows HIGH timing variability
    fwd_iat_cov = rng.uniform(0.5, 3.0, size=num_rows)
    fwd_iat_std = fwd_iat_mean * fwd_iat_cov

    # Backward IAT (similar distribution, slightly different params)
    bwd_iat_mean = np.maximum(
        0.01, np.exp(rng.normal(3.2, 1.6, size=num_rows))
    )
    bwd_iat_cov = rng.uniform(0.5, 3.0, size=num_rows)
    bwd_iat_std = bwd_iat_mean * bwd_iat_cov

    # --- Destination port: weighted random selection ---
    port_choices = [80, 443, 8080, 53, 22, 3389]
    port_weights = [0.20, 0.40, 0.10, 0.10, 0.08, 0.05]

    # 7% chance of a random high port (1024-65535)
    high_port_prob = 0.07
    port_weights_adj = [w * (1 - high_port_prob) for w in port_weights]

    dst_port = np.zeros(num_rows, dtype=int)
    for i in range(num_rows):
        if rng.random() < high_port_prob:
            dst_port[i] = rng.integers(1024, 65536)
        else:
            dst_port[i] = rng.choice(port_choices, p=[
                w / sum(port_weights_adj) for w in port_weights_adj
            ])

    # --- Protocol: 6=TCP, 17=UDP, 1=ICMP (CICIDS2017 uses numeric) ---
    protocol = rng.choice(
        [6, 17, 1],
        size=num_rows,
        p=[0.80, 0.18, 0.02],
    )

    df = pd.DataFrame({
        "flow_duration": flow_duration,
        "fwd_packets": fwd_packets,
        "bwd_packets": bwd_packets,
        "total_fwd_bytes": total_fwd_bytes,
        "total_bwd_bytes": total_bwd_bytes,
        "fwd_iat_mean": np.round(fwd_iat_mean, 4),
        "fwd_iat_std": np.round(fwd_iat_std, 4),
        "bwd_iat_mean": np.round(bwd_iat_mean, 4),
        "bwd_iat_std": np.round(bwd_iat_std, 4),
        "dst_port": dst_port,
        "protocol": protocol,
    })

    logger.info("Synthetic fallback generated: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Derived feature computation
# ---------------------------------------------------------------------------

def _compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features that match the feature engineering format
    used by the beacon detection pipeline.

    Derived columns:
        - mean_interval: from fwd_iat_mean (proxy for beacon interval)
        - std_interval: from fwd_iat_std
        - interval_cov: std/mean (CoV), handles division by zero
        - mean_bytes: average bytes per packet
        - bytes_cov: coefficient of variation for byte sizes

    Args:
        df: DataFrame with base flow features.

    Returns:
        DataFrame with derived features appended.
    """
    result = df.copy()

    # --- Interval features ---
    result["mean_interval"] = result["fwd_iat_mean"]
    result["std_interval"] = result["fwd_iat_std"]

    # Coefficient of Variation = std / mean (handle div-by-zero)
    result["interval_cov"] = np.where(
        result["mean_interval"] > 0,
        result["std_interval"] / result["mean_interval"],
        0.0,
    )
    result["interval_cov"] = result["interval_cov"].round(6)

    # --- Byte size features ---
    total_packets = result["fwd_packets"] + result["bwd_packets"]
    total_bytes = result["total_fwd_bytes"] + result["total_bwd_bytes"]

    result["mean_bytes"] = np.where(
        total_packets > 0,
        total_bytes / total_packets,
        0.0,
    )
    result["mean_bytes"] = result["mean_bytes"].round(2)

    # Approximate bytes_cov from the ratio of fwd/bwd bytes variance
    # Since we don't have per-packet data, we estimate from fwd vs bwd
    bytes_std = np.abs(
        result["total_fwd_bytes"] - result["total_bwd_bytes"]
    ) / np.sqrt(2)
    result["bytes_cov"] = np.where(
        result["mean_bytes"] > 0,
        bytes_std / (result["mean_bytes"] * total_packets),
        0.0,
    )
    result["bytes_cov"] = result["bytes_cov"].clip(0, 10).round(6)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_kaggle_baseline(
    data_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load CICIDS2017 benign traffic baseline with derived features.

    Behavior:
        1. Check if any CSV exists in data_dir (default: data/kaggle_reference/)
        2. If found: load, filter to BENIGN, extract relevant columns
        3. If not found: generate a synthetic fallback (~500 rows)
        4. Compute derived features for ML compatibility

    NOTE: This provides BENIGN traffic diversity only — NOT C2 ground truth.
    All returned rows represent normal/benign network activity.

    Args:
        data_dir: Path to directory containing CICIDS2017 CSV files.
                  Defaults to data/kaggle_reference/.

    Returns:
        pd.DataFrame with unified flow features + derived columns.
    """
    if data_dir is None:
        data_path = DEFAULT_DATA_DIR
    else:
        data_path = Path(data_dir)

    # Attempt to load real dataset
    df = _load_real_cicids(data_path)

    if df is None:
        logger.info("Falling back to synthetic CICIDS2017-like data")
        df = _generate_synthetic_fallback(num_rows=500)

    # Ensure all expected base columns exist (fill missing with 0)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = 0
            logger.warning("Column '%s' not found — filled with zeros", col)

    # Clean numeric columns (handle NaN, Inf)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    # Compute derived features
    df = _compute_derived_features(df)

    logger.info("Baseline loaded: %d rows, %d columns", len(df), len(df.columns))
    return df


def get_benign_feature_matrix(
    data_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Get a NumPy feature matrix of benign traffic, suitable for ML training.

    Extracts the derived features used by the beacon detection model:
        [mean_interval, std_interval, interval_cov, mean_bytes, bytes_cov]

    NOTE: This provides BENIGN traffic features ONLY — NOT C2 ground truth.
    Use this to supplement benign diversity in the training set.

    Args:
        data_dir: Path to directory containing CICIDS2017 CSV files.

    Returns:
        np.ndarray of shape (n_samples, n_features).
    """
    df = load_kaggle_baseline(data_dir)

    feature_cols = DERIVED_COLUMNS
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing derived feature columns: {missing}. "
            "Ensure _compute_derived_features() was called."
        )

    feature_matrix = df[feature_cols].values.astype(np.float64)

    # Final NaN/Inf cleanup for ML safety
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info("Feature matrix shape: %s", feature_matrix.shape)
    return feature_matrix


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("C2 Beaconing Detection Engine — Kaggle Baseline Loader")
    print("=" * 70)
    print()
    print("NOTE: This module provides BENIGN traffic diversity only.")
    print("      It does NOT contain C2 ground truth data.")
    print()

    # Load baseline
    df = load_kaggle_baseline()

    print("-" * 70)
    print("BASELINE SUMMARY")
    print("-" * 70)
    print(f"\nTotal rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    print("\nBase feature statistics:")
    for col in OUTPUT_COLUMNS:
        if col in df.columns and df[col].dtype in [np.float64, np.int64, float, int]:
            print(f"  {col:25s}  mean={df[col].mean():12.2f}  "
                  f"std={df[col].std():12.2f}  "
                  f"min={df[col].min():12.2f}  "
                  f"max={df[col].max():12.2f}")

    print("\nDerived feature statistics:")
    for col in DERIVED_COLUMNS:
        if col in df.columns:
            print(f"  {col:25s}  mean={df[col].mean():12.4f}  "
                  f"std={df[col].std():12.4f}  "
                  f"min={df[col].min():12.4f}  "
                  f"max={df[col].max():12.4f}")

    print("\nPort distribution:")
    for port, count in df["dst_port"].value_counts().head(10).items():
        print(f"  port {port}: {count:,}")

    # Feature matrix
    feature_matrix = get_benign_feature_matrix()
    print(f"\nFeature matrix shape: {feature_matrix.shape}")
    print(f"Feature matrix dtype: {feature_matrix.dtype}")

    print("\n" + "=" * 70)
    print("Baseline loading complete.")
    print("=" * 70)
