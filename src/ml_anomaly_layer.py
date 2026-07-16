"""
ml_anomaly_layer.py — Isolation Forest ML Anomaly Detection Layer

Provides an unsupervised machine-learning layer that complements the
deterministic beacon scoring engine.  An Isolation Forest is trained on
benign traffic features to learn the "normal" distribution; flows that
deviate significantly receive high anomaly scores.

Key capabilities:
    - Train on combined synthetic-benign + Kaggle-benign baselines
    - Score all flows with normalised anomaly scores (0–100)
    - Generate plain-text explanations for flagged flows by identifying
      which features deviate > 2σ from the benign baseline

Author: C2 Beaconing Detection Engine
License: MIT
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns used for ML training and scoring
# ---------------------------------------------------------------------------
ML_FEATURE_COLUMNS: List[str] = [
    'interval_cov',
    'mean_interval',
    'std_interval',
    'bytes_cov',
    'mean_bytes',
    'num_connections',
    'session_duration',
    'pct_connections_off_hours',
    'mean_connection_duration',
    'failed_connection_ratio',
    'payload_entropy',
]

# Human-readable descriptions for explainability
_FEATURE_DESCRIPTIONS: Dict[str, str] = {
    'interval_cov':              'timing regularity (lower = more regular)',
    'mean_interval':             'average time between connections',
    'std_interval':              'variability of connection timing',
    'bytes_cov':                 'payload size regularity',
    'mean_bytes':                'average payload size',
    'num_connections':           'total connection count',
    'session_duration':          'total session duration (seconds)',
    'pct_connections_off_hours': 'fraction of activity during off-hours',
    'mean_connection_duration':  'average connection duration',
    'failed_connection_ratio':   'fraction of failed connections',
    'payload_entropy':           'entropy of payload size distribution',
}


# =========================================================================
# Training Statistics (named dict structure)
# =========================================================================

def _compute_training_stats(
    benign_features: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """Compute per-feature mean and std from benign training data.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Mapping of feature_name → {'mean': ..., 'std': ...}.
    """
    stats: Dict[str, Dict[str, float]] = {}
    for col in ML_FEATURE_COLUMNS:
        if col in benign_features.columns:
            col_values = benign_features[col].dropna()
            stats[col] = {
                'mean': float(col_values.mean()),
                'std': float(col_values.std()) if len(col_values) > 1 else 0.0,
            }
        else:
            stats[col] = {'mean': 0.0, 'std': 1.0}
    return stats


# =========================================================================
# Training
# =========================================================================

def train_isolation_forest(
    benign_features_df: pd.DataFrame,
    kaggle_features_df: Optional[pd.DataFrame] = None,
    contamination: str = 'auto',
    n_estimators: int = 200,
    random_state: int = 42,
) -> Tuple[IsolationForest, StandardScaler, Dict[str, Dict[str, float]]]:
    """Train an Isolation Forest on benign traffic features.

    Parameters
    ----------
    benign_features_df : pd.DataFrame
        Synthetic benign features (must contain ML_FEATURE_COLUMNS).
    kaggle_features_df : pd.DataFrame, optional
        Optional supplementary benign features from Kaggle dataset.
        Will be concatenated with *benign_features_df* before training.
    contamination : str, optional
        Isolation Forest contamination parameter. Default 'auto'.
    n_estimators : int, optional
        Number of trees. Default 200.
    random_state : int, optional
        Random seed for reproducibility. Default 42.

    Returns
    -------
    (model, scaler, training_stats)
        - model: trained IsolationForest
        - scaler: fitted StandardScaler
        - training_stats: per-feature mean/std from benign data
    """
    # ------------------------------------------------------------------
    # Combine benign datasets
    # ------------------------------------------------------------------
    combined = benign_features_df.copy()

    if kaggle_features_df is not None and not kaggle_features_df.empty:
        logger.info(
            "Combining synthetic benign (%d rows) with Kaggle benign (%d rows).",
            len(benign_features_df), len(kaggle_features_df),
        )
        combined = pd.concat(
            [combined, kaggle_features_df], ignore_index=True,
        )

    # ------------------------------------------------------------------
    # Select and validate feature columns
    # ------------------------------------------------------------------
    available_cols = [c for c in ML_FEATURE_COLUMNS if c in combined.columns]
    missing_cols = set(ML_FEATURE_COLUMNS) - set(available_cols)
    if missing_cols:
        logger.warning(
            "Missing ML feature columns (will be filled with 0): %s",
            missing_cols,
        )
        for col in missing_cols:
            combined[col] = 0.0

    train_data = combined[ML_FEATURE_COLUMNS].copy()

    # Handle NaN / inf values
    train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_data.fillna(0.0, inplace=True)

    if train_data.empty:
        raise ValueError("No valid training data after preprocessing.")

    logger.info(
        "Training Isolation Forest on %d benign samples with %d features.",
        len(train_data), len(ML_FEATURE_COLUMNS),
    )

    # ------------------------------------------------------------------
    # Compute training statistics BEFORE scaling
    # ------------------------------------------------------------------
    training_stats = _compute_training_stats(train_data)

    # ------------------------------------------------------------------
    # Scale features
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(train_data)

    # ------------------------------------------------------------------
    # Train Isolation Forest
    # ------------------------------------------------------------------
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(scaled_data)

    logger.info("Isolation Forest training complete.")

    return model, scaler, training_stats


# =========================================================================
# Scoring
# =========================================================================

def score_flows(
    model: IsolationForest,
    scaler: StandardScaler,
    features_df: pd.DataFrame,
    training_stats: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Score all flows using the trained Isolation Forest.

    Parameters
    ----------
    model : IsolationForest
        Trained model from :func:`train_isolation_forest`.
    scaler : StandardScaler
        Fitted scaler from :func:`train_isolation_forest`.
    features_df : pd.DataFrame
        Features for all flows (benign + potentially malicious).
    training_stats : Dict
        Benign training statistics for reference.

    Returns
    -------
    pd.DataFrame
        Copy of *features_df* with ``ml_score`` column added (0–100,
        higher = more anomalous).
    """
    df = features_df.copy()

    if df.empty:
        logger.warning("Empty features DataFrame — nothing to score.")
        df['ml_score'] = pd.Series(dtype=float)
        return df

    # Prepare feature matrix
    score_data = df[ML_FEATURE_COLUMNS].copy()
    score_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    score_data.fillna(0.0, inplace=True)

    # Scale using the training scaler
    scaled = scaler.transform(score_data)

    # Raw anomaly scores from Isolation Forest
    # decision_function returns negative values for anomalies
    raw_scores = model.decision_function(scaled)

    # Normalise to 0–100 (more anomalous = higher score)
    raw_min = float(np.min(raw_scores))
    raw_max = float(np.max(raw_scores))

    if raw_max - raw_min > 0:
        normalised = 100.0 * (1.0 - (raw_scores - raw_min) / (raw_max - raw_min))
    else:
        # All scores are identical — assign neutral score
        normalised = np.full_like(raw_scores, 50.0)

    df['ml_score'] = np.round(normalised, 2)

    logger.info(
        "ML scoring complete: %d flows. "
        "Score range: [%.1f, %.1f], Median: %.1f, Flows > 70: %d",
        len(df),
        df['ml_score'].min(),
        df['ml_score'].max(),
        df['ml_score'].median(),
        (df['ml_score'] > 70).sum(),
    )

    return df


# =========================================================================
# Explainability — "Why was this flagged?"
# =========================================================================

def explain_anomaly(
    row: pd.Series,
    training_stats: Dict[str, Dict[str, float]],
    sigma_threshold: float = 2.0,
) -> str:
    """Generate a plain-text explanation of why a flow was flagged.

    For each ML feature, checks whether the flow's value is more than
    *sigma_threshold* standard deviations away from the benign mean.

    Parameters
    ----------
    row : pd.Series
        A single scored flow (must contain ML feature columns).
    training_stats : Dict
        Benign mean/std per feature from training.
    sigma_threshold : float, optional
        Number of standard deviations to consider anomalous. Default 2.0.

    Returns
    -------
    str
        Human-readable explanation. Empty string if no feature is anomalous.
    """
    explanations: List[str] = []

    for col in ML_FEATURE_COLUMNS:
        if col not in training_stats:
            continue

        value = float(row.get(col, 0.0))
        mean = training_stats[col]['mean']
        std = training_stats[col]['std']

        # Compute the threshold boundary
        if std > 0:
            threshold_low = mean - sigma_threshold * std
            threshold_high = mean + sigma_threshold * std
            deviation = abs(value - mean) / std
        else:
            # If std is 0, any different value is anomalous
            threshold_low = mean
            threshold_high = mean
            deviation = abs(value - mean) if value != mean else 0.0

        if deviation > sigma_threshold or (std == 0 and value != mean):
            desc = _FEATURE_DESCRIPTIONS.get(col, col)
            direction = 'unusually low' if value < mean else 'unusually high'
            explanations.append(
                f"{col} is {value:.4f} "
                f"(benign mean: {mean:.4f}, "
                f"{sigma_threshold:.0f}σ range: [{threshold_low:.4f}, {threshold_high:.4f}]) "
                f"— {direction} {desc}"
            )

    return '; '.join(explanations) if explanations else ''


def explain_all_anomalies(
    scored_df: pd.DataFrame,
    training_stats: Dict[str, Dict[str, float]],
    threshold: float = 70.0,
) -> pd.DataFrame:
    """Add ML explanations for all flows above the score threshold.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Must contain ``ml_score`` column.
    training_stats : Dict
        Benign training statistics.
    threshold : float, optional
        Only explain flows with ml_score > threshold. Default 70.

    Returns
    -------
    pd.DataFrame
        Copy with ``ml_explanation`` column added.
    """
    df = scored_df.copy()

    explanations: List[str] = []
    for _, row in df.iterrows():
        if float(row.get('ml_score', 0.0)) > threshold:
            explanations.append(explain_anomaly(row, training_stats))
        else:
            explanations.append('')

    df['ml_explanation'] = explanations

    num_explained = sum(1 for e in explanations if e)
    logger.info(
        "ML explanations generated: %d flows explained (threshold=%.0f).",
        num_explained, threshold,
    )

    return df


# =========================================================================
# Main — standalone testing
# =========================================================================

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    )

    print("\n" + "=" * 72)
    print("ML ANOMALY LAYER — DEMO")
    print("=" * 72)

    # Create synthetic benign training data
    np.random.seed(42)
    n_benign = 100
    benign_data = pd.DataFrame({
        'interval_cov':              np.random.uniform(0.5, 2.0, n_benign),
        'mean_interval':             np.random.uniform(10, 600, n_benign),
        'std_interval':              np.random.uniform(5, 300, n_benign),
        'bytes_cov':                 np.random.uniform(0.3, 2.0, n_benign),
        'mean_bytes':                np.random.uniform(100, 5000, n_benign),
        'num_connections':           np.random.randint(10, 200, n_benign),
        'session_duration':          np.random.uniform(600, 36000, n_benign),
        'pct_connections_off_hours': np.random.uniform(0.0, 0.3, n_benign),
        'mean_connection_duration':  np.random.uniform(0.1, 10.0, n_benign),
        'failed_connection_ratio':   np.random.uniform(0.0, 0.2, n_benign),
        'payload_entropy':           np.random.uniform(1.0, 4.0, n_benign),
    })

    # Train
    model, scaler, stats = train_isolation_forest(benign_data)

    # Create test data (mix of benign and suspicious)
    test_data = pd.DataFrame([
        {  # Suspicious: very regular beaconing
            'interval_cov': 0.03, 'mean_interval': 60, 'std_interval': 1.8,
            'bytes_cov': 0.05, 'mean_bytes': 256, 'num_connections': 100,
            'session_duration': 7200, 'pct_connections_off_hours': 0.7,
            'mean_connection_duration': 0.5, 'failed_connection_ratio': 0.0,
            'payload_entropy': 0.5,
            'src_ip': '192.168.1.10', 'dst_ip': '45.33.32.156', 'dst_port': 443,
        },
        {  # Normal browsing
            'interval_cov': 1.5, 'mean_interval': 120, 'std_interval': 180,
            'bytes_cov': 1.2, 'mean_bytes': 2500, 'num_connections': 30,
            'session_duration': 1800, 'pct_connections_off_hours': 0.1,
            'mean_connection_duration': 3.0, 'failed_connection_ratio': 0.05,
            'payload_entropy': 3.2,
            'src_ip': '192.168.1.20', 'dst_ip': '172.217.14.99', 'dst_port': 443,
        },
        {  # Moderately suspicious
            'interval_cov': 0.2, 'mean_interval': 90, 'std_interval': 18,
            'bytes_cov': 0.3, 'mean_bytes': 512, 'num_connections': 60,
            'session_duration': 5400, 'pct_connections_off_hours': 0.4,
            'mean_connection_duration': 1.0, 'failed_connection_ratio': 0.1,
            'payload_entropy': 1.5,
            'src_ip': '192.168.1.30', 'dst_ip': '10.0.0.50', 'dst_port': 8080,
        },
    ])

    # Score
    scored = score_flows(model, scaler, test_data, stats)

    # Explain
    explained = explain_all_anomalies(scored, stats, threshold=40)

    print("\nScoring Results:")
    print(
        explained[['src_ip', 'dst_ip', 'dst_port', 'ml_score']].to_string(
            index=False,
        )
    )
    print("\nExplanations for flagged flows:")
    for _, row in explained.iterrows():
        if row['ml_explanation']:
            print(f"\n  {row['src_ip']} → {row['dst_ip']}:{row['dst_port']} "
                  f"(score: {row['ml_score']:.1f}):")
            for part in row['ml_explanation'].split('; '):
                print(f"    • {part}")

    print("\n" + "=" * 72)
