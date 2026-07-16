"""
beacon_feature_engineering.py — Feature Engineering for C2 Beaconing Detection

Computes ALL statistical features per flow group.  This is the analytical
heart of the detection engine: raw connection records are transformed into
a rich feature vector that captures timing regularity, volume patterns,
and connection metadata — all signals that distinguish C2 beacons from
legitimate traffic.

Feature categories:
    1. Timing features    (8)  — interval statistics, autocorrelation,
                                  off-hours activity
    2. Volume features    (6)  — byte distribution, entropy, burst detection
    3. Connection features(5)  — count, duration, failure rate
    4. Cross-flow features(2)  — host rarity, DNS avoidance

Author: C2 Beaconing Detection Engine
License: MIT
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

# ---------------------------------------------------------------------------
# Sibling module imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_grouping import group_flows  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Type alias for flow group keys
FlowKey = Tuple[str, str, int]

# ---------------------------------------------------------------------------
# Feature columns used by downstream ML layer
# ---------------------------------------------------------------------------
ML_FEATURE_COLUMNS: List[str] = [
    'interval_cov', 'mean_interval', 'std_interval',
    'bytes_cov', 'mean_bytes',
    'num_connections', 'session_duration',
    'pct_connections_off_hours',
    'mean_connection_duration', 'failed_connection_ratio',
    'payload_entropy',
]


# =========================================================================
# Helper Functions
# =========================================================================

def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide with protection against zero / NaN denominators."""
    if denominator == 0 or np.isnan(denominator):
        return default
    return numerator / denominator


def _compute_autocorrelation(values: np.ndarray, max_lag: int = 5) -> float:
    """Compute maximum normalised autocorrelation for lags 1 through *max_lag*.

    Uses numpy's correlate in 'full' mode, then normalises by the
    zero-lag (variance) term.  Returns the maximum autocorrelation
    coefficient found for lags 1–max_lag.

    Returns 0.0 if the series is too short or constant.
    """
    if len(values) < max_lag + 2:
        return 0.0

    # Centre the series
    centred = values - np.mean(values)
    var = np.sum(centred ** 2)
    if var == 0:
        return 0.0

    # Full autocorrelation via numpy
    full_corr = np.correlate(centred, centred, mode='full')
    # Normalise
    full_corr = full_corr / var
    # The zero-lag peak is at the centre of the output
    mid = len(full_corr) // 2

    # Extract lags 1 through max_lag
    lag_values = []
    for lag in range(1, max_lag + 1):
        idx = mid + lag
        if idx < len(full_corr):
            lag_values.append(abs(full_corr[idx]))

    return float(max(lag_values)) if lag_values else 0.0


def _compute_payload_entropy(byte_values: np.ndarray, num_bins: int = 20) -> float:
    """Compute Shannon entropy of the packet-size distribution.

    Bin byte values into *num_bins* equal-width buckets, compute
    normalised frequency, then return Shannon entropy (base 2).
    """
    if len(byte_values) < 2:
        return 0.0

    # Create histogram (counts per bin)
    counts, _ = np.histogram(byte_values, bins=num_bins)
    # Normalise to probability distribution
    probs = counts / counts.sum()
    # Remove zero-probability bins (log(0) is undefined)
    probs = probs[probs > 0]

    return float(scipy_entropy(probs, base=2))


# =========================================================================
# Per-Flow Feature Computation
# =========================================================================

def compute_flow_features(flow_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute the full feature vector for a single flow group.

    Parameters
    ----------
    flow_df : pd.DataFrame
        Connection records for one (src_ip, dst_ip, dst_port) group,
        sorted by timestamp.  Expected columns: timestamp, packet_size,
        bytes_sent, bytes_received, connection_duration, is_failed,
        has_dns_lookup, dst_port.

    Returns
    -------
    Dict[str, Any]
        Feature name → value mapping.
    """
    features: Dict[str, Any] = {}

    # Ensure timestamps are datetime
    timestamps = pd.to_datetime(flow_df['timestamp'])
    n = len(flow_df)

    # === Flow identification ==============================================
    features['src_ip'] = flow_df['src_ip'].iloc[0]
    features['dst_ip'] = flow_df['dst_ip'].iloc[0]
    features['dst_port'] = int(flow_df['dst_port'].iloc[0])
    features['dst_domain'] = flow_df['dst_domain'].iloc[0] if 'dst_domain' in flow_df.columns else ""

    # =====================================================================
    # TIMING FEATURES
    # =====================================================================

    # 1. Inter-arrival times (seconds)
    if n >= 2:
        deltas = timestamps.diff().dropna().dt.total_seconds().values
    else:
        deltas = np.array([0.0])

    features['inter_arrival_times'] = deltas.tolist()

    # 2. Mean interval
    features['mean_interval'] = float(np.mean(deltas)) if len(deltas) > 0 else 0.0

    # 3. Std interval
    features['std_interval'] = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0

    # 4. Coefficient of variation (PRIMARY beaconing signal)
    features['interval_cov'] = _safe_div(
        features['std_interval'], features['mean_interval'], default=float('inf'),
    )

    # 5. Median absolute deviation of intervals (robust to outliers)
    if len(deltas) > 1:
        median_val = np.median(deltas)
        features['interval_mad'] = float(np.median(np.abs(deltas - median_val)))
    else:
        features['interval_mad'] = 0.0

    # 6. Autocorrelation score (periodicity detection)
    features['autocorrelation_score'] = _compute_autocorrelation(deltas, max_lag=5)

    # 7. Session duration (seconds between first and last connection)
    features['session_duration'] = (
        (timestamps.max() - timestamps.min()).total_seconds()
    )

    # 8. Percentage of connections during off-hours (00:00–05:59)
    hours = timestamps.dt.hour
    off_hours_mask = hours.between(0, 5)
    features['pct_connections_off_hours'] = float(off_hours_mask.mean())

    # =====================================================================
    # VOLUME FEATURES
    # =====================================================================

    # Total bytes per connection
    total_bytes = (
        flow_df['bytes_sent'].fillna(0) + flow_df['bytes_received'].fillna(0)
    ).values.astype(float)

    # 9. Mean bytes
    features['mean_bytes'] = float(np.mean(total_bytes))

    # 10. Std bytes
    features['std_bytes'] = float(np.std(total_bytes, ddof=1)) if n > 1 else 0.0

    # 11. Bytes coefficient of variation
    features['bytes_cov'] = _safe_div(
        features['std_bytes'], features['mean_bytes'], default=float('inf'),
    )

    # 12. Bytes sent/received ratio
    mean_sent = float(flow_df['bytes_sent'].fillna(0).mean())
    mean_recv = float(flow_df['bytes_received'].fillna(0).mean())
    features['bytes_sent_received_ratio'] = _safe_div(mean_sent, mean_recv, default=0.0)

    # 13. Payload entropy (Shannon entropy of packet-size distribution)
    packet_sizes = flow_df['packet_size'].fillna(0).values.astype(float)
    features['payload_entropy'] = _compute_payload_entropy(packet_sizes)

    # 14. Anomalous burst detection
    if features['mean_bytes'] > 0:
        features['has_anomalous_burst'] = int(
            np.any(total_bytes > 5.0 * features['mean_bytes'])
        )
    else:
        features['has_anomalous_burst'] = 0

    # =====================================================================
    # CONNECTION FEATURES
    # =====================================================================

    # 15. Number of connections
    features['num_connections'] = n

    # 16. Number of unique destination ports
    features['num_unique_ports'] = int(flow_df['dst_port'].nunique())

    # 17. Mean connection duration
    durations = flow_df['connection_duration'].fillna(0).values.astype(float)
    features['mean_connection_duration'] = float(np.mean(durations))

    # 18. Std connection duration
    features['std_connection_duration'] = (
        float(np.std(durations, ddof=1)) if n > 1 else 0.0
    )

    # 19. Failed connection ratio
    failed = flow_df['is_failed'].fillna(0).values.astype(int)
    features['failed_connection_ratio'] = float(np.mean(failed))

    # =====================================================================
    # DNS Feature (per-flow, not cross-flow)
    # =====================================================================
    dns_lookup = flow_df['has_dns_lookup'].fillna(0).values.astype(int)
    features['is_direct_ip_no_dns'] = int(np.mean(dns_lookup) < 0.5)

    # Drop the raw list from the output (it's large and not needed downstream)
    # Keep it available for debugging but don't include in the main DataFrame
    features.pop('inter_arrival_times', None)

    return features


# =========================================================================
# Batch Feature Computation
# =========================================================================

def compute_all_features(
    flow_groups: Dict[FlowKey, pd.DataFrame],
) -> pd.DataFrame:
    """Compute features for all flow groups.

    Parameters
    ----------
    flow_groups : Dict[FlowKey, pd.DataFrame]
        Output of :func:`flow_grouping.group_flows`.

    Returns
    -------
    pd.DataFrame
        One row per flow group with all feature columns.
    """
    if not flow_groups:
        logger.warning("No flow groups to process — returning empty DataFrame.")
        return pd.DataFrame()

    feature_rows: List[Dict[str, Any]] = []

    for idx, (key, flow_df) in enumerate(flow_groups.items()):
        try:
            features = compute_flow_features(flow_df)
            feature_rows.append(features)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to compute features for flow %s: %s", key, exc,
            )

        if (idx + 1) % 100 == 0:
            logger.info("Processed %d / %d flow groups...", idx + 1, len(flow_groups))

    features_df = pd.DataFrame(feature_rows)

    # Replace inf values with NaN, then fill with sensible defaults
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = features_df[numeric_cols].fillna(0.0)

    logger.info(
        "Feature engineering complete: %d flows, %d features each.",
        len(features_df),
        len(features_df.columns),
    )

    return features_df


def compute_cross_flow_features(
    features_df: pd.DataFrame,
    flow_groups: Dict[FlowKey, pd.DataFrame],
) -> pd.DataFrame:
    """Add cross-flow features that require global context.

    Parameters
    ----------
    features_df : pd.DataFrame
        Per-flow features from :func:`compute_all_features`.
    flow_groups : Dict[FlowKey, pd.DataFrame]
        Original flow groups for raw data access.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with cross-flow features added:
        - ``num_internal_hosts_contacting``: unique src_ips per (dst_ip, dst_port)
    """
    if features_df.empty:
        return features_df

    df = features_df.copy()

    # ------------------------------------------------------------------
    # num_internal_hosts_contacting
    # Count unique src_ips that contact each (dst_ip, dst_port) across
    # ALL flow groups.
    # ------------------------------------------------------------------
    dest_to_sources: Dict[Tuple[str, int], set] = {}
    for (src_ip, dst_ip, dst_port), _ in flow_groups.items():
        dest_key = (dst_ip, dst_port)
        if dest_key not in dest_to_sources:
            dest_to_sources[dest_key] = set()
        dest_to_sources[dest_key].add(src_ip)

    df['num_internal_hosts_contacting'] = df.apply(
        lambda row: len(
            dest_to_sources.get(
                (row['dst_ip'], int(row['dst_port'])), set()
            )
        ),
        axis=1,
    )

    logger.info(
        "Cross-flow features added. Unique destinations: %d",
        len(dest_to_sources),
    )

    return df


# =========================================================================
# Convenience Function — Full Pipeline
# =========================================================================

def engineer_features(
    flows_df: pd.DataFrame,
    min_connections: int = 10,
) -> pd.DataFrame:
    """End-to-end feature engineering: group → compute → cross-flow.

    Parameters
    ----------
    flows_df : pd.DataFrame
        Raw connection records.
    min_connections : int, optional
        Minimum connections per flow group. Default 10.

    Returns
    -------
    pd.DataFrame
        Fully engineered feature DataFrame ready for scoring.
    """
    logger.info(
        "Starting feature engineering on %d raw records "
        "(min_connections=%d)...",
        len(flows_df), min_connections,
    )

    # Step 1: Group flows
    flow_groups = group_flows(flows_df, min_connections=min_connections)

    if not flow_groups:
        logger.warning("No flow groups survived filtering — returning empty DataFrame.")
        return pd.DataFrame()

    # Step 2: Compute per-flow features
    features_df = compute_all_features(flow_groups)

    # Step 3: Compute cross-flow features
    features_df = compute_cross_flow_features(features_df, flow_groups)

    logger.info(
        "Feature engineering complete: %d flows with %d features.",
        len(features_df), len(features_df.columns),
    )

    return features_df


# =========================================================================
# Main — standalone execution
# =========================================================================

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    )

    _this_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        _this_dir, '..', 'data', 'processed', 'synthetic_flows.csv',
    )

    if not os.path.isfile(data_path):
        logger.error(
            "Synthetic flows CSV not found at %s — "
            "run the data generator first.",
            data_path,
        )
        sys.exit(1)

    logger.info("Loading synthetic flows from %s", data_path)
    raw_df = pd.read_csv(data_path)
    logger.info("Loaded %d raw connection records.", len(raw_df))

    # Run full feature engineering pipeline
    features = engineer_features(raw_df, min_connections=10)

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 80)
    print(f"Total flow groups processed: {len(features)}")
    print(f"\nFeature columns ({len(features.columns)}):")
    for col in sorted(features.columns):
        print(f"  - {col}")

    print(f"\nSample features (first 5 flows):")
    display_cols = [
        'src_ip', 'dst_ip', 'dst_port', 'num_connections',
        'interval_cov', 'mean_interval', 'autocorrelation_score',
        'bytes_cov', 'payload_entropy', 'pct_connections_off_hours',
        'num_internal_hosts_contacting', 'is_direct_ip_no_dns',
    ]
    available = [c for c in display_cols if c in features.columns]
    print(features[available].head().to_string(index=False))
    print("=" * 80)
