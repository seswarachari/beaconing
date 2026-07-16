"""
beacon_scoring_engine.py — Deterministic Beacon Scoring Engine

Produces a composite beacon score (0–100) for every flow group by
combining weighted sub-scores derived from timing regularity, volume
characteristics, connection metadata, and contextual signals.

The scoring weights are defined in a top-level configuration dict so
that SOC analysts can tune them without modifying logic.

Author: C2 Beaconing Detection Engine
License: MIT
"""

import logging
from typing import Dict, Optional

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# =========================================================================
# Scoring Weights — CONFIGURABLE
# =========================================================================
# Each weight represents the fraction of the final composite score
# contributed by the corresponding sub-score.  All weights MUST sum to 1.0.
#
# Rationale:
#   interval_cov (0.30)   — Primary beaconing signal.  Low CoV = regular
#                           timing = highly suspicious.
#   autocorrelation (0.15) — Strong repeating lag pattern confirms
#                           periodicity independent of CoV.
#   bytes_cov (0.10)      — Beacons typically send fixed-size payloads;
#                           low byte-size variance reinforces suspicion.
#   num_connections (0.10) — More data points increase confidence in the
#                           statistical signals.
#   session_duration (0.10) — Long-running beacon sessions are more
#                           operationally significant.
#   off_hours_pct (0.10)  — Activity concentrated outside business hours
#                           is a common C2 indicator.
#   not_allowlisted (0.05) — Penalty for destinations NOT in allowlist.
#   direct_ip_no_dns (0.05) — No DNS resolution is a stealth indicator.
#   low_internal_hosts (0.05) — Rarity: few hosts contacting destination.
# =========================================================================

SCORING_WEIGHTS: Dict[str, float] = {
    'interval_cov':       0.30,
    'bytes_cov':          0.10,
    'autocorrelation':    0.15,
    'num_connections':    0.10,
    'session_duration':   0.10,
    'off_hours_pct':      0.10,
    'not_allowlisted':    0.05,
    'direct_ip_no_dns':   0.05,
    'low_internal_hosts': 0.05,
}

assert abs(sum(SCORING_WEIGHTS.values()) - 1.0) < 1e-9, (
    "Scoring weights must sum to 1.0"
)


# =========================================================================
# MITRE ATT&CK Technique Mapping
# =========================================================================
_MITRE_MAP: Dict[str, str] = {
    'fixed':   'T1071.001 — Application Layer Protocol: Web Protocols '
               '(fixed-interval beacon)',
    'jittered': 'T1071.001 — Application Layer Protocol: Web Protocols '
                '(jittered beacon with sleep randomisation)',
    'evasive': 'T1573 — Encrypted Channel / T1090 — Proxy '
               '(low-and-slow evasive beacon)',
    'unlikely': 'No strong MITRE mapping — likely benign',
}

_EXFIL_TECHNIQUE = 'T1041 — Exfiltration Over C2 Channel'


# =========================================================================
# Sub-Score Computation
# =========================================================================

def compute_sub_scores(row: pd.Series) -> Dict[str, float]:
    """Compute individual sub-scores (each 0–100) from feature values.

    Parameters
    ----------
    row : pd.Series
        A single row from the engineered-features DataFrame.  Expected
        keys include ``interval_cov``, ``bytes_cov``,
        ``autocorrelation_score``, ``num_connections``,
        ``session_duration``, ``pct_connections_off_hours``,
        ``destination_in_allowlist``, ``is_direct_ip_no_dns``,
        ``num_internal_hosts_contacting``.

    Returns
    -------
    Dict[str, float]
        Mapping of sub-score name → value in [0, 100].
    """
    scores: Dict[str, float] = {}

    # --- Interval CoV (inverted: lower CoV → higher score) ---------------
    cov = float(row.get('interval_cov', 1.0))
    scores['interval_cov_score'] = min(100.0, max(0.0, 100.0 * (1.0 - cov / 0.5)))

    # --- Bytes CoV (inverted, threshold 1.0) -----------------------------
    bcov = float(row.get('bytes_cov', 2.0))
    scores['bytes_cov_score'] = min(100.0, max(0.0, 100.0 * (1.0 - bcov / 1.0)))

    # --- Autocorrelation (scale 0-1 → 0-100) -----------------------------
    acorr = float(row.get('autocorrelation_score', 0.0))
    scores['autocorrelation_sub_score'] = min(100.0, max(0.0, acorr * 100.0))

    # --- Num connections (more = higher confidence) ----------------------
    n = int(row.get('num_connections', 0))
    scores['num_connections_score'] = min(100.0, n * 2.0)

    # --- Session duration (up to 1 hour = 3600s → 100) -------------------
    dur = float(row.get('session_duration', 0.0))
    scores['session_duration_score'] = min(100.0, dur / 3600.0 * 100.0)

    # --- Off-hours percentage (already 0-1 fraction → ×100) --------------
    off = float(row.get('pct_connections_off_hours', 0.0))
    scores['off_hours_score'] = min(100.0, off * 100.0)

    # --- Not allowlisted (binary) ----------------------------------------
    allowlisted = bool(row.get('destination_in_allowlist', False))
    scores['not_allowlisted_score'] = 0.0 if allowlisted else 100.0

    # --- Direct IP / no DNS (binary) -------------------------------------
    direct = int(row.get('is_direct_ip_no_dns', 0))
    scores['direct_ip_score'] = 100.0 if direct else 0.0

    # --- Low internal hosts (fewer → higher score) -----------------------
    hosts = int(row.get('num_internal_hosts_contacting', 1))
    scores['low_internal_hosts_score'] = max(
        0.0, 100.0 * (1.0 - hosts / 10.0),
    )

    return scores


def compute_beacon_score(row: pd.Series) -> float:
    """Compute the final weighted composite beacon score (0–100).

    Parameters
    ----------
    row : pd.Series
        Must contain sub-score columns produced by :func:`compute_sub_scores`
        OR the raw feature columns (in which case sub-scores are computed
        on the fly).

    Returns
    -------
    float
        Composite deterministic score in [0, 100].
    """
    # Compute sub-scores if not already present
    if 'interval_cov_score' not in row.index:
        sub = compute_sub_scores(row)
    else:
        sub = {
            'interval_cov_score':        float(row['interval_cov_score']),
            'bytes_cov_score':           float(row['bytes_cov_score']),
            'autocorrelation_sub_score': float(row['autocorrelation_sub_score']),
            'num_connections_score':     float(row['num_connections_score']),
            'session_duration_score':    float(row['session_duration_score']),
            'off_hours_score':           float(row['off_hours_score']),
            'not_allowlisted_score':     float(row['not_allowlisted_score']),
            'direct_ip_score':           float(row['direct_ip_score']),
            'low_internal_hosts_score':  float(row['low_internal_hosts_score']),
        }

    # Map sub-score keys to weight keys
    weight_map = {
        'interval_cov_score':        'interval_cov',
        'bytes_cov_score':           'bytes_cov',
        'autocorrelation_sub_score': 'autocorrelation',
        'num_connections_score':     'num_connections',
        'session_duration_score':    'session_duration',
        'off_hours_score':           'off_hours_pct',
        'not_allowlisted_score':     'not_allowlisted',
        'direct_ip_score':           'direct_ip_no_dns',
        'low_internal_hosts_score':  'low_internal_hosts',
    }

    composite = sum(
        sub[score_key] * SCORING_WEIGHTS[weight_key]
        for score_key, weight_key in weight_map.items()
    )

    return round(min(100.0, max(0.0, composite)), 2)


# =========================================================================
# Beacon Type Classification
# =========================================================================

def guess_beacon_type(row: pd.Series) -> str:
    """Classify the likely beacon type based on CoV and interval magnitude.

    Parameters
    ----------
    row : pd.Series
        Must contain ``interval_cov`` and ``mean_interval``.

    Returns
    -------
    str
        One of 'fixed', 'jittered', 'evasive', 'unlikely'.
    """
    cov = float(row.get('interval_cov', 1.0))
    mean_int = float(row.get('mean_interval', 0.0))

    if mean_int > 300:
        return 'evasive'
    if cov < 0.1 and mean_int < 120:
        return 'fixed'
    if 0.1 <= cov <= 0.4 and mean_int < 120:
        return 'jittered'
    return 'unlikely'


def suggest_mitre_technique(
    beacon_type_guess: str,
    has_anomalous_burst: int = 0,
) -> str:
    """Map beacon type to MITRE ATT&CK technique IDs.

    Parameters
    ----------
    beacon_type_guess : str
        Output of :func:`guess_beacon_type`.
    has_anomalous_burst : int
        1 if the flow has an anomalous data burst (possible exfiltration).

    Returns
    -------
    str
        MITRE technique description string.
    """
    technique = _MITRE_MAP.get(beacon_type_guess, _MITRE_MAP['unlikely'])
    if has_anomalous_burst:
        technique += f' + {_EXFIL_TECHNIQUE}'
    return technique


# =========================================================================
# Batch Scoring
# =========================================================================

def score_all_flows(features_df: pd.DataFrame) -> pd.DataFrame:
    """Score every flow in the features DataFrame.

    Adds sub-score columns, ``deterministic_score``, ``beacon_type_guess``,
    and ``suggested_mitre_technique``.

    Parameters
    ----------
    features_df : pd.DataFrame
        Output of the feature engineering + allowlist stages.

    Returns
    -------
    pd.DataFrame
        Enriched copy with scoring columns.
    """
    df = features_df.copy()

    if df.empty:
        logger.warning("Empty features DataFrame — nothing to score.")
        return df

    # Compute sub-scores for every row
    sub_scores_list = df.apply(compute_sub_scores, axis=1)
    sub_scores_df = pd.DataFrame(sub_scores_list.tolist(), index=df.index)
    df = pd.concat([df, sub_scores_df], axis=1)

    # Compute composite deterministic score
    df['deterministic_score'] = df.apply(compute_beacon_score, axis=1)

    # Classify beacon type and suggest MITRE technique
    df['beacon_type_guess'] = df.apply(guess_beacon_type, axis=1)
    df['suggested_mitre_technique'] = df.apply(
        lambda r: suggest_mitre_technique(
            r['beacon_type_guess'],
            int(r.get('has_anomalous_burst', 0)),
        ),
        axis=1,
    )

    # Sort by score descending for analyst convenience
    df.sort_values('deterministic_score', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        "Scoring complete: %d flows scored. "
        "Max score: %.1f, Median: %.1f, Flows > 75: %d",
        len(df),
        df['deterministic_score'].max(),
        df['deterministic_score'].median(),
        (df['deterministic_score'] > 75).sum(),
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

    # Create a tiny synthetic feature set for demonstration
    demo_data = pd.DataFrame([
        {
            'src_ip': '192.168.1.10', 'dst_ip': '45.33.32.156', 'dst_port': 443,
            'interval_cov': 0.05, 'bytes_cov': 0.1, 'autocorrelation_score': 0.9,
            'num_connections': 80, 'session_duration': 7200,
            'pct_connections_off_hours': 0.6, 'mean_interval': 60,
            'destination_in_allowlist': False, 'is_direct_ip_no_dns': 1,
            'num_internal_hosts_contacting': 1, 'has_anomalous_burst': 0,
        },
        {
            'src_ip': '192.168.1.20', 'dst_ip': '172.217.14.99', 'dst_port': 443,
            'interval_cov': 1.2, 'bytes_cov': 1.5, 'autocorrelation_score': 0.1,
            'num_connections': 25, 'session_duration': 3600,
            'pct_connections_off_hours': 0.1, 'mean_interval': 200,
            'destination_in_allowlist': True, 'is_direct_ip_no_dns': 0,
            'num_internal_hosts_contacting': 8, 'has_anomalous_burst': 0,
        },
        {
            'src_ip': '192.168.1.30', 'dst_ip': '10.0.0.50', 'dst_port': 8080,
            'interval_cov': 0.25, 'bytes_cov': 0.3, 'autocorrelation_score': 0.6,
            'num_connections': 45, 'session_duration': 5400,
            'pct_connections_off_hours': 0.4, 'mean_interval': 90,
            'destination_in_allowlist': False, 'is_direct_ip_no_dns': 1,
            'num_internal_hosts_contacting': 2, 'has_anomalous_burst': 1,
        },
    ])

    scored = score_all_flows(demo_data)

    print("\n" + "=" * 72)
    print("BEACON SCORING ENGINE — DEMO")
    print("=" * 72)
    display_cols = [
        'src_ip', 'dst_ip', 'dst_port',
        'deterministic_score', 'beacon_type_guess',
        'suggested_mitre_technique',
    ]
    print(scored[display_cols].to_string(index=False))
    print()

    # Show sub-score breakdown for the top scorer
    top = scored.iloc[0]
    print(f"Sub-score breakdown for top scorer ({top['src_ip']} → "
          f"{top['dst_ip']}:{top['dst_port']}):")
    for col in sorted(scored.columns):
        if col.endswith('_score') and col != 'deterministic_score':
            print(f"  {col:35s} = {top[col]:6.1f}")
    print(f"  {'deterministic_score':35s} = {top['deterministic_score']:6.1f}")
    print("=" * 72)
