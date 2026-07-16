"""
detection_pipeline.py — Combined C2 Beaconing Detection Pipeline

Orchestrates all detection layers into a single end-to-end pipeline:

    1. Load data (synthetic CSV or parsed PCAP)
    2. Group flows by (src_ip, dst_ip, dst_port)
    3. Engineer statistical features per flow group
    4. Apply allowlist filtering for known-good destinations
    5. Run deterministic beacon scoring (weighted composite)
    6. Train Isolation Forest on benign baseline, score all flows
    7. Combine scores and produce final verdicts

Final verdict categories:
    - SUPPRESSED : destination is in the allowlist
    - HIGH       : deterministic score > 75
    - MEDIUM     : deterministic score 40–75 OR ML score > 70
    - CLEAR      : below all thresholds

Author: C2 Beaconing Detection Engine
License: MIT
"""

import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Sibling module imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_grouping import group_flows, get_flow_summary          # noqa: E402
from beacon_feature_engineering import engineer_features          # noqa: E402
from allowlist_filter import apply_allowlist                      # noqa: E402
from beacon_scoring_engine import (                               # noqa: E402
    score_all_flows,
    guess_beacon_type,
    suggest_mitre_technique,
)
from ml_anomaly_layer import (                                    # noqa: E402
    train_isolation_forest,
    score_flows,
    explain_all_anomalies,
    ML_FEATURE_COLUMNS,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers (relative to this file)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
_PROCESSED_DIR = os.path.join(_DATA_DIR, 'processed')
_KAGGLE_DIR = os.path.join(_DATA_DIR, 'kaggle_reference')
_REPORTS_DIR = os.path.join(_PROJECT_ROOT, 'reports')

# ---------------------------------------------------------------------------
# Output columns (final report)
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS = [
    'src_ip', 'dst_ip', 'dst_domain', 'dst_port',
    'num_connections', 'interval_cov', 'mean_interval',
    'deterministic_score', 'ml_score', 'final_verdict',
    'beacon_type_guess', 'suggested_mitre_technique',
    'ml_explanation', 'allowlist_reason',
]

# ---------------------------------------------------------------------------
# Verdict thresholds
# ---------------------------------------------------------------------------
DETERMINISTIC_HIGH_THRESHOLD = 75.0
DETERMINISTIC_MEDIUM_THRESHOLD = 40.0
ML_MEDIUM_THRESHOLD = 70.0


# =========================================================================
# Data Loading Helpers
# =========================================================================

def _load_synthetic_data() -> pd.DataFrame:
    """Load synthetic flow data from the processed directory."""
    csv_path = os.path.join(_PROCESSED_DIR, 'synthetic_flows.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Synthetic flows CSV not found at {csv_path}. "
            "Run the data generator first."
        )
    logger.info("Loading synthetic flows from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d raw connection records.", len(df))
    return df


def _load_pcap_data(pcap_dir: str) -> pd.DataFrame:
    """Load parsed PCAP data.

    Expects CSV files in *pcap_dir* with the standard column schema.
    If a ``parsed_connections.csv`` exists, it is used directly;
    otherwise all ``.csv`` files in the directory are concatenated.
    """
    single_file = os.path.join(pcap_dir, 'parsed_connections.csv')
    if os.path.isfile(single_file):
        logger.info("Loading PCAP data from %s", single_file)
        return pd.read_csv(single_file)

    csv_files = sorted(
        f for f in os.listdir(pcap_dir)
        if f.endswith('.csv')
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in PCAP directory: {pcap_dir}"
        )

    frames = []
    for fname in csv_files:
        fpath = os.path.join(pcap_dir, fname)
        logger.info("Loading %s", fpath)
        frames.append(pd.read_csv(fpath))

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d records from %d PCAP CSV files.", len(combined), len(csv_files))
    return combined


def _load_kaggle_benign() -> Optional[pd.DataFrame]:
    """Attempt to load Kaggle benign reference data.

    Returns None if no Kaggle data is available (non-fatal).
    """
    if not os.path.isdir(_KAGGLE_DIR):
        logger.info("No Kaggle reference directory found — skipping.")
        return None

    csv_files = [f for f in os.listdir(_KAGGLE_DIR) if f.endswith('.csv')]
    if not csv_files:
        logger.info("No CSV files in Kaggle reference directory — skipping.")
        return None

    frames = []
    for fname in csv_files:
        fpath = os.path.join(_KAGGLE_DIR, fname)
        logger.info("Loading Kaggle reference: %s", fpath)
        frames.append(pd.read_csv(fpath))

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d Kaggle benign records.", len(combined))
    return combined


# =========================================================================
# Verdict Logic
# =========================================================================

def _assign_verdict(row: pd.Series) -> str:
    """Assign a final verdict based on scoring thresholds.

    Priority order:
        1. SUPPRESSED — destination is allowlisted
        2. HIGH       — deterministic score > 75
        3. MEDIUM     — deterministic score 40–75 OR ML score > 70
        4. CLEAR      — everything else
    """
    if bool(row.get('destination_in_allowlist', False)):
        return 'SUPPRESSED'

    det_score = float(row.get('deterministic_score', 0.0))
    ml_score = float(row.get('ml_score', 0.0))

    if det_score > DETERMINISTIC_HIGH_THRESHOLD:
        return 'HIGH'

    if det_score > DETERMINISTIC_MEDIUM_THRESHOLD or ml_score > ML_MEDIUM_THRESHOLD:
        return 'MEDIUM'

    return 'CLEAR'


# =========================================================================
# Main Pipeline
# =========================================================================

def run_pipeline(
    data_source: str = 'synthetic',
    pcap_dir: Optional[str] = None,
    min_connections: int = 10,
    raw_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Execute the full C2 beaconing detection pipeline.

    Parameters
    ----------
    data_source : str
        One of 'synthetic' or 'pcap'.
    pcap_dir : str, optional
        Path to directory containing parsed PCAP CSV files.
        Required when *data_source* is 'pcap'.
    min_connections : int, optional
        Minimum connections per flow group for analysis. Default 10.
    raw_df : pd.DataFrame, optional
        If provided, skips loading and uses this dataframe directly.

    Returns
    -------
    pd.DataFrame
        Results table with columns defined in ``OUTPUT_COLUMNS``.
    """
    logger.info("=" * 72)
    logger.info("C2 BEACONING DETECTION PIPELINE — START")
    logger.info("Data source: %s", data_source)
    logger.info("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    logger.info("Step 1/7: Loading data...")
    if raw_df is not None:
        logger.info("Using provided raw_df.")
    elif data_source == 'pcap':
        if pcap_dir is None:
            raise ValueError("pcap_dir must be provided when data_source='pcap'")
        raw_df = _load_pcap_data(pcap_dir)
    else:
        raw_df = _load_synthetic_data()

    # ------------------------------------------------------------------
    # Step 2–3: Group flows and engineer features
    # ------------------------------------------------------------------
    logger.info("Step 2-3/7: Grouping flows and engineering features...")
    features_df = engineer_features(raw_df, min_connections=min_connections)

    if features_df.empty:
        logger.warning(
            "No flow groups survived filtering — pipeline returning empty results."
        )
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # ------------------------------------------------------------------
    # Step 4: Apply allowlist
    # ------------------------------------------------------------------
    logger.info("Step 4/7: Applying allowlist filter...")
    features_df = apply_allowlist(features_df)

    # ------------------------------------------------------------------
    # Step 5: Deterministic scoring
    # ------------------------------------------------------------------
    logger.info("Step 5/7: Running deterministic beacon scoring...")
    scored_df = score_all_flows(features_df)

    # ------------------------------------------------------------------
    # Step 6: ML anomaly detection
    # ------------------------------------------------------------------
    logger.info("Step 6/7: Training ML model and scoring flows...")

    # Separate benign flows for training (allowlisted or low det. score)
    benign_mask = (
        scored_df['destination_in_allowlist'].fillna(False)
        | (scored_df['deterministic_score'] < DETERMINISTIC_MEDIUM_THRESHOLD)
    )
    benign_features = scored_df.loc[benign_mask]

    # Optionally load Kaggle supplement
    kaggle_df = _load_kaggle_benign()

    if len(benign_features) >= 5:
        try:
            model, scaler, training_stats = train_isolation_forest(
                benign_features, kaggle_df,
            )
            scored_df = score_flows(model, scaler, scored_df, training_stats)
            scored_df = explain_all_anomalies(scored_df, training_stats, threshold=70)
        except Exception as exc:  # noqa: BLE001
            logger.error("ML scoring failed: %s — skipping ML layer.", exc)
            scored_df['ml_score'] = 0.0
            scored_df['ml_explanation'] = ''
    else:
        logger.warning(
            "Insufficient benign samples (%d) to train ML model — "
            "skipping ML layer.",
            len(benign_features),
        )
        scored_df['ml_score'] = 0.0
        scored_df['ml_explanation'] = ''

    # ------------------------------------------------------------------
    # Step 7: Assign final verdicts
    # ------------------------------------------------------------------
    logger.info("Step 7/7: Assigning final verdicts...")
    scored_df['final_verdict'] = scored_df.apply(_assign_verdict, axis=1)

    # ------------------------------------------------------------------
    # Assemble output table
    # ------------------------------------------------------------------
    # Ensure all output columns exist
    for col in OUTPUT_COLUMNS:
        if col not in scored_df.columns:
            scored_df[col] = ''

    results = scored_df[OUTPUT_COLUMNS].copy()
    results.sort_values('deterministic_score', ascending=False, inplace=True)
    results.reset_index(drop=True, inplace=True)

    logger.info("=" * 72)
    logger.info("PIPELINE COMPLETE — %d flows analysed.", len(results))
    logger.info("=" * 72)

    return results


# =========================================================================
# Output Utilities
# =========================================================================

def save_results(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> str:
    """Save results to a CSV file.

    Parameters
    ----------
    results_df : pd.DataFrame
        Pipeline output.
    output_path : str, optional
        Destination CSV path.  Defaults to ``reports/detection_results.csv``.

    Returns
    -------
    str
        Absolute path of the saved file.
    """
    if output_path is None:
        os.makedirs(_REPORTS_DIR, exist_ok=True)
        output_path = os.path.join(_REPORTS_DIR, 'detection_results.csv')

    results_df.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)
    return os.path.abspath(output_path)


def print_summary(results_df: pd.DataFrame) -> None:
    """Print a console-friendly summary of detection results."""
    if results_df.empty:
        print("\n[!] No results to summarise.\n")
        return

    total = len(results_df)
    verdict_counts = results_df['final_verdict'].value_counts()

    print("\n" + "=" * 72)
    print("C2 BEACONING DETECTION — RESULTS SUMMARY")
    print("=" * 72)
    print(f"Total flows analysed:  {total}")
    print()

    # Verdict breakdown
    print("Verdict Breakdown:")
    for verdict in ['HIGH', 'MEDIUM', 'CLEAR', 'SUPPRESSED']:
        count = verdict_counts.get(verdict, 0)
        pct = 100.0 * count / max(total, 1)
        bar = '█' * int(pct / 2)
        print(f"  {verdict:12s}  {count:4d}  ({pct:5.1f}%)  {bar}")
    print()

    # Top HIGH-verdict flows
    high_flows = results_df[results_df['final_verdict'] == 'HIGH']
    if not high_flows.empty:
        print(f"Top HIGH-verdict flows ({len(high_flows)} total):")
        display_cols = [
            'src_ip', 'dst_ip', 'dst_port',
            'deterministic_score', 'ml_score',
            'beacon_type_guess', 'num_connections',
        ]
        available = [c for c in display_cols if c in high_flows.columns]
        print(high_flows[available].head(10).to_string(index=False))
        print()

    # Top MEDIUM-verdict flows
    medium_flows = results_df[results_df['final_verdict'] == 'MEDIUM']
    if not medium_flows.empty:
        print(f"Top MEDIUM-verdict flows ({len(medium_flows)} total):")
        display_cols = [
            'src_ip', 'dst_ip', 'dst_port',
            'deterministic_score', 'ml_score',
            'beacon_type_guess',
        ]
        available = [c for c in display_cols if c in medium_flows.columns]
        print(medium_flows[available].head(5).to_string(index=False))
        print()

    # ML explanations for flagged flows
    if 'ml_explanation' in results_df.columns:
        explained = results_df[results_df['ml_explanation'].astype(str).str.len() > 0]
        if not explained.empty:
            print(f"ML Explanations ({len(explained)} flows):")
            for _, row in explained.head(5).iterrows():
                print(
                    f"  {row['src_ip']} → {row['dst_ip']}:{row['dst_port']} "
                    f"(ML: {row.get('ml_score', 'N/A')}):"
                )
                for part in str(row['ml_explanation']).split('; '):
                    if part:
                        print(f"    • {part}")
            print()

    print("=" * 72)


# =========================================================================
# Main — CLI entry point
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='C2 Beaconing Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python detection_pipeline.py --source synthetic\n'
            '  python detection_pipeline.py --source pcap --pcap-dir ../data/raw_pcap\n'
        ),
    )
    parser.add_argument(
        '--source',
        choices=['synthetic', 'pcap'],
        default='synthetic',
        help='Data source type (default: synthetic)',
    )
    parser.add_argument(
        '--pcap-dir',
        type=str,
        default=None,
        help='Path to directory containing parsed PCAP CSV files',
    )
    parser.add_argument(
        '--min-connections',
        type=int,
        default=10,
        help='Minimum connections per flow group (default: 10)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: reports/detection_results.csv)',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging',
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    )

    # Run the pipeline
    try:
        results = run_pipeline(
            data_source=args.source,
            pcap_dir=args.pcap_dir,
            min_connections=args.min_connections,
        )

        # Save results
        saved_path = save_results(results, args.output)
        print(f"\n[✓] Results saved to: {saved_path}")

        # Print summary
        print_summary(results)

    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)
