"""
flow_grouping.py — Flow Grouping Module for C2 Beaconing Detection Engine

Groups raw connection records into logical flows by the tuple
(src_ip, dst_ip, dst_port). Each flow group contains all connection records
between a specific source and destination, sorted chronologically.

This is the first stage of the detection pipeline: raw connection logs
are partitioned into per-destination flows that can be analyzed for
periodic beaconing patterns.

Author: C2 Beaconing Detection Engine
License: MIT
"""

import os
import sys
import logging
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Type alias for flow group keys
FlowKey = Tuple[str, str, int]


# =========================================================================
# Core Functions
# =========================================================================

def group_flows(
    df: pd.DataFrame,
    min_connections: int = 10,
) -> Dict[FlowKey, pd.DataFrame]:
    """Group connection records into flows by (src_ip, dst_ip, dst_port).

    Parameters
    ----------
    df : pd.DataFrame
        Raw connection records with at least the columns:
        flow_id, src_ip, dst_ip, dst_port, timestamp, packet_size,
        protocol, connection_duration, bytes_sent, bytes_received,
        is_failed, has_dns_lookup.
    min_connections : int, optional
        Minimum number of connection records required for a flow group
        to be retained. Groups with fewer records are discarded because
        statistical features are unreliable with too few data points.
        Default is 10.

    Returns
    -------
    Dict[FlowKey, pd.DataFrame]
        Dictionary keyed by (src_ip, dst_ip, dst_port) tuples. Each value
        is a DataFrame of that flow's connection records sorted by
        timestamp in ascending order.

    Raises
    ------
    ValueError
        If required columns are missing from the input DataFrame.
    """
    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------
    required_columns = {
        'src_ip', 'dst_ip', 'dst_port', 'timestamp',
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing}"
        )

    if df.empty:
        logger.warning("Input DataFrame is empty — returning no flow groups.")
        return {}

    # ------------------------------------------------------------------
    # Ensure timestamp is datetime-like for correct sorting
    # ------------------------------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        logger.info("Converting 'timestamp' column to datetime.")
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # ------------------------------------------------------------------
    # Group by the canonical flow key
    # ------------------------------------------------------------------
    grouped = df.groupby(['src_ip', 'dst_ip', 'dst_port'])

    flow_groups: Dict[FlowKey, pd.DataFrame] = {}
    total_groups = 0
    filtered_groups = 0

    for key, group_df in grouped:
        total_groups += 1
        if len(group_df) < min_connections:
            filtered_groups += 1
            continue

        # Sort by timestamp within each group
        sorted_df = group_df.sort_values('timestamp').reset_index(drop=True)
        flow_groups[key] = sorted_df

    logger.info(
        "Grouped %d total flows → %d retained (>= %d connections), "
        "%d filtered out.",
        total_groups,
        len(flow_groups),
        min_connections,
        filtered_groups,
    )

    return flow_groups


def get_flow_summary(flow_groups: Dict[FlowKey, pd.DataFrame]) -> pd.DataFrame:
    """Produce a one-row-per-flow summary DataFrame.

    Parameters
    ----------
    flow_groups : Dict[FlowKey, pd.DataFrame]
        Output of :func:`group_flows`.

    Returns
    -------
    pd.DataFrame
        Columns: src_ip, dst_ip, dst_port, num_connections, first_seen,
        last_seen.
    """
    if not flow_groups:
        logger.warning("No flow groups provided — returning empty summary.")
        return pd.DataFrame(
            columns=[
                'src_ip', 'dst_ip', 'dst_port',
                'num_connections', 'first_seen', 'last_seen',
            ]
        )

    rows = []
    for (src_ip, dst_ip, dst_port), flow_df in flow_groups.items():
        rows.append({
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'dst_port': dst_port,
            'num_connections': len(flow_df),
            'first_seen': flow_df['timestamp'].min(),
            'last_seen': flow_df['timestamp'].max(),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.sort_values(
        'num_connections', ascending=False, inplace=True,
    )
    summary_df.reset_index(drop=True, inplace=True)

    logger.info(
        "Flow summary: %d groups, total connections: %d",
        len(summary_df),
        summary_df['num_connections'].sum(),
    )

    return summary_df


# =========================================================================
# Main — standalone execution for testing / quick inspection
# =========================================================================

if __name__ == '__main__':
    # Configure console logging for standalone runs
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    )

    # Resolve the path to synthetic_flows.csv relative to this file
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

    # Group and summarise
    groups = group_flows(raw_df, min_connections=10)
    summary = get_flow_summary(groups)

    print("\n" + "=" * 72)
    print("FLOW GROUPING SUMMARY")
    print("=" * 72)
    print(f"Total flow groups (>= 10 connections): {len(groups)}")
    print(f"\nTop 20 flows by connection count:")
    print(summary.head(20).to_string(index=False))
    print("=" * 72)
