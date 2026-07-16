#!/usr/bin/env python3
"""
pcap_parser.py
==============
Parse real PCAP files into the same flow-record CSV format as the
synthetic data generator, enabling seamless integration of captured
traffic into the C2 Beaconing Detection Engine pipeline.

Supports:
    - .pcap and .pcapng file formats
    - IPv4 and IPv6 packets
    - TCP, UDP, and ICMP protocol extraction
    - Graceful handling of missing scapy dependency

Output format matches synthetic_flows.csv:
    flow_id, src_ip, dst_ip, dst_port, timestamp, packet_size,
    protocol, connection_duration, bytes_sent, bytes_received,
    is_failed, has_dns_lookup

Usage:
    python -m src.pcap_parser
    python src/pcap_parser.py

Author: C2 Beaconing Detection Engine Team
"""

import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PCAP_DIR = PROJECT_ROOT / "data" / "raw_pcap"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "parsed_pcap_flows.csv"

# Supported file extensions
PCAP_EXTENSIONS = {".pcap", ".pcapng"}

# Protocol number → name mapping (IP protocol field values)
PROTOCOL_MAP = {
    1: "ICMP",
    6: "TCP",
    17: "UDP",
    58: "ICMPv6",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scapy import with graceful fallback
# ---------------------------------------------------------------------------

try:
    from scapy.all import (
        rdpcap,
        IP,
        IPv6,
        TCP,
        UDP,
        ICMP,
        DNS,
        DNSRR,
    )
    SCAPY_AVAILABLE = True
    logger.info("Scapy imported successfully")
except ImportError:
    SCAPY_AVAILABLE = False
    logger.warning(
        "scapy is not installed. PCAP parsing will not be available. "
        "Install it with: pip install scapy>=2.5.0"
    )


# ---------------------------------------------------------------------------
# Packet parsing helpers
# ---------------------------------------------------------------------------

def _extract_packet_info(packet, packet_idx: int) -> Optional[dict]:
    """
    Extract flow-relevant fields from a single scapy packet.

    Handles both IPv4 and IPv6 packets. Extracts:
        - Source/destination IP
        - Destination port (TCP/UDP) or 0 (ICMP/other)
        - Timestamp
        - Packet size
        - Protocol name

    Args:
        packet: A scapy packet object.
        packet_idx: Index of the packet (for flow_id generation).

    Returns:
        Dict with extracted fields, or None if packet is not IP-based.
    """
    # --- Determine IP layer ---
    if IP in packet:
        ip_layer = packet[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto_num = ip_layer.proto
    elif IPv6 in packet:
        ip_layer = packet[IPv6]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto_num = ip_layer.nh  # Next Header field
    else:
        # Non-IP packet (e.g., ARP, L2) — skip
        return None

    # --- Extract protocol name ---
    protocol = PROTOCOL_MAP.get(proto_num, f"OTHER({proto_num})")

    # --- Extract destination port ---
    dst_port = 0
    if TCP in packet:
        dst_port = packet[TCP].dport
        protocol = "TCP"
    elif UDP in packet:
        dst_port = packet[UDP].dport
        protocol = "UDP"
    elif ICMP in packet:
        protocol = "ICMP"
        dst_port = 0

    # --- Timestamp ---
    try:
        pkt_time = float(packet.time)
        timestamp = datetime.fromtimestamp(pkt_time, tz=timezone.utc)
        timestamp_str = timestamp.isoformat()
    except (AttributeError, ValueError, OSError):
        timestamp_str = datetime.now(timezone.utc).isoformat()

    # --- Packet size ---
    packet_size = len(packet)

    # --- DNS lookup detection ---
    has_dns_lookup = 1 if DNS in packet else 0

    return {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "dst_port": int(dst_port),
        "timestamp": timestamp_str,
        "packet_size": int(packet_size),
        "protocol": protocol,
        "has_dns_lookup": has_dns_lookup,
        "_raw_time": pkt_time if 'pkt_time' in dir() else 0.0,
    }


def _aggregate_to_flows(packets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-packet records into flow-level records.

    Groups packets by (src_ip, dst_ip, dst_port, protocol) and computes
    flow-level statistics to match the synthetic_flows.csv schema.

    For each flow group, computes:
        - First timestamp as the flow timestamp
        - Sum of packet_size as total flow size
        - Connection duration from first to last packet
        - bytes_sent / bytes_received estimates

    Args:
        packets_df: DataFrame with per-packet records.

    Returns:
        DataFrame in flow-record format matching synthetic_flows.csv.
    """
    if packets_df.empty:
        return pd.DataFrame(columns=[
            "flow_id", "src_ip", "dst_ip", "dst_port", "timestamp",
            "packet_size", "protocol", "connection_duration",
            "bytes_sent", "bytes_received", "is_failed", "has_dns_lookup",
            "dst_domain"
        ])

    # Parse timestamps for duration computation
    packets_df = packets_df.copy()
    packets_df["_ts"] = pd.to_datetime(packets_df["timestamp"], utc=True)

    # Group by flow tuple
    flow_groups = packets_df.groupby(
        ["src_ip", "dst_ip", "dst_port", "protocol"],
        sort=False,
    )

    flow_records = []
    for flow_idx, ((src_ip, dst_ip, dst_port, protocol), group) in enumerate(
        flow_groups
    ):
        group_sorted = group.sort_values("_ts")

        # Flow timestamp = first packet time
        first_ts = group_sorted["_ts"].iloc[0]
        last_ts = group_sorted["_ts"].iloc[-1]

        # Connection duration (seconds)
        duration = (last_ts - first_ts).total_seconds()
        duration = round(max(0.001, duration), 4)

        # Total packet size for the flow
        total_size = int(group_sorted["packet_size"].sum())

        # Estimate bytes_sent / bytes_received
        # Approximate: first half of packets are "sent", second half "received"
        mid = len(group_sorted) // 2
        bytes_sent = int(group_sorted["packet_size"].iloc[:max(1, mid)].sum())
        bytes_received = int(group_sorted["packet_size"].iloc[max(1, mid):].sum())
        if bytes_received == 0:
            bytes_received = bytes_sent  # Symmetric if only 1 packet

        # DNS lookup: 1 if any packet in the flow had DNS
        has_dns = int(group_sorted["has_dns_lookup"].max())

        # TCP RST/FIN flags are not easily available at this level;
        # mark as not failed (can be refined with deeper TCP analysis)
        is_failed = 0

        flow_records.append({
            "flow_id": f"pcap_flow_{flow_idx:06d}",
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "dst_port": int(dst_port),
            "timestamp": first_ts.isoformat(),
            "packet_size": total_size,
            "protocol": protocol,
            "connection_duration": duration,
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received,
            "is_failed": is_failed,
            "has_dns_lookup": has_dns,
            "dst_domain": "",  # Will be mapped later
        })

    return pd.DataFrame(flow_records)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pcap_file(filepath: str) -> pd.DataFrame:
    """
    Parse a single PCAP/PCAPNG file into flow-level records.

    Reads all packets, extracts IP-layer information, aggregates into
    flows, and returns a DataFrame matching the synthetic_flows.csv schema.

    Args:
        filepath: Path to the .pcap or .pcapng file.

    Returns:
        pd.DataFrame with columns matching synthetic_flows.csv.

    Raises:
        RuntimeError: If scapy is not installed.
        FileNotFoundError: If the specified file does not exist.
    """
    if not SCAPY_AVAILABLE:
        raise RuntimeError(
            "scapy is required for PCAP parsing but is not installed. "
            "Install it with: pip install scapy>=2.5.0"
        )

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PCAP file not found: {filepath}")

    logger.info("Parsing PCAP file: %s", filepath.name)

    # Read all packets from the file
    try:
        packets = rdpcap(str(filepath))
    except Exception as exc:
        logger.error("Failed to read PCAP file %s: %s", filepath.name, exc)
        return pd.DataFrame()

    total_packets = len(packets)
    logger.info("  Total packets in file: %d", total_packets)

    # Extract per-packet info
    packet_records = []
    parse_errors = 0
    dns_map = {}

    for idx, packet in enumerate(packets):
        # --- DNS Extraction ---
        try:
            if DNS in packet and packet.haslayer(DNSRR):
                for x in range(packet[DNS].ancount):
                    try:
                        rr = packet[DNSRR][x]
                        if hasattr(rr, 'type') and rr.type == 1 and hasattr(rr, 'rdata'): # A record
                            ip_val = rr.rdata
                            if isinstance(ip_val, (str, bytes)):
                                if isinstance(ip_val, bytes):
                                    ip_val = ip_val.decode('utf-8')
                                domain = rr.rrname.decode('utf-8').rstrip('.') if isinstance(rr.rrname, bytes) else str(rr.rrname).rstrip('.')
                                dns_map[ip_val] = domain
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            info = _extract_packet_info(packet, idx)
            if info is not None:
                packet_records.append(info)
        except Exception as exc:
            parse_errors += 1
            if parse_errors <= 10:  # Log first 10 errors
                logger.warning("  Skipping malformed packet %d: %s", idx, exc)
            elif parse_errors == 11:
                logger.warning("  (Suppressing further malformed packet warnings)")

    parsed_count = len(packet_records)
    skipped = total_packets - parsed_count - parse_errors
    logger.info(
        "  Parsed: %d | Skipped (non-IP): %d | Errors: %d",
        parsed_count, skipped, parse_errors,
    )

    if not packet_records:
        logger.warning("  No IP packets extracted from %s", filepath.name)
        return pd.DataFrame()

    # Build per-packet DataFrame and aggregate to flows
    packets_df = pd.DataFrame(packet_records)
    flows_df = _aggregate_to_flows(packets_df)
    
    # Map DNS domains
    if not flows_df.empty and dns_map:
        flows_df['dst_domain'] = flows_df['dst_ip'].map(dns_map).fillna('')
        logger.info("  Mapped %d unique domains to IP flows", len(dns_map))

    logger.info("  Aggregated into %d flow records", len(flows_df))
    return flows_df


def parse_all_pcaps(
    pcap_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parse all PCAP/PCAPNG files in a directory and combine results.

    Scans the specified directory for .pcap and .pcapng files, parses
    each one, and returns a single combined DataFrame.

    Args:
        pcap_dir: Path to directory containing PCAP files.
                  Defaults to data/raw_pcap/.

    Returns:
        pd.DataFrame with all flows from all PCAP files combined.
    """
    if not SCAPY_AVAILABLE:
        raise RuntimeError(
            "scapy is required for PCAP parsing but is not installed. "
            "Install it with: pip install scapy>=2.5.0"
        )

    if pcap_dir is None:
        pcap_path = DEFAULT_PCAP_DIR
    else:
        pcap_path = Path(pcap_dir)

    if not pcap_path.exists():
        logger.warning("PCAP directory does not exist: %s", pcap_path)
        logger.info("Creating directory: %s", pcap_path)
        pcap_path.mkdir(parents=True, exist_ok=True)
        return pd.DataFrame()

    # Find all PCAP files
    pcap_files = sorted([
        f for f in pcap_path.iterdir()
        if f.suffix.lower() in PCAP_EXTENSIONS and f.is_file()
    ])

    if not pcap_files:
        logger.info("No PCAP files found in %s", pcap_path)
        logger.info(
            "Place .pcap or .pcapng files in %s and re-run.", pcap_path
        )
        return pd.DataFrame()

    logger.info("Found %d PCAP file(s) in %s", len(pcap_files), pcap_path)

    all_frames = []
    flow_id_offset = 0

    for pcap_file in pcap_files:
        try:
            df = parse_pcap_file(str(pcap_file))
            if not df.empty:
                # Re-index flow IDs to avoid collisions across files
                df["flow_id"] = [
                    f"pcap_flow_{flow_id_offset + i:06d}"
                    for i in range(len(df))
                ]
                flow_id_offset += len(df)
                all_frames.append(df)
        except Exception as exc:
            logger.error("Failed to parse %s: %s", pcap_file.name, exc)
            continue

    if not all_frames:
        logger.warning("No flows extracted from any PCAP files")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    logger.info("Total flows from all PCAPs: %d", len(combined))
    return combined


def save_parsed_flows(
    pcap_dir: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Optional[str]:
    """
    Parse all PCAPs in a directory and save the results to CSV.

    Args:
        pcap_dir: Path to directory containing PCAP files.
                  Defaults to data/raw_pcap/.
        output_path: Path for the output CSV file.
                     Defaults to data/processed/parsed_pcap_flows.csv.

    Returns:
        Path to the saved CSV file, or None if no data was parsed.
    """
    if output_path is None:
        out_path = DEFAULT_OUTPUT_PATH
    else:
        out_path = Path(output_path)

    flows_df = parse_all_pcaps(pcap_dir)

    if flows_df.empty:
        logger.info("No flows to save — output file not created")
        return None

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flows_df.to_csv(out_path, index=False)
    logger.info("Saved %d parsed flows to: %s", len(flows_df), out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("C2 Beaconing Detection Engine — PCAP Parser")
    print("=" * 70)

    if not SCAPY_AVAILABLE:
        print("\n[ERROR] scapy is not installed.")
        print("Install it with: pip install scapy>=2.5.0")
        print("\nThe PCAP parser requires scapy to read packet captures.")
        print("Other modules (synthetic generator, Kaggle loader) do not")
        print("require scapy and can be used independently.")
        exit(1)

    print(f"\nScanning for PCAP files in: {DEFAULT_PCAP_DIR}")

    result_path = save_parsed_flows()

    if result_path:
        print(f"\nParsed flows saved to: {result_path}")

        # Print summary
        df = pd.read_csv(result_path)
        print(f"\n{'─' * 70}")
        print("PARSE SUMMARY")
        print(f"{'─' * 70}")
        print(f"Total flow records: {len(df):,}")

        if not df.empty:
            print(f"\nProtocol distribution:")
            for proto, count in df["protocol"].value_counts().items():
                print(f"  {proto}: {count:,}")

            print(f"\nTop destination ports:")
            for port, count in df["dst_port"].value_counts().head(10).items():
                print(f"  port {port}: {count:,}")

            print(f"\nUnique source IPs: {df['src_ip'].nunique():,}")
            print(f"Unique destination IPs: {df['dst_ip'].nunique():,}")
            print(f"Packet size range: {df['packet_size'].min():,} - "
                  f"{df['packet_size'].max():,} bytes")
    else:
        print("\nNo PCAP files found or no flows could be extracted.")
        print(f"Place .pcap or .pcapng files in: {DEFAULT_PCAP_DIR}")

    print("\n" + "=" * 70)
    print("PCAP parsing complete.")
    print("=" * 70)
