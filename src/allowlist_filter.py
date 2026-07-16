"""
allowlist_filter.py — Allowlist Filter for C2 Beaconing Detection Engine

Filters known-good destinations to reduce false positives in beaconing
detection. Maintains a configurable set of rules covering NTP servers,
major CDN/cloud providers, Windows Update endpoints, and widely
contacted internal services.

Design: Rules are defined in a top-level configuration dict so that
SOC analysts can easily add or modify allowlist entries without touching
the scoring logic.

Author: C2 Beaconing Detection Engine
License: MIT
"""

import logging
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# =========================================================================
# Allowlist Configuration
# =========================================================================
# Each rule is a dict with:
#   - 'name'       : human-readable label for audit trails
#   - 'match'      : callable(dst_ip, dst_port, num_internal_hosts) -> bool
#   - 'reason'     : explanation string returned when the rule matches
#
# Rules are evaluated in order; the FIRST matching rule wins.
# =========================================================================

# --- Known-good IP prefixes (simulated CIDR-like matching) ----------------
MICROSOFT_PREFIXES: List[str] = [
    '13.107.',   # Microsoft corporate
    '20.190.',   # Microsoft Azure AD / auth
    '52.96.',    # Microsoft Office 365
]

GOOGLE_PREFIXES: List[str] = [
    '172.217.',  # Google services
]

CLOUDFLARE_PREFIXES: List[str] = [
    '104.16.',   # Cloudflare CDN
]

FASTLY_PREFIXES: List[str] = [
    '151.101.',  # Fastly CDN
]

# Combined set of all known-good CDN / cloud prefixes
KNOWN_GOOD_PREFIXES: List[str] = (
    MICROSOFT_PREFIXES
    + GOOGLE_PREFIXES
    + CLOUDFLARE_PREFIXES
    + FASTLY_PREFIXES
)

# Threshold: if more than this many internal hosts contact a destination,
# it is considered a widely-used legitimate service.
WIDELY_CONTACTED_THRESHOLD: int = 5


def _ip_matches_prefix(ip: str, prefixes: List[str]) -> bool:
    """Return True if *ip* starts with any of the given prefixes."""
    return any(ip.startswith(p) for p in prefixes)


# ---------------------------------------------------------------------------
# Rule definitions (order matters — first match wins)
# ---------------------------------------------------------------------------
ALLOWLIST_RULES: List[Dict[str, Any]] = [
    {
        'name': 'NTP',
        'match': lambda ip, port, hosts: port == 123,
        'reason': 'NTP traffic (dst_port 123)',
    },
    {
        'name': 'Windows Update (Microsoft HTTPS)',
        'match': lambda ip, port, hosts: (
            port == 443
            and _ip_matches_prefix(ip, MICROSOFT_PREFIXES)
        ),
        'reason': 'Windows Update / Microsoft HTTPS endpoint',
    },
    {
        'name': 'Known CDN / Cloud Provider',
        'match': lambda ip, port, hosts: (
            _ip_matches_prefix(ip, KNOWN_GOOD_PREFIXES)
        ),
        'reason': 'Known CDN/cloud provider IP range',
    },
    {
        'name': 'Widely Contacted Destination',
        'match': lambda ip, port, hosts: (
            hosts > WIDELY_CONTACTED_THRESHOLD
        ),
        'reason': (
            f'Widely contacted destination '
            f'(> {WIDELY_CONTACTED_THRESHOLD} internal hosts)'
        ),
    },
]


# =========================================================================
# Core Functions
# =========================================================================

def check_allowlist(
    dst_ip: str,
    dst_port: int,
    num_internal_hosts: int = 1,
) -> Tuple[bool, str]:
    """Check whether a destination is allowlisted.

    Parameters
    ----------
    dst_ip : str
        Destination IP address.
    dst_port : int
        Destination port number.
    num_internal_hosts : int, optional
        Number of unique internal hosts that contact this destination.
        Used by the 'widely contacted' rule. Default is 1.

    Returns
    -------
    (is_allowlisted, reason) : Tuple[bool, str]
        Whether the destination matched any allowlist rule and the
        human-readable reason (empty string if not allowlisted).
    """
    for rule in ALLOWLIST_RULES:
        try:
            if rule['match'](dst_ip, dst_port, num_internal_hosts):
                logger.debug(
                    "Allowlisted: %s:%d — rule '%s'",
                    dst_ip, dst_port, rule['name'],
                )
                return True, rule['reason']
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Allowlist rule '%s' raised an exception: %s",
                rule['name'], exc,
            )

    return False, ''


def apply_allowlist(
    features_df: pd.DataFrame,
    hosts_column: str = 'num_internal_hosts_contacting',
) -> pd.DataFrame:
    """Apply allowlist rules to every row in a features DataFrame.

    Parameters
    ----------
    features_df : pd.DataFrame
        Must contain columns ``dst_ip`` and ``dst_port``.  Optionally
        contains *hosts_column* (defaults to 1 when missing).

    Returns
    -------
    pd.DataFrame
        Copy of *features_df* with two new columns:
        - ``destination_in_allowlist`` (bool)
        - ``allowlist_reason`` (str)
    """
    df = features_df.copy()

    # Ensure required columns exist
    for col in ('dst_ip', 'dst_port'):
        if col not in df.columns:
            raise ValueError(f"features_df is missing required column: {col}")

    has_hosts_col = hosts_column in df.columns

    allowlisted_flags: List[bool] = []
    reasons: List[str] = []

    for _, row in df.iterrows():
        num_hosts = int(row[hosts_column]) if has_hosts_col else 1
        is_al, reason = check_allowlist(
            dst_ip=str(row['dst_ip']),
            dst_port=int(row['dst_port']),
            num_internal_hosts=num_hosts,
        )
        allowlisted_flags.append(is_al)
        reasons.append(reason)

    df['destination_in_allowlist'] = allowlisted_flags
    df['allowlist_reason'] = reasons

    num_suppressed = sum(allowlisted_flags)
    logger.info(
        "Allowlist applied: %d / %d flows allowlisted (%.1f%%).",
        num_suppressed,
        len(df),
        100.0 * num_suppressed / max(len(df), 1),
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
    print("ALLOWLIST FILTER — TEST CASES")
    print("=" * 72)

    test_cases = [
        # (dst_ip, dst_port, num_internal_hosts, expected_allowlisted)
        ('192.168.1.50', 123, 1, True),      # NTP
        ('13.107.42.14', 443, 1, True),       # Windows Update (MS HTTPS)
        ('13.107.42.14', 80, 1, True),        # Microsoft IP (CDN rule)
        ('172.217.14.99', 443, 1, True),      # Google CDN
        ('104.16.51.111', 443, 1, True),      # Cloudflare
        ('151.101.1.69', 443, 1, True),       # Fastly
        ('10.20.30.40', 8443, 8, True),       # Widely contacted
        ('45.33.32.156', 443, 1, False),      # Unknown — should NOT match
        ('192.168.1.1', 80, 2, False),        # Internal, few hosts
    ]

    all_passed = True
    for dst_ip, dst_port, hosts, expected in test_cases:
        is_al, reason = check_allowlist(dst_ip, dst_port, hosts)
        status = "PASS" if is_al == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(
            f"  [{status}]  {dst_ip}:{dst_port}  hosts={hosts}  "
            f"allowlisted={is_al}  reason='{reason}'"
        )

    print()
    if all_passed:
        print("All test cases PASSED.")
    else:
        print("Some test cases FAILED — review output above.")

    # --- Test apply_allowlist with a small DataFrame ---------------------
    print("\n--- Testing apply_allowlist on DataFrame ---")
    sample_df = pd.DataFrame([
        {'dst_ip': '13.107.42.14', 'dst_port': 443, 'num_internal_hosts_contacting': 3},
        {'dst_ip': '45.33.32.156', 'dst_port': 8080, 'num_internal_hosts_contacting': 1},
        {'dst_ip': '192.168.1.1', 'dst_port': 123, 'num_internal_hosts_contacting': 1},
    ])
    result = apply_allowlist(sample_df)
    print(result[['dst_ip', 'dst_port', 'destination_in_allowlist', 'allowlist_reason']].to_string(index=False))
    print("=" * 72)
