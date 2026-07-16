# C2 Beaconing Detection Engine — Analysis Report

## Executive Summary
This project implements a multi-layered detection engine for Command-and-Control (C2) beaconing behavior in network traffic. Using statistical timing analysis (Coefficient of Variation) combined with Isolation Forest anomaly detection, the engine identifies malware 'phoning home' at regular intervals. Testing against synthetic traffic simulating 3 distinct C2 beacon patterns achieved 80.4% precision and 100.0% recall on the combined pipeline. The allowlist layer successfully suppressed NTP false positives while maintaining detection coverage.

## Methodology

### Data Sources
- **Synthetic traffic simulation**: 300+ benign flows (web browsing, streaming, NTP, Windows Update) and 40-50 malicious beaconing flows across 3 C2 patterns
- **Kaggle CICIDS2017 supplement**: Used ONLY benign-labeled rows to enrich the normal traffic baseline for ML training. This dataset does NOT contain C2 beaconing labels — it supplements benign traffic diversity only. No major public Kaggle dataset specifically labels beaconing behavior; we are explicit and honest about this limitation.

### Detection Architecture
The engine uses a 3-layer approach:
1. **Deterministic Scoring**: Weights heavily toward low timing variance (CoV), producing a baseline score.
2. **ML Anomaly Layer**: An Isolation Forest trained on combined benign traffic flags evasive/jittered behaviors that evade pure deterministic thresholds.
3. **Allowlist Filter**: Suppresses well-known periodic benign traffic (e.g., NTP, Windows Update).

### Feature Engineering
19 statistical features are extracted per flow group, including:
- **Timing**: Interval CoV, Mean Interval, MAD, Autocorrelation Score
- **Volume**: Bytes CoV, Payload Entropy, Burst Detection
- **Connection**: Unique Ports, Session Duration
- **Destination**: Direct IP access (no DNS), Low internal host count

## MITRE ATT&CK Mapping

| Technique ID | Technique Name | Relevance |
|---|---|---|
| T1071.001 | Application Layer Protocol: Web Protocols | C2 communication over HTTP/HTTPS |
| T1573 | Encrypted Channel | Beaconing over port 443 (TLS) |
| T1029 | Scheduled Transfer | Regular-interval data exfiltration following beacon sequences |
| T1568 | Dynamic Resolution | Future extension: DGA domain detection |

## Results

### Detection Performance

| Method | Precision | Recall | F1 Score | False Positive Rate |
|---|---|---|---|---|
| Deterministic Only | 1.0000 | 0.2667 | 0.4211 | 0.0000 |
| ML Only | 0.8958 | 0.9556 | 0.9247 | 0.1316 |
| Combined Pipeline | 0.8036 | 1.0000 | 0.8911 | 0.2895 |

### Detection by Beacon Type

| Beacon Type | Recall | Notes |
|---|---|---|
| Fixed-interval | 100.0% | Easiest to detect — near-zero CoV is a strong signal |
| Jittered (Cobalt Strike-style) | 100.0% | Moderate difficulty — CoV still lower than legitimate traffic |
| Evasive long-sleep | 100.0% | Hardest — fewer data points, longer intervals blend with legitimate low-frequency traffic |

## False Positive Analysis

### NTP Traffic (Port 123)
NTP has regular ~64-second intervals with very low CoV, making it look exactly like beaconing. The allowlist layer suppresses NTP destinations, preventing false alerts. Without the allowlist, NTP traffic would score 63.4 on the deterministic engine.

### Windows Update / Telemetry
Periodic telemetry check-ins can exhibit beacon-like behavior. The allowlist filters known Microsoft autonomous system blocks to suppress these.

## Industry Context: RITA Comparison
RITA (Real Intelligence Threat Analytics), an open-source tool by Active Countermeasures, uses a similar CoV-based approach for beacon detection in production environments. Our implementation follows the same fundamental principle — that beaconing traffic exhibits unusually low timing variance — while adding an ML anomaly layer for improved coverage of jittered/evasive patterns that pure CoV thresholds may miss.

## Limitations
- **Synthetic data**: All malicious traffic is synthetic. Real C2 beacons (e.g., Cobalt Strike, Metasploit) may exhibit patterns not captured here.
- **No real PCAP validation**: The pipeline supports real PCAP input but has not been validated against known-malicious captures (e.g., from malware-traffic-analysis.net).
- **Kaggle dataset scope**: CICIDS2017 provides benign traffic diversity only — it does not contain C2 beaconing labels.
- **No TLS inspection**: JA3/JA3S fingerprinting not implemented; encrypted beacon payloads are not analyzed.
- **No DNS analysis**: DGA (Domain Generation Algorithm) detection not implemented.
- **Static allowlist**: Real deployment would need dynamic allowlist maintenance.

## Future Improvements
1. **JA3/JA3S TLS fingerprinting** — detect known C2 framework TLS signatures
2. **DNS tunneling / DGA detection** — identify algorithmically generated domains used by C2
3. **Real PCAP validation** — test against malware-traffic-analysis.net samples
4. **WHOIS domain-age enrichment** — newly registered domains are suspicious
5. **Adaptive thresholds** — ML-driven threshold tuning based on environment baseline
6. **Integration with SIEM** — output in CEF/LEEF format for Splunk/QRadar ingestion
