import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import matplotlib.pyplot as plt

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.pcap_parser import parse_pcap_file
from src.detection_pipeline import run_pipeline


st.set_page_config(page_title="C2 Beaconing Detection Engine", layout="wide")

st.title("C2 Beaconing Detection Engine")
st.markdown("A multi-layered network security pipeline analyzing PCAP/flow data to detect Command-and-Control (C2) beaconing behavior using statistical timing analysis (CoV) and Machine Learning (Isolation Forest).")

@st.cache_data
def load_data():
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    results_path = os.path.join(base_dir, 'reports', 'detection_results.csv')
    flows_path = os.path.join(base_dir, 'data', 'processed', 'synthetic_flows.csv')
    
    if not os.path.exists(results_path) or not os.path.exists(flows_path):
        return None, None
        
    results_df = pd.read_csv(results_path)
    flows_df = pd.read_csv(flows_path, parse_dates=['timestamp'])
    return results_df, flows_df

results_df, flows_df = load_data()

# If no previous data exists, initialize empty DataFrames
if results_df is None:
    results_df = pd.DataFrame(columns=['src_ip', 'dst_ip', 'dst_domain', 'dst_port', 'num_connections', 'interval_cov', 'deterministic_score', 'ml_score', 'final_verdict', 'beacon_type_guess', 'ml_explanation'])
    flows_df = pd.DataFrame(columns=['src_ip', 'dst_ip', 'dst_port', 'timestamp', 'packet_size', 'protocol', 'has_dns_lookup', 'dst_domain'])
    st.info("👋 Welcome! No existing data found. Please **Upload a PCAP File** using the sidebar to begin your C2 beaconing analysis.")
    # We don't st.stop() because we want them to use the upload feature!

# --- PCAP UPLOAD ---
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload PCAP File", type=["pcap", "pcapng"])

if uploaded_file is not None:
    with st.spinner("Parsing PCAP and running detection pipeline (this may take a moment)..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
            
        try:
            uploaded_flows_df = parse_pcap_file(tmp_path)
            if uploaded_flows_df.empty:
                st.sidebar.error("No valid IP flows extracted from PCAP.")
            else:
                uploaded_flows_df['timestamp'] = pd.to_datetime(uploaded_flows_df['timestamp'])
                uploaded_results_df = run_pipeline(raw_df=uploaded_flows_df)
                if not uploaded_results_df.empty:
                    # Override the default data with the uploaded data
                    results_df = uploaded_results_df
                    flows_df = uploaded_flows_df
                    st.sidebar.success(f"Successfully analyzed {len(results_df)} flows from PCAP!")
                else:
                    st.sidebar.warning("No flows survived filtering (e.g. min 10 connections).")
        except Exception as e:
            st.sidebar.error(f"Error processing PCAP: {e}")
        finally:
            os.remove(tmp_path)

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")

verdicts = results_df['final_verdict'].unique().tolist()
selected_verdicts = st.sidebar.multiselect("Final Verdict", verdicts, default=verdicts)

if 'beacon_type_guess' in results_df.columns:
    beacon_types = results_df['beacon_type_guess'].unique().tolist()
    selected_beacon_types = st.sidebar.multiselect("Beacon Type Guess", beacon_types, default=beacon_types)
else:
    beacon_types = []
    selected_beacon_types = []

min_det_score = st.sidebar.slider("Min Deterministic Score", 0, 100, 0)
min_ml_score = st.sidebar.slider("Min ML Anomaly Score", 0, 100, 0)

# Apply filters
mask = (results_df['final_verdict'].isin(selected_verdicts if selected_verdicts else verdicts)) & \
       (results_df['deterministic_score'] >= min_det_score) & \
       (results_df['ml_score'] >= min_ml_score)

if beacon_types:
    mask = mask & (results_df['beacon_type_guess'].isin(selected_beacon_types if selected_beacon_types else beacon_types))

filtered_df = results_df[mask]

# --- SUMMARY METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Flows Analyzed", len(results_df))
col2.metric("HIGH Alerts", len(results_df[results_df['final_verdict'] == 'HIGH']))
col3.metric("MEDIUM Alerts", len(results_df[results_df['final_verdict'] == 'MEDIUM']))
col4.metric("SUPPRESSED (Allowlist)", len(results_df[results_df['final_verdict'] == 'SUPPRESSED']))

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Data Explorer", "Visualizations", "Host Investigation"])

with tab1:
    # --- SCORED DESTINATIONS TABLE ---
    st.subheader("Scored Destinations")
    
    col_table_head, col_download = st.columns([4, 1])
    with col_download:
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name='c2_alerts.csv',
            mime='text/csv',
        )
    display_cols = ['src_ip', 'dst_ip', 'dst_domain', 'dst_port', 'num_connections', 'interval_cov', 
                    'deterministic_score', 'ml_score', 'final_verdict']
    if 'beacon_type_guess' in results_df.columns:
        display_cols.append('beacon_type_guess')
        
    avail_cols = [c for c in display_cols if c in filtered_df.columns]

    def color_verdict(val):
        if val == 'HIGH':
            color = '#ff4b4b'
        elif val == 'MEDIUM':
            color = '#ffa421'
        elif val == 'SUPPRESSED':
            color = '#808495'
        else:
            color = '#21c354'
        return f'color: {color}; font-weight: bold'
        
    if not filtered_df.empty:
        st.dataframe(filtered_df[avail_cols].style.map(color_verdict, subset=['final_verdict']), use_container_width=True)
    else:
        st.dataframe(filtered_df[avail_cols], use_container_width=True)

    # --- INDIVIDUAL INVESTIGATION ---
    st.subheader("Investigate Specific Flow")

    if not filtered_df.empty:
        flow_options = filtered_df.apply(lambda r: f"{r['src_ip']} -> {r['dst_ip']}:{r['dst_port']}", axis=1).tolist()
        selected_flow = st.selectbox("Select a flow to investigate:", flow_options)
        
        idx = flow_options.index(selected_flow)
        row = filtered_df.iloc[idx]
        
        col_info, col_exp = st.columns(2)
        
        with col_info:
            st.markdown(f"**Verdict:** <span style='color:{color_verdict(row['final_verdict'])}'>{row['final_verdict']}</span>", unsafe_allow_html=True)
            st.write(f"**Deterministic Score:** {row['deterministic_score']:.1f}")
            st.write(f"**ML Score:** {row['ml_score']:.1f}")
            if 'beacon_type_guess' in row:
                st.write(f"**Beacon Type Guess:** {row['beacon_type_guess']}")
            if 'suggested_mitre_technique' in row:
                st.write(f"**Suggested MITRE:** {row['suggested_mitre_technique']}")
            
            if pd.notna(row.get('allowlist_reason')):
                st.warning(f"**Allowlist Match:** {row['allowlist_reason']}")
                
        with col_exp:
            if pd.notna(row['ml_explanation']) and row['ml_explanation'] != '':
                st.info(f"**Why Flagged (ML Anomaly):**\n\n{row['ml_explanation']}")
            else:
                st.write("No strong ML anomalies detected beyond baseline.")

        # Comb-pattern plot
        st.write("### Connection Timeline (Comb Pattern)")
        f_data = flows_df[(flows_df['src_ip'] == row['src_ip']) & 
                          (flows_df['dst_ip'] == row['dst_ip']) & 
                          (flows_df['dst_port'] == row['dst_port'])].copy()
        f_data = f_data.sort_values('timestamp')
        
        if len(f_data) > 1:
            times = (f_data['timestamp'] - f_data['timestamp'].min()).dt.total_seconds().values
            fig, ax = plt.subplots(figsize=(10, 2))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            
            # Color based on verdict
            if row['final_verdict'] == 'HIGH': c = 'red'
            elif row['final_verdict'] == 'MEDIUM': c = 'orange'
            else: c = 'green'
                
            ax.eventplot(times, lineoffsets=1, linelengths=0.8, colors=c)
            ax.set_yticks([])
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
                
            ax.set_xlabel("Time (seconds)", color='white')
            st.pyplot(fig)
        else:
            st.write("Not enough connections to plot timeline.")
            
        # Full Feature Breakdown
        st.write("### Full Feature Breakdown")
        st.json(row.to_dict())
    else:
        st.write("No flows match the current filters.")

with tab2:
    st.subheader("Visualizations")
    if not results_df.empty:
        # 1. CoV Scatter Plot
        st.write("#### Beacon Clustering: Timing Regularity vs Volume")
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 5))
        fig_scatter.patch.set_facecolor('#0e1117')
        ax_scatter.set_facecolor('#0e1117')
        
        benign = results_df[results_df['final_verdict'] == 'CLEAR']
        high = results_df[results_df['final_verdict'] == 'HIGH']
        med = results_df[results_df['final_verdict'] == 'MEDIUM']
        
        ax_scatter.scatter(benign['interval_cov'], benign['num_connections'], 
                    c='green', alpha=0.5, label='CLEAR', marker='o')
        ax_scatter.scatter(med['interval_cov'], med['num_connections'], 
                    c='orange', alpha=0.8, label='MEDIUM', marker='^')
        ax_scatter.scatter(high['interval_cov'], high['num_connections'], 
                    c='red', alpha=0.8, label='HIGH', marker='s')
        
        ax_scatter.axvline(x=0.5, color='white', linestyle='--', alpha=0.5, label='Deterministic Threshold')
        ax_scatter.set_xscale('log')
        ax_scatter.set_yscale('log')
        ax_scatter.set_xlabel('Interval CoV (Log Scale)', color='white')
        ax_scatter.set_ylabel('Num Connections (Log Scale)', color='white')
        
        ax_scatter.tick_params(colors='white')
        for spine in ax_scatter.spines.values():
            spine.set_color('white')
            
        ax_scatter.legend(facecolor='#0e1117', edgecolor='white', labelcolor='white')
        st.pyplot(fig_scatter)
        
        # 2. Score Distributions
        st.write("#### Score Distributions")
        fig_hist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig_hist.patch.set_facecolor('#0e1117')
        
        for ax in [ax1, ax2]:
            ax.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
                
        # Deterministic Score
        ax1.hist(benign['deterministic_score'], bins=30, alpha=0.6, color='green', label='CLEAR', density=False)
        ax1.hist(med['deterministic_score'], bins=30, alpha=0.6, color='orange', label='MEDIUM', density=False)
        ax1.hist(high['deterministic_score'], bins=30, alpha=0.6, color='red', label='HIGH', density=False)
        ax1.set_title('Deterministic Score', color='white')
        ax1.set_xlabel('Score', color='white')
        
        # ML Score
        ax2.hist(benign['ml_score'], bins=30, alpha=0.6, color='green', label='CLEAR', density=False)
        ax2.hist(med['ml_score'], bins=30, alpha=0.6, color='orange', label='MEDIUM', density=False)
        ax2.hist(high['ml_score'], bins=30, alpha=0.6, color='red', label='HIGH', density=False)
        ax2.set_title('ML Anomaly Score', color='white')
        ax2.set_xlabel('Score', color='white')
        
        ax1.legend(facecolor='#0e1117', edgecolor='white', labelcolor='white')
        ax2.legend(facecolor='#0e1117', edgecolor='white', labelcolor='white')
        
        st.pyplot(fig_hist)
    else:
        st.write("No data available for visualizations.")

with tab3:
    st.subheader("Host Investigation (Internal IPs)")
    if not results_df.empty:
        internal_ips = sorted(results_df['src_ip'].unique().tolist())
        selected_host = st.selectbox("Select an internal host (src_ip) to investigate:", internal_ips)
        
        host_df = results_df[results_df['src_ip'] == selected_host].copy()
        
        st.write(f"### Activity for {selected_host}")
        st.metric("Total Unique Destinations Contacted", len(host_df))
        
        # Display the host's flows
        st.write("#### Associated Destinations and DNS Requests")
        display_cols_host = ['dst_ip', 'dst_domain', 'dst_port', 'final_verdict', 'deterministic_score', 'ml_score', 'num_connections']
        if 'beacon_type_guess' in host_df.columns:
            display_cols_host.append('beacon_type_guess')
            
        avail_cols = [c for c in display_cols_host if c in host_df.columns]
        
        # We define color_verdict again or rely on the one defined in tab1?
        # Actually it's defined inside tab1 context, but python functions have module scope, 
        # so it's accessible here.
        st.dataframe(host_df[avail_cols].style.map(color_verdict, subset=['final_verdict']), use_container_width=True)
        
        st.write("#### DNS / Domain Breakdown")
        if 'dst_domain' in host_df.columns:
            # Filter out empty domains
            domain_counts = host_df[host_df['dst_domain'] != '']['dst_domain'].value_counts()
            if not domain_counts.empty:
                st.bar_chart(domain_counts)
            else:
                st.info("No associated DNS requests (domains) found for this host's flows (they might be direct-IP C2 or benign traffic without DNS).")
        else:
            st.warning("dst_domain data not available.")
            
        st.write("#### Full Flow Details")
        st.dataframe(host_df)
    else:
        st.write("No data available.")
