import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from evaluate import load_ground_truth, evaluate_combined, evaluate_deterministic, evaluate_ml

logger = logging.getLogger(__name__)

def load_data():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    flows_path = os.path.join(base_dir, 'data', 'processed', 'synthetic_flows.csv')
    gt_path = os.path.join(base_dir, 'data', 'processed', 'synthetic_ground_truth.csv')
    results_path = os.path.join(base_dir, 'reports', 'detection_results.csv')
    
    if not (os.path.exists(flows_path) and os.path.exists(gt_path) and os.path.exists(results_path)):
        logger.error("Missing data files. Run detection pipeline first.")
        return None, None, None
        
    flows = pd.read_csv(flows_path, parse_dates=['timestamp'])
    gt = load_ground_truth(gt_path)
    results = pd.read_csv(results_path)
    return flows, gt, results

def plot_comb_patterns(flows_df, gt_df, output_path):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Select one of each type
    benign = gt_df[gt_df['beacon_type'] == 'none'].sample(1, random_state=42).iloc[0]
    fixed = gt_df[gt_df['beacon_type'] == 'fixed'].sample(1, random_state=42).iloc[0]
    jittered = gt_df[gt_df['beacon_type'] == 'jittered'].sample(1, random_state=42).iloc[0]
    
    targets = [
        (benign, "Benign Bursty Web Browsing", axes[0], 'green'),
        (fixed, "Fixed-Interval Beacon (Malicious)", axes[1], 'red'),
        (jittered, "Jittered Beacon (Malicious)", axes[2], 'orange')
    ]
    
    for target_row, title, ax, color in targets:
        f = flows_df[(flows_df['src_ip'] == target_row['src_ip']) & 
                     (flows_df['dst_ip'] == target_row['dst_ip']) & 
                     (flows_df['dst_port'] == target_row['dst_port'])].copy()
        f = f.sort_values('timestamp')
        
        # Calculate relative time in seconds from first connection
        times = (f['timestamp'] - f['timestamp'].min()).dt.total_seconds().values
        
        if len(times) > 1:
            intervals = np.diff(times)
            mean_int = np.mean(intervals)
            cov = np.std(intervals) / mean_int if mean_int > 0 else 0
            title_ext = f"{title}\nCoV: {cov:.3f} | Mean Interval: {mean_int:.1f}s | {len(times)} connections"
        else:
            title_ext = title
            
        ax.eventplot(times, lineoffsets=1, linelengths=0.8, colors=color)
        ax.set_title(title_ext, fontsize=10, pad=10)
        ax.set_yticks([])
        ax.grid(True, alpha=0.2)
        
    axes[2].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_cov_scatter(results_df, gt_df, output_path):
    plt.style.use('dark_background')
    
    merged = pd.merge(results_df, gt_df[['src_ip', 'dst_ip', 'dst_port', 'is_malicious', 'beacon_type']],
                      on=['src_ip', 'dst_ip', 'dst_port'])
    
    plt.figure(figsize=(10, 6))
    
    benign = merged[merged['is_malicious'] == 0]
    fixed = merged[merged['beacon_type'] == 'fixed']
    jittered = merged[merged['beacon_type'] == 'jittered']
    evasive = merged[merged['beacon_type'] == 'evasive']
    
    plt.scatter(benign['interval_cov'], benign['num_connections'], 
                c='green', alpha=0.5, label='Benign', marker='o')
    plt.scatter(fixed['interval_cov'], fixed['num_connections'], 
                c='red', alpha=0.8, label='Fixed Beacon', marker='s')
    plt.scatter(jittered['interval_cov'], jittered['num_connections'], 
                c='orange', alpha=0.8, label='Jittered Beacon', marker='^')
    plt.scatter(evasive['interval_cov'], evasive['num_connections'], 
                c='magenta', alpha=0.8, label='Evasive Beacon', marker='D')
    
    plt.axvline(x=0.5, color='white', linestyle='--', alpha=0.5, label='Approx. Decision Boundary')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Interval Coefficient of Variation (CoV) - Log Scale')
    plt.ylabel('Number of Connections - Log Scale')
    plt.title('Beacon Clustering: Timing Regularity vs Volume')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_feature_distributions(results_df, gt_df, output_path):
    plt.style.use('dark_background')
    
    merged = pd.merge(results_df, gt_df[['src_ip', 'dst_ip', 'dst_port', 'is_malicious']],
                      on=['src_ip', 'dst_ip', 'dst_port'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    benign = merged[merged['is_malicious'] == 0]
    malicious = merged[merged['is_malicious'] == 1]
    
    # 1. interval_cov
    axes[0, 0].hist(benign['interval_cov'].clip(upper=2.0), bins=30, alpha=0.6, color='blue', label='Benign', density=True)
    axes[0, 0].hist(malicious['interval_cov'].clip(upper=2.0), bins=30, alpha=0.6, color='red', label='Malicious', density=True)
    axes[0, 0].set_title('Interval CoV Distribution')
    axes[0, 0].legend()
    
    # 2. mean_interval
    # We might not have bytes_cov in results_df, let's just plot deterministic_score and ml_score if available
    # Or just use mean_interval
    axes[0, 1].hist(np.log1p(benign['mean_interval']), bins=30, alpha=0.6, color='blue', label='Benign', density=True)
    axes[0, 1].hist(np.log1p(malicious['mean_interval']), bins=30, alpha=0.6, color='red', label='Malicious', density=True)
    axes[0, 1].set_title('Log(Mean Interval) Distribution')
    axes[0, 1].legend()
    
    # 3. Deterministic Score
    axes[1, 0].hist(benign['deterministic_score'], bins=30, alpha=0.6, color='blue', label='Benign', density=True)
    axes[1, 0].hist(malicious['deterministic_score'], bins=30, alpha=0.6, color='red', label='Malicious', density=True)
    axes[1, 0].set_title('Deterministic Score Distribution')
    axes[1, 0].legend()
    
    # 4. ML Score
    axes[1, 1].hist(benign['ml_score'], bins=30, alpha=0.6, color='blue', label='Benign', density=True)
    axes[1, 1].hist(malicious['ml_score'], bins=30, alpha=0.6, color='red', label='Malicious', density=True)
    axes[1, 1].set_title('ML Anomaly Score Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_detection_summary(results_df, gt_df, output_path):
    plt.style.use('dark_background')
    
    gt_dedup = gt_df.drop_duplicates(subset=['src_ip', 'dst_ip', 'dst_port'])
    merged = pd.merge(results_df, gt_dedup[['src_ip', 'dst_ip', 'dst_port', 'is_malicious', 'beacon_type', 'mitre_technique_id']], 
                      on=['src_ip', 'dst_ip', 'dst_port'], how='inner')
    
    det_m = evaluate_deterministic(merged)
    ml_m = evaluate_ml(merged)
    comb_m = evaluate_combined(merged)
    
    labels = ['Precision', 'Recall', 'F1 Score']
    det_vals = [det_m['precision'], det_m['recall'], det_m['f1']]
    ml_vals = [ml_m['precision'], ml_m['recall'], ml_m['f1']]
    comb_vals = [comb_m['precision'], comb_m['recall'], comb_m['f1']]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, det_vals, width, label='Deterministic')
    rects2 = ax.bar(x, ml_vals, width, label='ML Anomaly')
    rects3 = ax.bar(x + width, comb_vals, width, label='Combined')
    
    ax.set_ylabel('Score')
    ax.set_title('Detection Performance by Method')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower center')
    ax.set_ylim(0, 1.1)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    
    flows, gt, results = load_data()
    
    if flows is not None:
        logger.info("Generating comb pattern timelines...")
        plot_comb_patterns(flows, gt, os.path.join(out_dir, 'comb_pattern_timelines.png'))
        
        logger.info("Generating CoV scatter plot...")
        plot_cov_scatter(results, gt, os.path.join(out_dir, 'cov_scatter.png'))
        
        logger.info("Generating feature distribution histograms...")
        plot_feature_distributions(results, gt, os.path.join(out_dir, 'feature_distributions.png'))
        
        logger.info("Generating detection summary chart...")
        plot_detection_summary(results, gt, os.path.join(out_dir, 'detection_summary.png'))
        
        logger.info(f"All plots saved to {out_dir}")
