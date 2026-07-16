import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

def load_ground_truth(path):
    """Load ground truth CSV."""
    return pd.read_csv(path)

def merge_results_with_ground_truth(results_df, ground_truth_df):
    """Merge pipeline results with ground truth labels."""
    # Ensure flow_id exists in both or merge on src, dst, port
    # In synthetic data, ground truth is one row per flow.
    # Group truth by flow definition just to be safe
    gt_dedup = ground_truth_df.drop_duplicates(subset=['src_ip', 'dst_ip', 'dst_port'])
    merged = pd.merge(results_df, gt_dedup[['src_ip', 'dst_ip', 'dst_port', 'is_malicious', 'beacon_type', 'mitre_technique_id']], 
                      on=['src_ip', 'dst_ip', 'dst_port'], how='inner')
    return merged

def compute_metrics(merged_df, y_true_col='is_malicious', y_pred_col=None, y_pred_array=None):
    """Compute standard metrics."""
    y_true = merged_df[y_true_col].values
    
    if y_pred_array is not None:
        y_pred = y_pred_array
    else:
        y_pred = merged_df[y_pred_col].values
        
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    if len(y_true) > 0 and len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    else:
        # Fallback if only one class present
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fp_rate': fp_rate,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }

def evaluate_deterministic(merged_df, threshold=75):
    """Evaluate deterministic engine only (score > threshold)."""
    # Note: allowlisted items should have score 0, but just in case, we also check final_verdict != SUPPRESSED
    # if we want strict deterministic isolation. Here we just use the score.
    preds = (merged_df['deterministic_score'] > threshold).astype(int)
    return compute_metrics(merged_df, y_pred_array=preds)

def evaluate_ml(merged_df, threshold=70):
    """Evaluate ML engine only (ml_score > threshold)."""
    preds = (merged_df['ml_score'] > threshold).astype(int)
    return compute_metrics(merged_df, y_pred_array=preds)

def evaluate_combined(merged_df):
    """Evaluate combined pipeline (final_verdict is HIGH or MEDIUM)."""
    preds = merged_df['final_verdict'].isin(['HIGH', 'MEDIUM']).astype(int)
    return compute_metrics(merged_df, y_pred_array=preds)

def breakdown_by_beacon_type(merged_df):
    """Calculate recall for each specific beacon type."""
    metrics_by_type = {}
    preds = merged_df['final_verdict'].isin(['HIGH', 'MEDIUM']).astype(int)
    merged_df = merged_df.copy()
    merged_df['prediction'] = preds
    
    for btype in merged_df['beacon_type'].unique():
        if btype == 'none' or pd.isna(btype):
            continue
            
        type_df = merged_df[merged_df['beacon_type'] == btype]
        if len(type_df) > 0:
            recall = np.mean(type_df['prediction'])  # Since true label is 1, mean(pred) is recall
            metrics_by_type[btype] = {'recall': recall, 'count': len(type_df)}
            
    return metrics_by_type

def false_positive_analysis(merged_df):
    """Analyze false positives, specifically checking NTP and Windows Update."""
    preds = merged_df['final_verdict'].isin(['HIGH', 'MEDIUM']).astype(int)
    fp_df = merged_df[(merged_df['is_malicious'] == 0) & (preds == 1)].copy()
    
    return fp_df

def generate_confusion_matrix_plot(merged_df, y_pred_array, title, output_path):
    """Generate and save confusion matrix plot."""
    y_true = merged_df['is_malicious'].values
    cm = confusion_matrix(y_true, y_pred_array)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malicious'])
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def generate_all_evaluation_plots(merged_df, output_dir):
    """Generate confusion matrices for all 3 methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Deterministic
    preds_det = (merged_df['deterministic_score'] > 75).astype(int)
    generate_confusion_matrix_plot(merged_df, preds_det, "Confusion Matrix: Deterministic Only", 
                                  os.path.join(output_dir, 'confusion_matrix_deterministic.png'))
    
    # ML
    preds_ml = (merged_df['ml_score'] > 70).astype(int)
    generate_confusion_matrix_plot(merged_df, preds_ml, "Confusion Matrix: ML Only", 
                                  os.path.join(output_dir, 'confusion_matrix_ml.png'))
    
    # Combined
    preds_comb = merged_df['final_verdict'].isin(['HIGH', 'MEDIUM']).astype(int)
    generate_confusion_matrix_plot(merged_df, preds_comb, "Confusion Matrix: Combined Pipeline", 
                                  os.path.join(output_dir, 'confusion_matrix_combined.png'))

def run_full_evaluation(results_df, ground_truth_path, output_dir):
    """Run full evaluation suite and print report."""
    logger.info(f"Loading ground truth from {ground_truth_path}")
    gt_df = load_ground_truth(ground_truth_path)
    
    merged = merge_results_with_ground_truth(results_df, gt_df)
    logger.info(f"Merged evaluation dataset: {len(merged)} flows")
    
    det_metrics = evaluate_deterministic(merged)
    ml_metrics = evaluate_ml(merged)
    comb_metrics = evaluate_combined(merged)
    
    print("\n--- DETECTION PERFORMANCE ---")
    print(f"{'Method':<20} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'FP Rate':<10}")
    print("-" * 70)
    print(f"{'Deterministic Only':<20} | {det_metrics['precision']:.4f}     | {det_metrics['recall']:.4f}     | {det_metrics['f1']:.4f}     | {det_metrics['fp_rate']:.4f}")
    print(f"{'ML Only':<20} | {ml_metrics['precision']:.4f}     | {ml_metrics['recall']:.4f}     | {ml_metrics['f1']:.4f}     | {ml_metrics['fp_rate']:.4f}")
    print(f"{'Combined Pipeline':<20} | {comb_metrics['precision']:.4f}     | {comb_metrics['recall']:.4f}     | {comb_metrics['f1']:.4f}     | {comb_metrics['fp_rate']:.4f}")
    
    print("\n--- RECALL BY BEACON TYPE (COMBINED PIPELINE) ---")
    type_metrics = breakdown_by_beacon_type(merged)
    for btype, metrics in type_metrics.items():
        print(f"{btype:<20}: {metrics['recall']:.4f} ({int(metrics['recall']*metrics['count'])}/{metrics['count']})")
        
    print("\n--- FALSE POSITIVE ANALYSIS ---")
    fps = false_positive_analysis(merged)
    print(f"Total False Positives: {len(fps)}")
    if len(fps) > 0:
        print("\nTop 5 False Positives:")
        print(fps[['src_ip', 'dst_ip', 'dst_port', 'deterministic_score', 'ml_score', 'final_verdict', 'ml_explanation']].head())
    
    # Check NTP
    ntp_flows = merged[(merged['dst_port'] == 123) & (merged['is_malicious'] == 0)]
    ntp_fps = ntp_flows[ntp_flows['final_verdict'].isin(['HIGH', 'MEDIUM'])]
    print(f"\nNTP flows analyzed: {len(ntp_flows)}")
    print(f"NTP false positives: {len(ntp_fps)}")
    if len(ntp_flows) > 0:
        avg_det_score = ntp_flows['deterministic_score'].mean()
        print(f"Average deterministic score for NTP: {avg_det_score:.1f} (Without allowlist, these would be alerts)")
        
    print("\nGenerating confusion matrix plots...")
    generate_all_evaluation_plots(merged, output_dir)
    print(f"Plots saved to {output_dir}")
    
    fps.to_csv(os.path.join(output_dir, 'false_positive_analysis.csv'), index=False)
    
    return {
        'deterministic': det_metrics,
        'ml': ml_metrics,
        'combined': comb_metrics,
        'by_type': type_metrics
    }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    results_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'detection_results.csv')
    gt_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'synthetic_ground_truth.csv')
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
    
    if os.path.exists(results_path) and os.path.exists(gt_path):
        results_df = pd.read_csv(results_path)
        run_full_evaluation(results_df, gt_path, out_dir)
    else:
        print("Missing data files. Run detection_pipeline.py first.")
