#!/usr/bin/env python3
"""
Generate comprehensive reports and visualizations from evaluation results.
"""

import argparse
import sys
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import load_json, setup_logging

def load_evaluation_data(data_dir: str = "data") -> dict:
    """Load all evaluation data files."""
    data = {}
    
    files_to_load = [
        "evaluation_report.json",
        "performance_summary.json", 
        "final_performance.json",
        "calibration_report.json",
        "canonical_metrics.json",
        "tasks.json"
    ]
    
    for filename in files_to_load:
        filepath = os.path.join(data_dir, filename)
        try:
            data[filename.replace('.json', '')] = load_json(filepath)
        except FileNotFoundError:
            print(f"Warning: {filename} not found, skipping...")
            data[filename.replace('.json', '')] = None
    
    return data

def create_performance_charts(data: dict, output_dir: str = "reports"):
    """Create performance visualization charts."""
    os.makedirs(output_dir, exist_ok=True)
    
    if not data['final_performance']:
        print("No performance data available for charts")
        return
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Overall Performance Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = list(data['final_performance'].keys())
    scores = list(data['final_performance'].values())
    
    bars = ax.bar(metrics, scores, alpha=0.8)
    ax.set_ylabel('Score (0-1)')
    ax.set_title('Overall Performance by Metric')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Tier Performance Comparison
    if data['performance_summary'] and data['performance_summary'].get('tier_performance'):
        tier_data = data['performance_summary']['tier_performance']
        
        # Create DataFrame for easier plotting
        tier_df = pd.DataFrame(tier_data).T
        
        fig, ax = plt.subplots(figsize=(12, 8))
        tier_df.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_ylabel('Score (0-1)')
        ax.set_title('Performance by Task Tier')
        ax.set_ylim(0, 1)
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tier_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Performance Distribution
    if data['performance_summary'] and data['performance_summary'].get('statistics'):
        stats = data['performance_summary']['statistics']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot of metric scores
        metric_names = list(stats.keys())
        metric_stats = [stats[m] for m in metric_names]
        
        box_data = []
        labels = []
        for i, (name, stat) in enumerate(zip(metric_names, metric_stats)):
            # Create synthetic data points for box plot
            mean = stat['mean']
            std = stat['std']
            q25 = stat['q25']
            q75 = stat['q75']
            
            # Generate points around the quartiles
            points = [stat['min'], q25, mean, q75, stat['max']]
            box_data.append(points)
            labels.append(name)
        
        ax1.boxplot(box_data, labels=labels)
        ax1.set_ylabel('Score (0-1)')
        ax1.set_title('Score Distribution by Metric')
        ax1.tick_params(axis='x', rotation=45)
        
        # Histogram of all scores
        all_means = [stat['mean'] for stat in metric_stats]
        ax2.hist(all_means, bins=10, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Score (0-1)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Metric Means')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/score_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_calibration_charts(data: dict, output_dir: str = "reports"):
    """Create calibration and reliability charts."""
    if not data['calibration_report']:
        print("No calibration data available for charts")
        return
    
    calibration = data['calibration_report']
    
    # 1. Judge Bias Heatmap
    if calibration.get('bias_analysis', {}).get('judge_biases'):
        bias_data = calibration['bias_analysis']['judge_biases']
        
        # Convert to DataFrame
        bias_df = pd.DataFrame(bias_data).T
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(bias_df, annot=True, cmap='RdBu_r', center=0, 
                   fmt='.3f', ax=ax, cbar_kws={'label': 'Bias'})
        ax.set_title('Judge Bias by Metric')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Judges')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/judge_bias_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Inter-Judge Agreement
    if calibration.get('agreement_analysis', {}).get('metric_agreements'):
        agreement_data = calibration['agreement_analysis']['metric_agreements']
        
        metrics = list(agreement_data.keys())
        kappa_scores = [agreement_data[m].get('cohens_kappa', 0) for m in metrics]
        pearson_scores = [agreement_data[m].get('pearson_correlation', 0) for m in metrics]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, kappa_scores, width, label="Cohen's κ", alpha=0.8)
        ax.bar(x + width/2, pearson_scores, width, label='Pearson r', alpha=0.8)
        
        ax.set_ylabel('Agreement Score')
        ax.set_title('Inter-Judge Agreement by Metric')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/judge_agreement.png", dpi=300, bbox_inches='tight')
        plt.close()

def generate_html_report(data: dict, output_dir: str = "reports") -> str:
    """Generate an HTML report with all results."""
    report_path = os.path.join(output_dir, "evaluation_report.html")
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AgEval - Three-Judge Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 30px 0; }}
        .metric-table {{ border-collapse: collapse; width: 100%; }}
        .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .metric-table th {{ background-color: #f2f2f2; }}
        .chart {{ text-align: center; margin: 20px 0; }}
        .warning {{ color: #d9534f; font-weight: bold; }}
        .success {{ color: #5cb85c; font-weight: bold; }}
        .info {{ color: #5bc0de; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AgEval - Three-Judge Evaluation Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        {f'<p><strong>Evaluation ID:</strong> {data["evaluation_report"]["evaluation_id"]}</p>' if data.get("evaluation_report") else ''}
        {f'<p><strong>Agent:</strong> {data["evaluation_report"]["agent_info"]["name"]} ({data["evaluation_report"]["agent_info"]["model"]})</p>' if data.get("evaluation_report") else ''}
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        {generate_summary_section(data)}
    </div>
    
    <div class="section">
        <h2>Performance Results</h2>
        {generate_performance_section(data)}
    </div>
    
    <div class="section">
        <h2>Calibration & Reliability</h2>
        {generate_calibration_section(data)}
    </div>
    
    <div class="section">
        <h2>Visualizations</h2>
        <div class="chart">
            <h3>Overall Performance</h3>
            <img src="overall_performance.png" alt="Overall Performance Chart" style="max-width: 100%;">
        </div>
        <div class="chart">
            <h3>Performance by Task Tier</h3>
            <img src="tier_performance.png" alt="Tier Performance Chart" style="max-width: 100%;">
        </div>
        <div class="chart">
            <h3>Judge Bias Analysis</h3>
            <img src="judge_bias_heatmap.png" alt="Judge Bias Heatmap" style="max-width: 100%;">
        </div>
        <div class="chart">
            <h3>Inter-Judge Agreement</h3>
            <img src="judge_agreement.png" alt="Judge Agreement Chart" style="max-width: 100%;">
        </div>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        {generate_recommendations_section(data)}
    </div>
    
</body>
</html>
"""
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return report_path

def generate_summary_section(data: dict) -> str:
    """Generate the executive summary section."""
    if not data.get('performance_summary'):
        return "<p>No summary data available.</p>"
    
    summary = data['performance_summary']['summary_metrics']
    
    return f"""
    <table class="metric-table">
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Overall Mean Score</td><td>{summary.get('overall_mean', 0):.3f}</td></tr>
        <tr><td>Overall Median Score</td><td>{summary.get('overall_median', 0):.3f}</td></tr>
        <tr><td>Metrics Above 70%</td><td>{summary.get('metrics_above_70', 0)}</td></tr>
        <tr><td>Metrics Below 50%</td><td>{summary.get('metrics_below_50', 0)}</td></tr>
        <tr><td>Score Standard Deviation</td><td>{summary.get('overall_std', 0):.3f}</td></tr>
    </table>
    """

def generate_performance_section(data: dict) -> str:
    """Generate the performance results section."""
    if not data.get('final_performance'):
        return "<p>No performance data available.</p>"
    
    performance = data['final_performance']
    
    html = '<table class="metric-table"><tr><th>Metric</th><th>Score</th><th>Status</th></tr>'
    
    for metric, score in performance.items():
        if score >= 0.7:
            status = '<span class="success">Good</span>'
        elif score >= 0.5:
            status = '<span class="info">Fair</span>'
        else:
            status = '<span class="warning">Poor</span>'
        
        html += f'<tr><td>{metric}</td><td>{score:.3f}</td><td>{status}</td></tr>'
    
    html += '</table>'
    return html

def generate_calibration_section(data: dict) -> str:
    """Generate the calibration analysis section."""
    if not data.get('calibration_report'):
        return "<p>No calibration data available.</p>"
    
    calibration = data['calibration_report']
    
    html = "<h3>Bias Analysis</h3>"
    large_biases = calibration.get('bias_analysis', {}).get('large_biases', [])
    
    if large_biases:
        html += '<p class="warning">Large biases detected:</p><ul>'
        for bias in large_biases:
            html += f'<li>{bias["judge"]} on {bias["metric"]}: {bias["bias"]:.3f}</li>'
        html += '</ul>'
    else:
        html += '<p class="success">No significant biases detected.</p>'
    
    html += "<h3>Agreement Analysis</h3>"
    problematic = calibration.get('agreement_analysis', {}).get('problematic_metrics', [])
    
    if problematic:
        html += '<p class="warning">Metrics with low inter-judge agreement:</p><ul>'
        for metric in problematic:
            html += f'<li>{metric}</li>'
        html += '</ul>'
    else:
        html += '<p class="success">Good inter-judge agreement across all metrics.</p>'
    
    return html

def generate_recommendations_section(data: dict) -> str:
    """Generate the recommendations section."""
    recommendations = []
    
    if data.get('performance_summary', {}).get('recommendations'):
        recommendations.extend(data['performance_summary']['recommendations'])
    
    if data.get('calibration_report', {}).get('recommendations'):
        recommendations.extend(data['calibration_report']['recommendations'])
    
    if not recommendations:
        return '<p class="success">No specific recommendations - evaluation completed successfully!</p>'
    
    html = '<ul>'
    for rec in recommendations:
        html += f'<li>{rec}</li>'
    html += '</ul>'
    
    return html

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation reports and visualizations")
    parser.add_argument("--data-dir", default="data", help="Directory containing evaluation data")
    parser.add_argument("--output-dir", default="reports", help="Output directory for reports")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        print("Loading evaluation data...")
        data = load_evaluation_data(args.data_dir)
        
        print("Creating performance charts...")
        create_performance_charts(data, args.output_dir)
        
        print("Creating calibration charts...")
        create_calibration_charts(data, args.output_dir)
        
        print("Generating HTML report...")
        report_path = generate_html_report(data, args.output_dir)
        
        print(f"\nReport generation completed!")
        print(f"HTML Report: {report_path}")
        print(f"Charts saved to: {args.output_dir}/")
        print(f"\nOpen {report_path} in your browser to view the complete report.")
        
    except Exception as e:
        print(f"Error generating report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 