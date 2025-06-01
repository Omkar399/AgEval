"""
Calibration and reliability analysis for the three-judge evaluation system.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score
from .utils import normalize_score

logger = logging.getLogger(__name__)

class Calibrator:
    """Handles bias calibration and inter-judge agreement analysis."""
    
    def __init__(self, calibration_threshold: float = 0.3, agreement_threshold: float = 0.6):
        self.calibration_threshold = calibration_threshold
        self.agreement_threshold = agreement_threshold
    
    def calibrate_judges(self, 
                        anchor_scores: Dict[str, Dict[str, Dict[str, float]]], 
                        anchor_gold: Dict[str, Dict[str, float]],
                        metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Compute bias offsets for each judge on each metric using anchor set."""
        logger.info("Computing bias calibration offsets...")
        
        bias_offsets = {}
        
        for judge_name in anchor_scores.keys():
            bias_offsets[judge_name] = {}
            
            for metric in metrics:
                metric_name = metric['name']
                
                # Collect judge scores and gold scores for this metric
                judge_scores = []
                gold_scores = []
                
                for task_id in anchor_scores[judge_name].keys():
                    if task_id in anchor_gold and metric_name in anchor_gold[task_id]:
                        judge_score = anchor_scores[judge_name][task_id].get(metric_name, 0.0)
                        gold_score = anchor_gold[task_id][metric_name]
                        
                        judge_scores.append(judge_score)
                        gold_scores.append(gold_score)
                
                if judge_scores and gold_scores:
                    # Compute bias as mean difference
                    bias = np.mean(judge_scores) - np.mean(gold_scores)
                    bias_offsets[judge_name][metric_name] = bias
                    
                    logger.info(f"Judge {judge_name}, Metric {metric_name}: bias = {bias:.3f}")
                    
                    if abs(bias) > self.calibration_threshold:
                        logger.warning(f"Large bias detected for {judge_name} on {metric_name}: {bias:.3f}")
                else:
                    logger.warning(f"No anchor data for {judge_name} on {metric_name}")
                    bias_offsets[judge_name][metric_name] = 0.0
        
        return bias_offsets
    
    def apply_calibration(self, 
                         scores: Dict[str, Dict[str, Dict[str, float]]], 
                         bias_offsets: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Apply bias correction to judge scores."""
        logger.info("Applying bias calibration...")
        
        calibrated_scores = {}
        
        for judge_name, judge_scores in scores.items():
            calibrated_scores[judge_name] = {}
            
            for task_id, task_scores in judge_scores.items():
                calibrated_scores[judge_name][task_id] = {}
                
                for metric_name, score in task_scores.items():
                    # Apply bias correction and clip to [0, 1]
                    bias = bias_offsets.get(judge_name, {}).get(metric_name, 0.0)
                    corrected_score = max(0.0, min(1.0, score - bias))
                    calibrated_scores[judge_name][task_id][metric_name] = corrected_score
        
        return calibrated_scores
    
    def compute_inter_judge_agreement(self, 
                                    calibrated_scores: Dict[str, Dict[str, Dict[str, float]]], 
                                    metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Compute inter-judge agreement for each metric."""
        logger.info("Computing inter-judge agreement...")
        
        agreement_results = {}
        judge_names = list(calibrated_scores.keys())
        
        for metric in metrics:
            metric_name = metric['name']
            scale = metric['scale'].lower()
            
            agreement_results[metric_name] = {}
            
            # Collect scores for this metric across all tasks and judges
            task_ids = set()
            for judge_scores in calibrated_scores.values():
                task_ids.update(judge_scores.keys())
            
            # Build score matrix: tasks x judges
            score_matrix = []
            valid_tasks = []
            
            for task_id in task_ids:
                task_scores = []
                valid = True
                
                for judge_name in judge_names:
                    if (task_id in calibrated_scores[judge_name] and 
                        metric_name in calibrated_scores[judge_name][task_id]):
                        score = calibrated_scores[judge_name][task_id][metric_name]
                        task_scores.append(score)
                    else:
                        valid = False
                        break
                
                if valid and len(task_scores) == len(judge_names):
                    score_matrix.append(task_scores)
                    valid_tasks.append(task_id)
            
            if len(score_matrix) < 2:
                logger.warning(f"Insufficient data for agreement analysis on {metric_name}")
                agreement_results[metric_name] = {
                    'cohens_kappa': 0.0,
                    'pearson_correlation': 0.0,
                    'spearman_correlation': 0.0,
                    'mean_pairwise_agreement': 0.0,
                    'valid_tasks': 0
                }
                continue
            
            score_matrix = np.array(score_matrix)
            agreement_results[metric_name]['valid_tasks'] = len(valid_tasks)
            
            # Compute different agreement metrics
            if scale == 'binary' or "confidence" not in scale.lower():
                # For legacy binary metrics, use Cohen's kappa
                kappa_scores = []
                for i in range(len(judge_names)):
                    for j in range(i + 1, len(judge_names)):
                        # Convert to binary for kappa
                        scores_i = (score_matrix[:, i] > 0.5).astype(int)
                        scores_j = (score_matrix[:, j] > 0.5).astype(int)
                        
                        if len(np.unique(scores_i)) > 1 and len(np.unique(scores_j)) > 1:
                            kappa = cohen_kappa_score(scores_i, scores_j)
                            kappa_scores.append(kappa)
                
                agreement_results[metric_name]['cohens_kappa'] = np.mean(kappa_scores) if kappa_scores else 0.0
            else:
                # For confidence-based metrics, compute Intraclass Correlation Coefficient (ICC) instead
                try:
                    # Simple ICC approximation using correlation
                    all_pairs_corr = []
                    for i in range(len(judge_names)):
                        for j in range(i + 1, len(judge_names)):
                            scores_i = score_matrix[:, i]
                            scores_j = score_matrix[:, j]
                            if np.std(scores_i) > 0 and np.std(scores_j) > 0:
                                corr, _ = pearsonr(scores_i, scores_j)
                                if not np.isnan(corr):
                                    all_pairs_corr.append(corr)
                    
                    agreement_results[metric_name]['cohens_kappa'] = np.mean(all_pairs_corr) if all_pairs_corr else 0.0
                except:
                    agreement_results[metric_name]['cohens_kappa'] = 0.0
            
            # Compute correlations for all metrics
            pearson_correlations = []
            spearman_correlations = []
            
            for i in range(len(judge_names)):
                for j in range(i + 1, len(judge_names)):
                    scores_i = score_matrix[:, i]
                    scores_j = score_matrix[:, j]
                    
                    # Pearson correlation
                    if np.std(scores_i) > 0 and np.std(scores_j) > 0:
                        pearson_corr, _ = pearsonr(scores_i, scores_j)
                        pearson_correlations.append(pearson_corr)
                    
                    # Spearman correlation
                    spearman_corr, _ = spearmanr(scores_i, scores_j)
                    spearman_correlations.append(spearman_corr)
            
            agreement_results[metric_name]['pearson_correlation'] = np.mean(pearson_correlations) if pearson_correlations else 0.0
            agreement_results[metric_name]['spearman_correlation'] = np.mean(spearman_correlations) if spearman_correlations else 0.0
            
            # Compute mean pairwise agreement (proportion of exact matches for binary, MAE for numeric)
            pairwise_agreements = []
            for i in range(len(judge_names)):
                for j in range(i + 1, len(judge_names)):
                    scores_i = score_matrix[:, i]
                    scores_j = score_matrix[:, j]
                    
                    if scale == 'binary' and "confidence" not in scale.lower():
                        # Proportion of exact matches for binary
                        agreement = np.mean((scores_i > 0.5) == (scores_j > 0.5))
                    else:
                        # For confidence scores, use multiple agreement measures
                        # 1. Inverted Mean Absolute Error (higher is better)
                        mae = np.mean(np.abs(scores_i - scores_j))
                        mae_agreement = max(0.0, 1.0 - mae)
                        
                        # 2. Agreement within tolerance bands
                        tolerance_01 = np.mean(np.abs(scores_i - scores_j) <= 0.1)  # Within 10%
                        tolerance_02 = np.mean(np.abs(scores_i - scores_j) <= 0.2)  # Within 20%
                        
                        # 3. Directional agreement (both high, both medium, both low)
                        high_i = scores_i >= 0.7
                        high_j = scores_j >= 0.7
                        med_i = (scores_i >= 0.4) & (scores_i < 0.7)
                        med_j = (scores_j >= 0.4) & (scores_j < 0.7)
                        low_i = scores_i < 0.4
                        low_j = scores_j < 0.4
                        
                        directional_agreement = np.mean(
                            (high_i & high_j) | (med_i & med_j) | (low_i & low_j)
                        )
                        
                        # Combine measures (weighted average)
                        agreement = (0.4 * mae_agreement + 
                                   0.3 * tolerance_02 + 
                                   0.3 * directional_agreement)
                    
                    pairwise_agreements.append(agreement)
            
            agreement_results[metric_name]['mean_pairwise_agreement'] = np.mean(pairwise_agreements) if pairwise_agreements else 0.0
            
            # Log results
            logger.info(f"Metric {metric_name} agreement:")
            logger.info(f"  Cohen's κ: {agreement_results[metric_name]['cohens_kappa']:.3f}")
            logger.info(f"  Pearson r: {agreement_results[metric_name]['pearson_correlation']:.3f}")
            logger.info(f"  Spearman ρ: {agreement_results[metric_name]['spearman_correlation']:.3f}")
            logger.info(f"  Pairwise agreement: {agreement_results[metric_name]['mean_pairwise_agreement']:.3f}")
        
        return agreement_results
    
    def identify_problematic_metrics(self, agreement_results: Dict[str, Dict[str, float]]) -> List[str]:
        """Identify metrics with low inter-judge agreement."""
        problematic = []
        
        for metric_name, results in agreement_results.items():
            # Check multiple agreement measures
            kappa = results.get('cohens_kappa', 0.0)
            pearson = results.get('pearson_correlation', 0.0)
            spearman = results.get('spearman_correlation', 0.0)
            pairwise = results.get('mean_pairwise_agreement', 0.0)
            
            # Consider metric problematic if multiple measures are below threshold
            low_agreement_count = 0
            if kappa < self.agreement_threshold:
                low_agreement_count += 1
            if pearson < self.agreement_threshold:
                low_agreement_count += 1
            if spearman < self.agreement_threshold:
                low_agreement_count += 1
            if pairwise < self.agreement_threshold:
                low_agreement_count += 1
            
            if low_agreement_count >= 2:  # At least 2 measures below threshold
                problematic.append(metric_name)
                logger.warning(f"Low agreement detected for metric: {metric_name}")
        
        return problematic
    
    def generate_calibration_report(self, 
                                  bias_offsets: Dict[str, Dict[str, float]], 
                                  agreement_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate a comprehensive calibration report."""
        report = {
            'bias_analysis': {
                'judge_biases': bias_offsets,
                'large_biases': [],
                'bias_summary': {}
            },
            'agreement_analysis': {
                'metric_agreements': agreement_results,
                'problematic_metrics': self.identify_problematic_metrics(agreement_results),
                'overall_agreement': {}
            },
            'recommendations': []
        }
        
        # Analyze biases
        for judge_name, judge_biases in bias_offsets.items():
            for metric_name, bias in judge_biases.items():
                if abs(bias) > self.calibration_threshold:
                    report['bias_analysis']['large_biases'].append({
                        'judge': judge_name,
                        'metric': metric_name,
                        'bias': bias
                    })
        
        # Compute overall agreement statistics
        all_kappas = [r.get('cohens_kappa', 0.0) for r in agreement_results.values()]
        all_pearsons = [r.get('pearson_correlation', 0.0) for r in agreement_results.values()]
        all_spearmans = [r.get('spearman_correlation', 0.0) for r in agreement_results.values()]
        
        report['agreement_analysis']['overall_agreement'] = {
            'mean_cohens_kappa': np.mean(all_kappas),
            'mean_pearson_correlation': np.mean(all_pearsons),
            'mean_spearman_correlation': np.mean(all_spearmans)
        }
        
        # Generate recommendations
        if report['bias_analysis']['large_biases']:
            report['recommendations'].append("Consider replacing or fine-tuning judges with large biases")
        
        if report['agreement_analysis']['problematic_metrics']:
            report['recommendations'].append("Refine definitions for metrics with low inter-judge agreement")
        
        if report['agreement_analysis']['overall_agreement']['mean_pearson_correlation'] < 0.5:
            report['recommendations'].append("Overall judge agreement is low - consider using more similar judge models")
        
        return report 