"""
Score aggregation and final performance computation for the three-judge evaluation system.
"""

import numpy as np
import logging
from typing import Dict, Any, List
from scipy import stats

logger = logging.getLogger(__name__)

class Aggregator:
    """Handles aggregation of calibrated judge scores into final performance metrics."""
    
    def __init__(self):
        pass
    
    def aggregate_scores(self, 
                        calibrated_scores: Dict[str, Dict[str, Dict[str, float]]], 
                        metrics: List[Dict[str, Any]],
                        aggregation_method: str = "mean") -> Dict[str, Dict[str, float]]:
        """Aggregate calibrated scores from all judges."""
        logger.info(f"Aggregating scores using {aggregation_method} method...")
        
        aggregated_scores = {}
        judge_names = list(calibrated_scores.keys())
        
        # Get all task IDs
        all_task_ids = set()
        for judge_scores in calibrated_scores.values():
            all_task_ids.update(judge_scores.keys())
        
        for task_id in all_task_ids:
            aggregated_scores[task_id] = {}
            
            for metric in metrics:
                metric_name = metric['name']
                scale = metric['scale'].lower()
                
                # Collect scores from all judges for this task and metric
                judge_scores = []
                for judge_name in judge_names:
                    if (task_id in calibrated_scores[judge_name] and 
                        metric_name in calibrated_scores[judge_name][task_id]):
                        score = calibrated_scores[judge_name][task_id][metric_name]
                        judge_scores.append(score)
                
                if not judge_scores:
                    logger.warning(f"No scores available for task {task_id}, metric {metric_name}")
                    aggregated_scores[task_id][metric_name] = 0.0
                    continue
                
                # Aggregate based on method and scale
                if aggregation_method == "mean":
                    aggregated_score = np.mean(judge_scores)
                elif aggregation_method == "median":
                    aggregated_score = np.median(judge_scores)
                elif aggregation_method == "majority_vote" and scale == "binary":
                    # For binary metrics, use majority vote
                    binary_scores = [1 if score > 0.5 else 0 for score in judge_scores]
                    aggregated_score = float(stats.mode(binary_scores, keepdims=True)[0][0])
                elif aggregation_method == "weighted_mean":
                    # Could implement judge-specific weights here
                    aggregated_score = np.mean(judge_scores)  # Default to equal weights
                else:
                    # Default to mean
                    aggregated_score = np.mean(judge_scores)
                
                aggregated_scores[task_id][metric_name] = float(aggregated_score)
        
        logger.info(f"Aggregated scores for {len(aggregated_scores)} tasks")
        return aggregated_scores
    
    def compute_overall_performance(self, 
                                  aggregated_scores: Dict[str, Dict[str, float]], 
                                  metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute overall performance vector across all tasks."""
        logger.info("Computing overall performance metrics...")
        
        overall_performance = {}
        
        for metric in metrics:
            metric_name = metric['name']
            
            # Collect all scores for this metric across tasks
            metric_scores = []
            for task_id, task_scores in aggregated_scores.items():
                if metric_name in task_scores:
                    metric_scores.append(task_scores[metric_name])
            
            if metric_scores:
                overall_performance[metric_name] = float(np.mean(metric_scores))
            else:
                logger.warning(f"No scores available for metric {metric_name}")
                overall_performance[metric_name] = 0.0
        
        logger.info("Overall performance computed:")
        for metric_name, score in overall_performance.items():
            logger.info(f"  {metric_name}: {score:.3f}")
        
        return overall_performance
    
    def compute_tier_performance(self, 
                               aggregated_scores: Dict[str, Dict[str, float]], 
                               tasks: List[Dict[str, Any]], 
                               metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Compute performance broken down by task tier."""
        logger.info("Computing tier-specific performance...")
        
        # Group tasks by tier
        tier_tasks = {}
        for task in tasks:
            tier = task.get('tier', 'unknown')
            if tier not in tier_tasks:
                tier_tasks[tier] = []
            tier_tasks[tier].append(task['id'])
        
        tier_performance = {}
        
        for tier, task_ids in tier_tasks.items():
            tier_performance[tier] = {}
            
            for metric in metrics:
                metric_name = metric['name']
                
                # Collect scores for this tier and metric
                tier_scores = []
                for task_id in task_ids:
                    if (task_id in aggregated_scores and 
                        metric_name in aggregated_scores[task_id]):
                        tier_scores.append(aggregated_scores[task_id][metric_name])
                
                if tier_scores:
                    tier_performance[tier][metric_name] = float(np.mean(tier_scores))
                else:
                    tier_performance[tier][metric_name] = 0.0
        
        logger.info("Tier performance computed:")
        for tier, performance in tier_performance.items():
            logger.info(f"  {tier}:")
            for metric_name, score in performance.items():
                logger.info(f"    {metric_name}: {score:.3f}")
        
        return tier_performance
    
    def identify_outliers(self, 
                         aggregated_scores: Dict[str, Dict[str, float]], 
                         metrics: List[Dict[str, Any]], 
                         threshold: float = 0.4) -> Dict[str, List[str]]:
        """Identify tasks with unusually low scores (potential outliers)."""
        logger.info(f"Identifying outlier tasks with scores < {threshold}...")
        
        outliers = {}
        
        for metric in metrics:
            metric_name = metric['name']
            outliers[metric_name] = []
            
            for task_id, task_scores in aggregated_scores.items():
                if metric_name in task_scores:
                    score = task_scores[metric_name]
                    if score < threshold:
                        outliers[metric_name].append(task_id)
        
        # Log outliers
        for metric_name, outlier_tasks in outliers.items():
            if outlier_tasks:
                logger.warning(f"Outlier tasks for {metric_name}: {outlier_tasks}")
        
        return outliers
    
    def compute_score_statistics(self, 
                               aggregated_scores: Dict[str, Dict[str, float]], 
                               metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Compute detailed statistics for each metric."""
        logger.info("Computing score statistics...")
        
        statistics = {}
        
        for metric in metrics:
            metric_name = metric['name']
            
            # Collect all scores for this metric
            metric_scores = []
            for task_scores in aggregated_scores.values():
                if metric_name in task_scores:
                    metric_scores.append(task_scores[metric_name])
            
            if metric_scores:
                statistics[metric_name] = {
                    'mean': float(np.mean(metric_scores)),
                    'median': float(np.median(metric_scores)),
                    'std': float(np.std(metric_scores)),
                    'min': float(np.min(metric_scores)),
                    'max': float(np.max(metric_scores)),
                    'q25': float(np.percentile(metric_scores, 25)),
                    'q75': float(np.percentile(metric_scores, 75)),
                    'count': len(metric_scores)
                }
            else:
                statistics[metric_name] = {
                    'mean': 0.0, 'median': 0.0, 'std': 0.0,
                    'min': 0.0, 'max': 0.0, 'q25': 0.0, 'q75': 0.0,
                    'count': 0
                }
        
        return statistics
    
    def generate_performance_summary(self, 
                                   overall_performance: Dict[str, float],
                                   tier_performance: Dict[str, Dict[str, float]],
                                   statistics: Dict[str, Dict[str, float]],
                                   outliers: Dict[str, List[str]],
                                   agent_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive performance summary."""
        summary = {
            'agent_info': agent_info,
            'overall_performance': overall_performance,
            'tier_performance': tier_performance,
            'statistics': statistics,
            'outliers': outliers,
            'summary_metrics': {},
            'recommendations': []
        }
        
        # Compute summary metrics
        all_scores = list(overall_performance.values())
        if all_scores:
            summary['summary_metrics'] = {
                'overall_mean': float(np.mean(all_scores)),
                'overall_median': float(np.median(all_scores)),
                'overall_std': float(np.std(all_scores)),
                'min_metric_score': float(np.min(all_scores)),
                'max_metric_score': float(np.max(all_scores)),
                'metrics_above_70': sum(1 for score in all_scores if score >= 0.7),
                'metrics_below_50': sum(1 for score in all_scores if score < 0.5)
            }
        
        # Generate recommendations
        if summary['summary_metrics'].get('overall_mean', 0) < 0.6:
            summary['recommendations'].append("Overall performance is below 60% - consider model improvements")
        
        if summary['summary_metrics'].get('metrics_below_50', 0) > 0:
            summary['recommendations'].append(f"{summary['summary_metrics']['metrics_below_50']} metrics scored below 50%")
        
        # Check for tier-specific issues
        for tier, tier_scores in tier_performance.items():
            tier_mean = np.mean(list(tier_scores.values())) if tier_scores else 0
            if tier_mean < 0.5:
                summary['recommendations'].append(f"Poor performance on {tier} tasks (mean: {tier_mean:.2f})")
        
        # Check for outliers
        total_outliers = sum(len(tasks) for tasks in outliers.values())
        if total_outliers > 0:
            summary['recommendations'].append(f"{total_outliers} task-metric combinations scored below 40%")
        
        return summary 