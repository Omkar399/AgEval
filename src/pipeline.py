"""
Main evaluation pipeline for the three-judge evaluation system.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from .utils import load_config, load_json, save_json, ResponseCache, setup_logging
from .judge import JudgeManager
from .agent import Agent
from .metrics import MetricProposer, MetricConsolidator
from .calibration import Calibrator
from .aggregation import Aggregator

logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """Main pipeline that orchestrates the entire three-judge evaluation process."""
    
    def __init__(self, config_path: str = "config/judges_config.yaml"):
        """Initialize the evaluation pipeline."""
        self.config = load_config(config_path)
        self.cache = ResponseCache() if self.config.get('optimization', {}).get('cache_responses', True) else None
        
        # Initialize components
        self.judge_manager = JudgeManager(self.config['judges'], self.cache)
        self.agent = Agent(self.config['agent'], self.cache)
        self.metric_proposer = MetricProposer()
        self.metric_consolidator = MetricConsolidator()
        self.calibrator = Calibrator(
            calibration_threshold=self.config.get('evaluation', {}).get('calibration_threshold', 0.3),
            agreement_threshold=self.config.get('evaluation', {}).get('agreement_threshold', 0.6)
        )
        self.aggregator = Aggregator()
        
        # State tracking
        self.tasks = None
        self.anchors = None
        self.canonical_metrics = None
        self.agent_outputs = None
        self.anchor_outputs = None
        self.raw_scores = None
        self.anchor_scores = None
        self.bias_offsets = None
        self.calibrated_scores = None
        self.aggregated_scores = None
        self.final_performance = None
        
        logger.info("Evaluation pipeline initialized")
    
    def load_tasks(self, tasks_path: str = "data/tasks.json") -> List[Dict[str, Any]]:
        """Load the task suite."""
        logger.info(f"Loading tasks from {tasks_path}")
        self.tasks = load_json(tasks_path)
        logger.info(f"Loaded {len(self.tasks)} tasks")
        return self.tasks
    
    def load_anchors(self, anchors_path: str = "data/anchors.json") -> List[Dict[str, Any]]:
        """Load the anchor set for calibration."""
        logger.info(f"Loading anchors from {anchors_path}")
        self.anchors = load_json(anchors_path)
        logger.info(f"Loaded {len(self.anchors)} anchor tasks")
        return self.anchors
    
    def phase_1_task_suite(self, tasks_path: str = "data/tasks.json") -> List[Dict[str, Any]]:
        """Phase 1: Load and validate task suite."""
        logger.info("=== Phase 1: Task Suite Definition ===")
        return self.load_tasks(tasks_path)
    
    def phase_2_configure_judges(self) -> None:
        """Phase 2: Judge configuration (already done in __init__)."""
        logger.info("=== Phase 2: Judge Configuration ===")
        judge_names = self.judge_manager.get_judge_names()
        logger.info(f"Configured judges: {judge_names}")
    
    def phase_3_metric_proposal(self) -> Dict[str, List[Dict[str, Any]]]:
        """Phase 3: Collect metric proposals from judges."""
        logger.info("=== Phase 3: Metric Proposal ===")
        if not self.tasks:
            raise ValueError("Tasks must be loaded before metric proposal")
        
        proposals = self.metric_proposer.collect_proposals(self.judge_manager, self.tasks)
        save_json(proposals, "data/metric_proposals.json")
        return proposals
    
    def phase_4_metric_consolidation(self, proposals: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Phase 4: Consolidate metrics into canonical set."""
        logger.info("=== Phase 4: Metric Consolidation ===")
        self.canonical_metrics = self.metric_consolidator.consolidate_metrics(proposals)
        save_json(self.canonical_metrics, "data/canonical_metrics.json")
        return self.canonical_metrics
    
    def phase_5_generate_outputs(self) -> Dict[str, Dict[str, Any]]:
        """Phase 5: Generate agent outputs for all tasks."""
        logger.info("=== Phase 5: Agent Output Generation ===")
        if not self.tasks:
            raise ValueError("Tasks must be loaded before output generation")
        
        self.agent_outputs = self.agent.generate_all_outputs(self.tasks)
        save_json(self.agent_outputs, "data/agent_outputs.json")
        
        # Also generate outputs for anchor tasks
        if self.anchors:
            self.anchor_outputs = self.agent.generate_all_outputs(self.anchors)
            save_json(self.anchor_outputs, "data/anchor_outputs.json")
        
        return self.agent_outputs
    
    def phase_6_scoring(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Phase 6: Score outputs with all judges."""
        logger.info("=== Phase 6: Scoring Phase ===")
        if not self.canonical_metrics or not self.agent_outputs:
            raise ValueError("Metrics and agent outputs must be available before scoring")
        
        # Score main tasks
        self.raw_scores = self.judge_manager.score_outputs(self.agent_outputs, self.canonical_metrics)
        save_json(self.raw_scores, "data/raw_scores.json")
        
        # Score anchor tasks if available
        if self.anchor_outputs:
            self.anchor_scores = self.judge_manager.score_outputs(self.anchor_outputs, self.canonical_metrics)
            save_json(self.anchor_scores, "data/anchor_scores.json")
        
        return self.raw_scores
    
    def phase_7_calibration(self) -> Dict[str, Dict[str, float]]:
        """Phase 7: Calibration and reliability analysis."""
        logger.info("=== Phase 7: Calibration & Reliability ===")
        if not self.raw_scores or not self.canonical_metrics:
            raise ValueError("Raw scores and metrics must be available before calibration")
        
        # Compute bias offsets using anchor set
        if self.anchor_scores and self.anchors:
            # Convert anchors to gold standard format
            anchor_gold = {}
            for anchor in self.anchors:
                if 'gold_metrics' in anchor:
                    anchor_gold[anchor['id']] = anchor['gold_metrics']
            
            self.bias_offsets = self.calibrator.calibrate_judges(
                self.anchor_scores, anchor_gold, self.canonical_metrics
            )
        else:
            logger.warning("No anchor data available - skipping bias calibration")
            self.bias_offsets = {judge: {metric['name']: 0.0 for metric in self.canonical_metrics} 
                               for judge in self.judge_manager.get_judge_names()}
        
        save_json(self.bias_offsets, "data/bias_offsets.json")
        
        # Apply calibration
        self.calibrated_scores = self.calibrator.apply_calibration(self.raw_scores, self.bias_offsets)
        save_json(self.calibrated_scores, "data/calibrated_scores.json")
        
        # Compute inter-judge agreement
        agreement_results = self.calibrator.compute_inter_judge_agreement(
            self.calibrated_scores, self.canonical_metrics
        )
        
        # Generate calibration report
        calibration_report = self.calibrator.generate_calibration_report(
            self.bias_offsets, agreement_results
        )
        save_json(calibration_report, "data/calibration_report.json")
        
        return self.bias_offsets
    
    def phase_8_aggregation(self) -> Dict[str, float]:
        """Phase 8: Aggregate scores and compute final performance."""
        logger.info("=== Phase 8: Aggregation & Reporting ===")
        if not self.calibrated_scores or not self.canonical_metrics:
            raise ValueError("Calibrated scores and metrics must be available before aggregation")
        
        # Aggregate scores
        self.aggregated_scores = self.aggregator.aggregate_scores(
            self.calibrated_scores, self.canonical_metrics
        )
        save_json(self.aggregated_scores, "data/aggregated_scores.json")
        
        # Compute overall performance
        self.final_performance = self.aggregator.compute_overall_performance(
            self.aggregated_scores, self.canonical_metrics
        )
        save_json(self.final_performance, "data/final_performance.json")
        
        # Compute tier-specific performance
        tier_performance = self.aggregator.compute_tier_performance(
            self.aggregated_scores, self.tasks, self.canonical_metrics
        )
        
        # Identify outliers
        outliers = self.aggregator.identify_outliers(
            self.aggregated_scores, self.canonical_metrics
        )
        
        # Compute statistics
        statistics = self.aggregator.compute_score_statistics(
            self.aggregated_scores, self.canonical_metrics
        )
        
        # Generate comprehensive summary
        performance_summary = self.aggregator.generate_performance_summary(
            self.final_performance, tier_performance, statistics, outliers, self.agent.get_info()
        )
        save_json(performance_summary, "data/performance_summary.json")
        
        return self.final_performance
    
    def run_full_evaluation(self, 
                          tasks_path: str = "data/tasks.json",
                          anchors_path: str = "data/anchors.json") -> Dict[str, Any]:
        """Run the complete evaluation pipeline."""
        logger.info("Starting full three-judge evaluation pipeline...")
        start_time = datetime.now()
        
        try:
            # Phase 1: Task Suite
            self.phase_1_task_suite(tasks_path)
            self.load_anchors(anchors_path)
            
            # Phase 2: Judge Configuration
            self.phase_2_configure_judges()
            
            # Phase 3: Metric Proposal
            proposals = self.phase_3_metric_proposal()
            
            # Phase 4: Metric Consolidation
            self.phase_4_metric_consolidation(proposals)
            
            # Phase 5: Generate Outputs
            self.phase_5_generate_outputs()
            
            # Phase 6: Scoring
            self.phase_6_scoring()
            
            # Phase 7: Calibration
            self.phase_7_calibration()
            
            # Phase 8: Aggregation
            self.phase_8_aggregation()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Create final evaluation report
            evaluation_report = {
                'evaluation_id': f"eval_{start_time.strftime('%Y%m%d_%H%M%S')}",
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'agent_info': self.agent.get_info(),
                'judge_names': self.judge_manager.get_judge_names(),
                'num_tasks': len(self.tasks),
                'num_anchors': len(self.anchors) if self.anchors else 0,
                'num_metrics': len(self.canonical_metrics),
                'final_performance': self.final_performance,
                'status': 'completed'
            }
            
            save_json(evaluation_report, "data/evaluation_report.json")
            
            logger.info(f"Evaluation completed successfully in {duration:.1f} seconds")
            logger.info("Final Performance:")
            for metric_name, score in self.final_performance.items():
                logger.info(f"  {metric_name}: {score:.3f}")
            
            return evaluation_report
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            error_report = {
                'evaluation_id': f"eval_{start_time.strftime('%Y%m%d_%H%M%S')}",
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            }
            save_json(error_report, "data/evaluation_report.json")
            raise
    
    def load_previous_results(self, results_dir: str = "data") -> None:
        """Load results from a previous evaluation run."""
        logger.info("Loading previous evaluation results...")
        
        try:
            self.tasks = load_json(f"{results_dir}/tasks.json")
            self.canonical_metrics = load_json(f"{results_dir}/canonical_metrics.json")
            self.final_performance = load_json(f"{results_dir}/final_performance.json")
            logger.info("Previous results loaded successfully")
        except FileNotFoundError as e:
            logger.warning(f"Could not load previous results: {e}")
    
    def get_results(self) -> Dict[str, Any]:
        """Get current evaluation results."""
        return {
            'tasks': self.tasks,
            'canonical_metrics': self.canonical_metrics,
            'final_performance': self.final_performance,
            'agent_info': self.agent.get_info() if self.agent else None
        } 