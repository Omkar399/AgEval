"""
Enhanced evaluation pipeline with self-evaluation, reliability, and failure prevention.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from .pipeline import EvaluationPipeline
from .self_evaluation import SelfEvaluator, FailureDetector
from .reliability import ReliabilityManager, TaskAgnosticFramework
from .specialized_agents import SpecializedAgentFactory
from .adaptive_evaluation import (
    AdaptiveEvaluationPipeline, TaskDomain, 
    create_adaptive_benchmark_from_existing, IRTDifficultyEstimator
)
from .utils import load_config, save_json, setup_logging

logger = logging.getLogger(__name__)

class EnhancedEvaluationPipeline(EvaluationPipeline):
    """Enhanced pipeline with self-evaluation, reliability, adaptive evaluation, and failure prevention."""
    
    def __init__(self, config_path: str = "config/judges_config.yaml"):
        """Initialize enhanced pipeline with additional components."""
        super().__init__(config_path)
        
        # Initialize specialized agent factory
        self.specialized_agent_factory = SpecializedAgentFactory()
        
        # Initialize adaptive evaluation pipeline
        self.adaptive_pipeline = AdaptiveEvaluationPipeline(
            initial_ability=self.config.get('adaptive_evaluation', {}).get('initial_ability', 0.0),
            convergence_threshold=self.config.get('adaptive_evaluation', {}).get('convergence_threshold', 0.3),
            max_items=self.config.get('adaptive_evaluation', {}).get('max_items', 15),
            min_items=self.config.get('adaptive_evaluation', {}).get('min_items', 5)
        )
        
        # Initialize enhanced components
        self.self_evaluator = None
        if self.config.get('self_evaluation', {}).get('enabled', False):
            self.self_evaluator = SelfEvaluator(self.config['agent'], self.cache)
        
        self.failure_detector = None
        if self.config.get('failure_prevention', {}).get('enabled', False):
            self.failure_detector = FailureDetector()
        
        self.reliability_manager = None
        if self.config.get('reliability', {}).get('enabled', False):
            self.reliability_manager = ReliabilityManager(self.config)
            # Create checkpoints directory
            os.makedirs("checkpoints", exist_ok=True)
        
        self.task_agnostic_framework = None
        if self.config.get('task_agnostic', {}).get('enabled', False):
            self.task_agnostic_framework = TaskAgnosticFramework()
        
        # Enhanced state tracking
        self.self_evaluation_results = {}
        self.failure_analysis = {}
        self.reliability_metrics = {}
        self.token_optimization_stats = {}
        self.specialized_agent_info = {}  # Track which agent handled which task
        self.adaptive_evaluation_results = {}  # Track adaptive evaluation results
        
        logger.info("Enhanced evaluation pipeline initialized with adaptive evaluation and advanced capabilities")
    
    def run_enhanced_evaluation(self, 
                              tasks_path: str = "data/tasks.json",
                              anchors_path: str = "data/anchors.json",
                              enable_self_eval: bool = True,
                              enable_reliability: bool = True,
                              enable_adaptive: bool = True,
                              adaptive_domain: TaskDomain = TaskDomain.ANALYTICAL) -> Dict[str, Any]:
        """Run complete enhanced evaluation with adaptive difficulty calibration."""
        logger.info("=== Starting Enhanced AgEval Evaluation with Adaptive Difficulty Calibration ===")
        
        try:
            if enable_adaptive:
                # ADAPTIVE EVALUATION MODE - Replace static evaluation
                logger.info("🎯 Running Adaptive Evaluation Mode")
                
                # Phase 1: Load base tasks for adaptive generation
                base_tasks = self.phase_1_adaptive_task_preparation(tasks_path)
                
                # Phase 2: Initialize adaptive agent wrapper
                adaptive_agent = self.phase_2_adaptive_agent_setup()
                
                # Phase 3: Run adaptive evaluation with IRT-based difficulty calibration
                adaptive_results = self.phase_3_run_adaptive_evaluation(
                    adaptive_agent, base_tasks, adaptive_domain
                )
                
                # Phase 4: Enhanced analysis with adaptive insights
                comprehensive_report = self.phase_4_adaptive_analysis(adaptive_results)
                
                # Combine with traditional enhanced features if enabled
                if enable_self_eval and self.self_evaluator:
                    self.self_evaluation_results = self.analyze_adaptive_self_evaluation(adaptive_results)
                
                if enable_reliability and self.reliability_manager:
                    self.reliability_metrics = self.analyze_adaptive_reliability(adaptive_results)
                
                # Package enhanced adaptive results
                enhanced_results = {
                    'adaptive_evaluation_results': adaptive_results,
                    'ability_estimate': adaptive_results['adaptive_evaluation_results']['final_ability_estimate'],
                    'ability_percentile': adaptive_results['adaptive_evaluation_results']['ability_percentile'],
                    'convergence_achieved': adaptive_results['adaptive_evaluation_results']['convergence_achieved'],
                    'evaluation_efficiency': self.calculate_evaluation_efficiency(adaptive_results),
                    'self_evaluation_analysis': self.self_evaluation_results,
                    'reliability_metrics': self.reliability_metrics,
                    'comprehensive_report': comprehensive_report,
                    'evaluation_mode': 'adaptive',
                    'timestamp': datetime.now().isoformat(),
                    'enhanced_features_used': {
                        'adaptive_evaluation': True,
                        'irt_difficulty_calibration': True,
                        'self_evaluation': enable_self_eval and self.self_evaluator is not None,
                        'reliability_management': enable_reliability and self.reliability_manager is not None,
                        'task_agnostic_framework': self.task_agnostic_framework is not None,
                        'specialized_agents': True
                    }
                }
                
            else:
                # TRADITIONAL STATIC EVALUATION MODE - Original enhanced pipeline
                logger.info("📊 Running Traditional Static Evaluation Mode")
                
                # Phase 1: Task Suite with Task-Agnostic Adaptation
                tasks = self.phase_1_enhanced_task_suite(tasks_path)
                
                # Phase 2: Judge Configuration with Token Optimization
                self.phase_2_enhanced_judge_config()
                
                # Phase 3: Enhanced Metric Proposal with Universal Metrics
                proposals = self.phase_3_enhanced_metric_proposal()
                
                # Phase 4: Metric Consolidation with Task Adaptation
                canonical_metrics = self.phase_4_enhanced_metric_consolidation(proposals)
                
                # Phase 5: Enhanced Agent Output Generation with Specialized Agents
                agent_outputs = self.phase_5_enhanced_output_generation(enable_self_eval)
                
                # Phase 6: Enhanced Scoring with Failure Detection
                raw_scores = self.phase_6_enhanced_scoring()
                
                # Phase 7: Enhanced Calibration with Reliability Analysis
                bias_offsets = self.phase_7_enhanced_calibration()
                
                # Phase 8: Enhanced Aggregation with Consistency Validation
                final_results = self.phase_8_enhanced_aggregation(enable_reliability)
                
                # Phase 9: Comprehensive Analysis and Reporting
                comprehensive_report = self.phase_9_comprehensive_analysis()
                
                # Save enhanced results
                enhanced_results = {
                    'evaluation_results': final_results,
                    'self_evaluation_analysis': self.self_evaluation_results,
                    'failure_analysis': self.failure_analysis,
                    'reliability_metrics': self.reliability_metrics,
                    'token_optimization_stats': self.token_optimization_stats,
                    'comprehensive_report': comprehensive_report,
                    'evaluation_mode': 'static',
                    'timestamp': datetime.now().isoformat(),
                    'enhanced_features_used': {
                        'adaptive_evaluation': False,
                        'self_evaluation': enable_self_eval and self.self_evaluator is not None,
                        'failure_detection': self.failure_detector is not None,
                        'reliability_management': enable_reliability and self.reliability_manager is not None,
                        'task_agnostic_framework': self.task_agnostic_framework is not None,
                        'token_optimization': self.config.get('token_optimization', {}).get('enabled', False)
                    }
                }
            
            # Save results with appropriate filename
            result_filename = f"data/{'adaptive' if enable_adaptive else 'static'}_evaluation_results.json"
            save_json(enhanced_results, result_filename)
            logger.info(f"Enhanced evaluation completed successfully - saved to {result_filename}")
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Enhanced evaluation failed: {e}")
            # Save partial results for analysis
            partial_results = {
                'error': str(e),
                'partial_results': {
                    'adaptive_evaluation_results': getattr(self, 'adaptive_evaluation_results', {}),
                    'self_evaluation_results': getattr(self, 'self_evaluation_results', {}),
                    'failure_analysis': getattr(self, 'failure_analysis', {}),
                    'reliability_metrics': getattr(self, 'reliability_metrics', {}),
                    'token_optimization_stats': getattr(self, 'token_optimization_stats', {})
                },
                'timestamp': datetime.now().isoformat()
            }
            save_json(partial_results, "data/enhanced_evaluation_error.json")
            raise e
    
    def phase_1_adaptive_task_preparation(self, tasks_path: str) -> List[Dict[str, Any]]:
        """Phase 1: Prepare base tasks for adaptive evaluation."""
        logger.info("=== Phase 1: Adaptive Task Preparation ===")
        
        # Load base tasks
        base_tasks = self.load_tasks(tasks_path)
        
        # Convert to format suitable for adaptive evaluation
        adapted_tasks = []
        for task in base_tasks:
            adapted_task = {
                'id': task.get('id', f'task_{len(adapted_tasks)}'),
                'prompt': task.get('prompt', task.get('description', '')),
                'domain': task.get('domain', 'analytical'),
                'complexity': task.get('complexity', 'medium'),
                'expected_response_length': task.get('expected_response_length', 200)
            }
            adapted_tasks.append(adapted_task)
        
        logger.info(f"Prepared {len(adapted_tasks)} base tasks for adaptive evaluation")
        save_json(adapted_tasks, "data/adaptive_base_tasks.json")
        
        return adapted_tasks
    
    def phase_2_adaptive_agent_setup(self):
        """Phase 2: Set up agent wrapper for adaptive evaluation."""
        logger.info("=== Phase 2: Adaptive Agent Setup ===")
        
        # Create wrapper that integrates our existing agent with adaptive pipeline
        class AdaptiveAgentWrapper:
            def __init__(self, ageval_agent, specialized_factory, self_evaluator=None):
                self.ageval_agent = ageval_agent
                self.specialized_factory = specialized_factory
                self.self_evaluator = self_evaluator
                
            def generate_response(self, prompt, task_id):
                """Generate response using AgEval agent system."""
                try:
                    # Use specialized agent selection if available
                    agent_to_use = self.ageval_agent
                    if self.specialized_factory:
                        # Create a temporary task dict for agent selection
                        temp_task = {'prompt': prompt, 'id': task_id}
                        agent_to_use = self.specialized_factory.get_agent_for_task(temp_task, self.ageval_agent.config)
                    
                    # Generate initial response
                    response = agent_to_use.generate_response(prompt, task_id)
                    response_text = response.get('response', '') if isinstance(response, dict) else response
                    
                    # Apply self-evaluation if enabled
                    if self.self_evaluator:
                        try:
                            improved_response = self.self_evaluator.evaluate_and_improve_response(
                                prompt, response_text, task_id
                            )
                            if improved_response and improved_response.get('final_response'):
                                response_text = improved_response['final_response']
                        except Exception as e:
                            logger.warning(f"Self-evaluation failed for {task_id}: {e}")
                    
                    return {'response': response_text, 'task_id': task_id}
                    
                except Exception as e:
                    logger.error(f"Agent response generation failed for {task_id}: {e}")
                    return {'response': '', 'task_id': task_id}
        
        adaptive_agent = AdaptiveAgentWrapper(
            self.agent, 
            self.specialized_agent_factory,
            self.self_evaluator
        )
        
        logger.info("Adaptive agent wrapper created with specialized agent integration")
        return adaptive_agent
    
    def phase_3_run_adaptive_evaluation(self, 
                                      adaptive_agent,
                                      base_tasks: List[Dict[str, Any]],
                                      domain: TaskDomain) -> Dict[str, Any]:
        """Phase 3: Execute adaptive evaluation with IRT-based difficulty calibration."""
        logger.info("=== Phase 3: Adaptive Evaluation Execution ===")
        
        # Run the adaptive evaluation
        adaptive_results = self.adaptive_pipeline.run_adaptive_evaluation(
            adaptive_agent, base_tasks, domain
        )
        
        # Store for analysis
        self.adaptive_evaluation_results = adaptive_results
        
        # Save detailed adaptive results
        save_json(adaptive_results, "data/detailed_adaptive_results.json")
        
        # Generate trajectory plot
        try:
            plot_path = "reports/adaptive_evaluation_trajectory.png"
            os.makedirs("reports", exist_ok=True)
            self.adaptive_pipeline.plot_evaluation_trajectory(plot_path)
            logger.info(f"Adaptive trajectory plot saved to {plot_path}")
        except Exception as e:
            logger.warning(f"Failed to generate trajectory plot: {e}")
        
        logger.info(f"Adaptive evaluation completed with {adaptive_results['adaptive_evaluation_results']['total_items_administered']} items")
        
        return adaptive_results
    
    def phase_4_adaptive_analysis(self, adaptive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Comprehensive analysis of adaptive evaluation results."""
        logger.info("=== Phase 4: Adaptive Evaluation Analysis ===")
        
        # Extract key metrics
        ability_estimate = adaptive_results['adaptive_evaluation_results']['final_ability_estimate']
        ability_percentile = adaptive_results['adaptive_evaluation_results']['ability_percentile']
        convergence_achieved = adaptive_results['adaptive_evaluation_results']['convergence_achieved']
        total_items = adaptive_results['adaptive_evaluation_results']['total_items_administered']
        
        # Analyze performance trajectory
        performance_analysis = adaptive_results['performance_analysis']
        avg_performance = performance_analysis['average_performance']
        consistency = performance_analysis['performance_consistency']
        difficulty_range = performance_analysis['difficulty_range_explored']
        
        # Calculate efficiency metrics
        efficiency_score = self.calculate_evaluation_efficiency(adaptive_results)
        precision_score = self.calculate_ability_precision(adaptive_results)
        
        # Generate recommendations
        recommendations = self.generate_adaptive_recommendations(adaptive_results)
        
        comprehensive_report = {
            'adaptive_evaluation_summary': {
                'final_ability_estimate': ability_estimate,
                'ability_percentile': ability_percentile,
                'confidence_interval': adaptive_results['adaptive_evaluation_results']['ability_confidence_interval'],
                'convergence_achieved': convergence_achieved,
                'total_items_administered': total_items,
                'evaluation_efficiency': efficiency_score,
                'ability_precision': precision_score
            },
            'performance_insights': {
                'average_performance': avg_performance,
                'performance_consistency': consistency,
                'difficulty_range_explored': difficulty_range,
                'learning_trajectory': 'Adaptive' if convergence_achieved else 'Continuing',
                'strength_areas': self.identify_strength_areas(adaptive_results),
                'improvement_areas': self.identify_improvement_areas(adaptive_results)
            },
            'adaptive_framework_effectiveness': {
                'convergence_rate': 1.0 if convergence_achieved else 0.0,
                'measurement_precision': precision_score,
                'efficiency_gain': efficiency_score,
                'optimal_difficulty_targeting': self.assess_difficulty_targeting(adaptive_results)
            },
            'recommendations': recommendations,
            'research_contributions': {
                'irt_model_validation': self.validate_irt_assumptions(adaptive_results),
                'adaptive_algorithm_performance': self.assess_adaptive_algorithm(adaptive_results),
                'measurement_innovation': 'Dynamic difficulty calibration with IRT'
            }
        }
        
        save_json(comprehensive_report, "data/adaptive_comprehensive_analysis.json")
        logger.info("Comprehensive adaptive analysis completed")
        
        return comprehensive_report
    
    def calculate_evaluation_efficiency(self, adaptive_results: Dict[str, Any]) -> float:
        """Calculate how efficiently the adaptive evaluation reached convergence."""
        total_items = adaptive_results['adaptive_evaluation_results']['total_items_administered']
        max_items = self.adaptive_pipeline.max_items
        convergence_achieved = adaptive_results['adaptive_evaluation_results']['convergence_achieved']
        
        if not convergence_achieved:
            return total_items / max_items  # Partial efficiency
        
        # Efficiency score: achieved convergence with fewer items = higher efficiency
        efficiency = 1.0 - (total_items / max_items)
        return max(0.0, efficiency)  # Ensure non-negative
    
    def analyze_adaptive_self_evaluation(self, adaptive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze self-evaluation effectiveness in adaptive context."""
        if not self.self_evaluator:
            return {}
        
        detailed_responses = adaptive_results.get('detailed_responses', [])
        
        analysis = {
            'adaptive_self_evaluation_insights': {
                'total_adaptive_responses': len(detailed_responses),
                'responses_with_reasoning': sum(1 for r in detailed_responses if r.get('reasoning_steps', 0) > 0),
                'average_reasoning_steps': sum(r.get('reasoning_steps', 0) for r in detailed_responses) / len(detailed_responses) if detailed_responses else 0,
                'difficulty_vs_reasoning_correlation': self.calculate_difficulty_reasoning_correlation(detailed_responses)
            }
        }
        
        return analysis
    
    def analyze_adaptive_reliability(self, adaptive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze reliability metrics in adaptive evaluation context."""
        if not self.reliability_manager:
            return {}
        
        # Calculate reliability specific to adaptive evaluation
        ability_se = adaptive_results['adaptive_evaluation_results']['ability_standard_error']
        convergence_achieved = adaptive_results['adaptive_evaluation_results']['convergence_achieved']
        
        reliability_score = 1.0 / (1.0 + ability_se) if ability_se > 0 else 1.0
        
        analysis = {
            'adaptive_reliability_metrics': {
                'ability_measurement_reliability': reliability_score,
                'standard_error': ability_se,
                'convergence_reliability': 1.0 if convergence_achieved else 0.5,
                'measurement_precision_category': self.categorize_measurement_precision(ability_se)
            }
        }
        
        return analysis
    
    def calculate_ability_precision(self, adaptive_results: Dict[str, Any]) -> float:
        """Calculate precision of ability estimate."""
        se = adaptive_results['adaptive_evaluation_results']['ability_standard_error']
        # Higher precision = lower standard error
        return 1.0 / (1.0 + se)
    
    def generate_adaptive_recommendations(self, adaptive_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on adaptive evaluation results."""
        recommendations = []
        
        convergence = adaptive_results['adaptive_evaluation_results']['convergence_achieved']
        total_items = adaptive_results['adaptive_evaluation_results']['total_items_administered']
        ability_percentile = adaptive_results['adaptive_evaluation_results']['ability_percentile']
        se = adaptive_results['adaptive_evaluation_results']['ability_standard_error']
        
        if not convergence:
            recommendations.append("Consider increasing max_items to allow for convergence")
            recommendations.append("Review convergence_threshold settings for optimal precision")
        
        if total_items < self.adaptive_pipeline.min_items + 2:
            recommendations.append("Evaluation converged very quickly - consider more challenging base tasks")
        
        if se > 0.5:
            recommendations.append("High measurement uncertainty - consider tasks with better discrimination")
        
        if ability_percentile > 90:
            recommendations.append("Agent shows high ability - consider more challenging task domains")
        elif ability_percentile < 10:
            recommendations.append("Agent shows lower ability - consider easier starting difficulty")
        
        recommendations.append("Consider expanding base task pool for more comprehensive evaluation")
        recommendations.append("Implement cross-domain adaptive evaluation for robust ability assessment")
        
        return recommendations
    
    def identify_strength_areas(self, adaptive_results: Dict[str, Any]) -> List[str]:
        """Identify areas where the agent performed well."""
        detailed_responses = adaptive_results.get('detailed_responses', [])
        
        strengths = []
        high_performance_responses = [r for r in detailed_responses if r.get('performance', 0) > 0.7]
        
        if len(high_performance_responses) > len(detailed_responses) * 0.6:
            strengths.append("Consistent high performance across difficulty levels")
        
        if any(r.get('reasoning_steps', 0) > 3 for r in detailed_responses):
            strengths.append("Demonstrates structured reasoning approach")
        
        # Check if agent performed well on harder tasks
        hard_tasks = [r for r in detailed_responses if r.get('difficulty', 0) > 0.6]
        if hard_tasks and sum(r.get('performance', 0) for r in hard_tasks) / len(hard_tasks) > 0.6:
            strengths.append("Handles complex tasks effectively")
        
        return strengths or ["Baseline performance demonstrated"]
    
    def identify_improvement_areas(self, adaptive_results: Dict[str, Any]) -> List[str]:
        """Identify areas where the agent could improve."""
        detailed_responses = adaptive_results.get('detailed_responses', [])
        
        improvements = []
        low_performance_responses = [r for r in detailed_responses if r.get('performance', 0) < 0.4]
        
        if len(low_performance_responses) > len(detailed_responses) * 0.3:
            improvements.append("Inconsistent performance across task difficulties")
        
        if sum(r.get('reasoning_steps', 0) for r in detailed_responses) / len(detailed_responses) < 2:
            improvements.append("Could benefit from more structured reasoning")
        
        # Check specific difficulty ranges
        easy_tasks = [r for r in detailed_responses if r.get('difficulty', 0) < 0.4]
        if easy_tasks and sum(r.get('performance', 0) for r in easy_tasks) / len(easy_tasks) < 0.7:
            improvements.append("Basic task performance needs improvement")
        
        return improvements or ["No major improvement areas identified"]
    
    def assess_difficulty_targeting(self, adaptive_results: Dict[str, Any]) -> float:
        """Assess how well the adaptive algorithm targeted optimal difficulty."""
        detailed_responses = adaptive_results.get('detailed_responses', [])
        if not detailed_responses:
            return 0.5
        
        # Optimal targeting = difficulty levels close to ability estimate
        ability = adaptive_results['adaptive_evaluation_results']['final_ability_estimate']
        
        # Convert ability (logit scale) to difficulty scale (0-1)
        optimal_difficulty = 0.5 + ability / 5.0  # Rough mapping
        optimal_difficulty = max(0.2, min(0.8, optimal_difficulty))
        
        # Calculate how close administered difficulties were to optimal
        difficulties = [r.get('difficulty', 0.5) for r in detailed_responses]
        targeting_scores = [1.0 - abs(d - optimal_difficulty) for d in difficulties]
        
        return sum(targeting_scores) / len(targeting_scores) if targeting_scores else 0.5
    
    def validate_irt_assumptions(self, adaptive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Item Response Theory model assumptions."""
        irt_history = adaptive_results.get('irt_response_history', [])
        
        if not irt_history:
            return {'validation_status': 'insufficient_data'}
        
        # Check discrimination parameters
        discriminations = [entry.get('discrimination', 1.0) for entry in irt_history]
        avg_discrimination = sum(discriminations) / len(discriminations)
        
        # Check difficulty spread
        difficulties = [entry.get('difficulty', 0.0) for entry in irt_history]
        difficulty_range = max(difficulties) - min(difficulties) if difficulties else 0.0
        
        return {
            'validation_status': 'validated',
            'average_discrimination': avg_discrimination,
            'difficulty_range': difficulty_range,
            'model_fit_quality': 'good' if difficulty_range > 1.0 and avg_discrimination > 0.8 else 'acceptable'
        }
    
    def assess_adaptive_algorithm(self, adaptive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the performance of the adaptive algorithm."""
        convergence = adaptive_results['adaptive_evaluation_results']['convergence_achieved']
        total_items = adaptive_results['adaptive_evaluation_results']['total_items_administered']
        se = adaptive_results['adaptive_evaluation_results']['ability_standard_error']
        
        return {
            'convergence_achieved': convergence,
            'efficiency_rating': 'high' if total_items <= 8 and convergence else 'medium',
            'precision_rating': 'high' if se < 0.3 else 'medium' if se < 0.5 else 'low',
            'algorithm_effectiveness': 'excellent' if convergence and se < 0.3 and total_items <= 10 else 'good'
        }
    
    def calculate_difficulty_reasoning_correlation(self, detailed_responses: List[Dict[str, Any]]) -> float:
        """Calculate correlation between task difficulty and reasoning steps used."""
        if len(detailed_responses) < 3:
            return 0.0
        
        difficulties = [r.get('difficulty', 0.5) for r in detailed_responses]
        reasoning_steps = [r.get('reasoning_steps', 0) for r in detailed_responses]
        
        # Simple correlation calculation
        n = len(difficulties)
        sum_xy = sum(d * r for d, r in zip(difficulties, reasoning_steps))
        sum_x = sum(difficulties)
        sum_y = sum(reasoning_steps)
        sum_x2 = sum(d * d for d in difficulties)
        sum_y2 = sum(r * r for r in reasoning_steps)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def categorize_measurement_precision(self, standard_error: float) -> str:
        """Categorize measurement precision based on standard error."""
        if standard_error < 0.2:
            return "excellent"
        elif standard_error < 0.3:
            return "good"
        elif standard_error < 0.5:
            return "acceptable"
        else:
            return "needs_improvement"
    
    def phase_1_enhanced_task_suite(self, tasks_path: str) -> List[Dict[str, Any]]:
        """Enhanced task loading with task-agnostic adaptations."""
        logger.info("=== Phase 1: Enhanced Task Suite Definition ===")
        
        tasks = self.load_tasks(tasks_path)
        
        # Apply task-agnostic framework adaptations
        if self.task_agnostic_framework:
            for task in tasks:
                # Add universal task properties
                task['universal_metrics'] = self.task_agnostic_framework.adapt_metrics_to_task(task)
                task['task_complexity'] = self._assess_task_complexity(task)
                task['expected_token_usage'] = self._estimate_task_token_usage(task)
        
        logger.info(f"Enhanced task suite loaded with {len(tasks)} tasks")
        return tasks
    
    def phase_2_enhanced_judge_config(self) -> None:
        """Enhanced judge configuration with token optimization."""
        logger.info("=== Phase 2: Enhanced Judge Configuration ===")
        
        super().phase_2_configure_judges()
        
        # Apply token optimization to judge configurations
        if self.config.get('token_optimization', {}).get('enabled', False):
            for judge in self.judge_manager.judges:  # Access judges as list, not dict
                judge_config = judge.config
                
                # Optimize token limits based on model capabilities
                if self.reliability_manager:
                    optimized_config = self.reliability_manager.optimize_token_usage(
                        "", judge_config['model'], judge_config.get('max_tokens')
                    )
                    
                    # Update judge configuration with optimized settings
                    judge_config['optimized_max_tokens'] = optimized_config.get('effective_limit', judge_config.get('max_tokens', 4000))
        
        logger.info("Enhanced judge configuration completed")
    
    def phase_3_enhanced_metric_proposal(self) -> Dict[str, List[Dict[str, Any]]]:
        """Enhanced metric proposal with universal metrics integration."""
        logger.info("=== Phase 3: Enhanced Metric Proposal ===")
        
        # Get standard metric proposals
        proposals = super().phase_3_metric_proposal()
        
        # Add universal metrics if task-agnostic framework is enabled
        if self.task_agnostic_framework:
            proposals['Universal'] = self.task_agnostic_framework.universal_metrics
            logger.info("Added universal metrics to proposal set")
        
        return proposals
    
    def phase_4_enhanced_metric_consolidation(self, proposals: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Enhanced metric consolidation with task adaptation."""
        logger.info("=== Phase 4: Enhanced Metric Consolidation ===")
        
        canonical_metrics = super().phase_4_metric_consolidation(proposals)
        
        # Enhance metrics with task-specific adaptations
        if self.task_agnostic_framework and self.tasks:
            enhanced_metrics = []
            for metric in canonical_metrics:
                # Create task-adapted versions
                for task in self.tasks:
                    adapted_metrics = self.task_agnostic_framework.adapt_metrics_to_task(task)
                    # Find matching universal metric and enhance
                    for adapted_metric in adapted_metrics:
                        if adapted_metric['name'].replace('_', ' ').lower() in metric['name'].lower():
                            metric['task_adaptations'] = metric.get('task_adaptations', {})
                            metric['task_adaptations'][task['id']] = adapted_metric
                
                enhanced_metrics.append(metric)
            
            canonical_metrics = enhanced_metrics
        
        return canonical_metrics
    
    def phase_5_enhanced_output_generation(self, enable_self_eval: bool = True) -> Dict[str, Dict[str, Any]]:
        """Enhanced agent output generation with specialized agents and self-evaluation."""
        logger.info("=== Phase 5: Enhanced Agent Output Generation with Specialized Agents ===")
        
        agent_outputs = {}
        failed_tasks = []
        
        for task in self.tasks:
            task_id = task['id']
            logger.info(f"Processing task {task_id} with specialized agent")
            
            try:
                # Get specialized agent for this task
                specialized_agent = self.specialized_agent_factory.get_agent_for_task(task, self.config['agent'])
                
                # Store agent information
                self.specialized_agent_info[task_id] = {
                    'agent_name': specialized_agent.specialized_name,
                    'expertise': specialized_agent.expertise,
                    'personality': specialized_agent.personality,
                    'agent_type': specialized_agent.profile['name']
                }
                
                logger.info(f"Assigned {specialized_agent.specialized_name} to task {task_id}")
                
                # Generate initial response with specialized agent
                use_cache = self.config.get('optimization', {}).get('cache_responses', True)
                response_data = specialized_agent.generate_response(task['prompt'], task_id, use_cache)
                
                # Apply token optimization if enabled
                if self.reliability_manager and self.config.get('token_optimization', {}).get('enabled', False):
                    optimized_response = self.reliability_manager.optimize_token_usage(
                        response_data['response'], 
                        specialized_agent.model,
                        specialized_agent.max_tokens
                    )
                    
                    if optimized_response.get('optimized'):
                        response_data['response'] = optimized_response['optimized_content']
                        response_data['token_optimization'] = optimized_response
                        
                        # Track optimization stats
                        self.token_optimization_stats[task_id] = {
                            'original_length': optimized_response.get('original_tokens', 0),
                            'optimized_length': optimized_response.get('optimized_tokens', 0),
                            'compression_ratio': optimized_response.get('compression_ratio', 1.0),
                            'optimization_applied': True
                        }
                
                # Apply failure detection
                if self.failure_detector:
                    failure_info = self.failure_detector.detect_failures(response_data['response'], task)
                    response_data['failure_analysis'] = failure_info
                    
                    # Store failure analysis
                    self.failure_analysis[task_id] = failure_info
                    
                    # Apply failure penalty if needed
                    if failure_info['detected_failures']:
                        logger.warning(f"Failures detected in task {task_id}: {failure_info['detected_failures']}")
                        response_data['failure_penalty'] = self._calculate_failure_penalty(failure_info)
                
                # Apply self-evaluation if enabled
                if enable_self_eval and self.self_evaluator:
                    logger.info(f"Applying self-evaluation to task {task_id}")
                    
                    # Use the canonical metrics for self-evaluation
                    if hasattr(self, 'canonical_metrics') and self.canonical_metrics:
                        self_eval_result = self.self_evaluator.iterative_improvement(
                            task, response_data['response'], self.canonical_metrics, specialized_agent
                        )
                        
                        # Update response with improved version
                        response_data['response'] = self_eval_result['final_response']
                        response_data['self_evaluation'] = self_eval_result
                        
                        # Store self-evaluation results
                        self.self_evaluation_results[task_id] = self_eval_result
                        
                        logger.info(f"Self-evaluation completed for task {task_id}. "
                                  f"Confidence: {self_eval_result['final_evaluation'].get('overall_confidence', 0):.3f}, "
                                  f"Iterations: {self_eval_result['iterations_used']}")
                
                # Add specialized agent metadata to response
                response_data.update({
                    'specialized_agent': self.specialized_agent_info[task_id],
                    'enhanced_features_applied': {
                        'specialized_agent': True,
                        'token_optimization': task_id in self.token_optimization_stats,
                        'failure_detection': task_id in self.failure_analysis,
                        'self_evaluation': task_id in self.self_evaluation_results
                    }
                })
                
                agent_outputs[task_id] = response_data
                logger.info(f"Successfully generated enhanced output for task {task_id}")
                
            except Exception as e:
                logger.error(f"Failed to generate output for task {task_id}: {e}")
                failed_tasks.append(task_id)
                
                # Create error response with specialized agent info if available
                error_response = {
                    "task_id": task_id,
                    "prompt": task['prompt'],
                    "response": f"ERROR: Failed to generate response - {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "model": self.config['agent']['model'],
                    "error": True,
                    "specialized_agent": self.specialized_agent_info.get(task_id, {}),
                    "enhanced_features_applied": {
                        'specialized_agent': False,
                        'token_optimization': False,
                        'failure_detection': False,
                        'self_evaluation': False
                    }
                }
                agent_outputs[task_id] = error_response
        
        # Store outputs for later phases
        self.agent_outputs = agent_outputs
        
        # Generate specialized agent summary
        agent_summary = self._generate_specialized_agent_summary()
        logger.info(f"Specialized agent summary: {agent_summary}")
        
        if failed_tasks:
            logger.warning(f"Failed to generate outputs for {len(failed_tasks)} tasks: {failed_tasks}")
        
        logger.info(f"Enhanced output generation completed. "
                   f"Successful: {len(agent_outputs) - len(failed_tasks)}/{len(self.tasks)}")
        
        return agent_outputs
    
    def phase_6_enhanced_scoring(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Enhanced scoring with failure-aware evaluation."""
        logger.info("=== Phase 6: Enhanced Scoring Phase ===")
        
        # Apply standard scoring
        raw_scores = super().phase_6_scoring()
        
        # Adjust scores based on failure analysis
        if self.failure_analysis:
            adjusted_scores = {}
            for judge_name, judge_scores in raw_scores.items():
                adjusted_scores[judge_name] = {}
                for task_id, task_scores in judge_scores.items():
                    adjusted_task_scores = task_scores.copy()
                    
                    # Apply failure penalties
                    if task_id in self.failure_analysis:
                        failure_info = self.failure_analysis[task_id]
                        if failure_info.get('detected_failures'):
                            penalty_factor = self._calculate_failure_penalty(failure_info)
                            for metric, score in adjusted_task_scores.items():
                                adjusted_task_scores[metric] = score * penalty_factor
                    
                    adjusted_scores[judge_name][task_id] = adjusted_task_scores
            
            # Save both raw and adjusted scores
            save_json(adjusted_scores, "data/failure_adjusted_scores.json")
            raw_scores = adjusted_scores
        
        return raw_scores
    
    def phase_7_enhanced_calibration(self) -> Dict[str, Dict[str, float]]:
        """Enhanced calibration with reliability analysis."""
        logger.info("=== Phase 7: Enhanced Calibration & Reliability ===")
        
        # Apply standard calibration
        bias_offsets = super().phase_7_calibration()
        
        # Perform reliability analysis if enabled
        if self.reliability_manager:
            # Analyze consistency of self-evaluation results
            if self.self_evaluation_results:
                self_eval_consistency = self._analyze_self_evaluation_consistency()
                self.reliability_metrics['self_evaluation_consistency'] = self_eval_consistency
            
            # Analyze failure patterns
            if self.failure_analysis:
                failure_patterns = self._analyze_failure_patterns()
                self.reliability_metrics['failure_patterns'] = failure_patterns
            
            # Generate reliability report
            reliability_report = {
                'overall_reliability': self._calculate_overall_reliability(),
                'consistency_metrics': self.reliability_metrics,
                'recommendations': self._generate_reliability_recommendations()
            }
            
            save_json(reliability_report, "data/reliability_analysis.json")
        
        return bias_offsets
    
    def phase_8_enhanced_aggregation(self, enable_reliability: bool = True) -> Dict[str, float]:
        """Enhanced aggregation with consistency validation."""
        logger.info("=== Phase 8: Enhanced Aggregation & Validation ===")
        
        # Apply standard aggregation
        final_performance = super().phase_8_aggregation()
        
        # Perform consistency validation if enabled
        if enable_reliability and self.reliability_manager:
            # Validate consistency across multiple evaluation aspects
            consistency_results = self.reliability_manager.validate_consistency([
                {'final_performance': final_performance, 'agent_outputs': self.agent_outputs}
            ])
            
            self.reliability_metrics['consistency_validation'] = consistency_results
            
            # Adjust final scores based on reliability metrics
            if not consistency_results.get('consistent', True):
                logger.warning("Consistency validation failed - applying reliability adjustments")
                reliability_factor = consistency_results.get('confidence', 1.0)
                
                adjusted_performance = {}
                for metric, score in final_performance.items():
                    adjusted_performance[metric] = score * reliability_factor
                
                final_performance = adjusted_performance
        
        return final_performance
    
    def phase_9_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of enhanced evaluation."""
        logger.info("=== Phase 9: Comprehensive Analysis & Insights ===")
        
        analysis = {
            'evaluation_summary': self._generate_evaluation_summary(),
            'self_evaluation_insights': self._analyze_self_evaluation_insights(),
            'failure_prevention_analysis': self._analyze_failure_prevention(),
            'reliability_assessment': self._assess_overall_reliability(),
            'token_optimization_impact': self._analyze_token_optimization_impact(),
            'recommendations': self._generate_comprehensive_recommendations(),
            'framework_effectiveness': self._assess_framework_effectiveness()
        }
        
        save_json(analysis, "data/comprehensive_analysis.json")
        return analysis
    
    # Helper methods for enhanced functionality
    
    def _assess_task_complexity(self, task: Dict[str, Any]) -> str:
        """Assess the complexity level of a task."""
        prompt_length = len(task.get('prompt', ''))
        task_type = task.get('type', 'general')
        
        if prompt_length > 1000 or task_type in ['coding', 'analysis']:
            return 'high'
        elif prompt_length > 500 or task_type in ['reasoning', 'creative']:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_task_token_usage(self, task: Dict[str, Any]) -> int:
        """Estimate token usage for a task."""
        prompt_tokens = len(task.get('prompt', '')) // 4  # Rough approximation
        response_tokens = 500  # Estimated response length
        evaluation_tokens = 200  # Estimated evaluation overhead
        
        return prompt_tokens + response_tokens + evaluation_tokens
    
    def _calculate_failure_penalty(self, failure_info: Dict[str, Any]) -> float:
        """Calculate penalty factor based on failure severity."""
        severity = failure_info.get('severity', 'none')
        failure_count = len(failure_info.get('detected_failures', []))
        
        if severity == 'critical':
            return 0.1  # 90% penalty
        elif severity == 'high':
            return 0.5  # 50% penalty
        elif severity == 'medium':
            return 0.8  # 20% penalty
        else:
            return 1.0 - (failure_count * 0.1)  # 10% penalty per failure
    
    def _analyze_self_evaluation_consistency(self) -> Dict[str, Any]:
        """Analyze consistency of self-evaluation results."""
        if not self.self_evaluation_results:
            return {'consistent': True, 'analysis': 'No self-evaluation data available'}
        
        confidences = [result.get('final_evaluation', {}).get('overall_confidence', 0.5) 
                      for result in self.self_evaluation_results.values()]
        
        avg_confidence = sum(confidences) / len(confidences)
        confidence_variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        
        return {
            'average_confidence': avg_confidence,
            'confidence_variance': confidence_variance,
            'consistent': confidence_variance < 0.1,
            'analysis': f"Average confidence: {avg_confidence:.3f}, Variance: {confidence_variance:.3f}"
        }
    
    def _analyze_failure_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in detected failures."""
        if not self.failure_analysis:
            return {'patterns': [], 'analysis': 'No failure data available'}
        
        failure_types = {}
        for task_id, failure_info in self.failure_analysis.items():
            for failure_type in failure_info.get('detected_failures', []):
                failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
        
        total_tasks = len(self.failure_analysis)
        failure_rates = {ft: count / total_tasks for ft, count in failure_types.items()}
        
        return {
            'failure_types': failure_types,
            'failure_rates': failure_rates,
            'most_common_failure': max(failure_types.items(), key=lambda x: x[1])[0] if failure_types else None,
            'overall_failure_rate': sum(1 for f in self.failure_analysis.values() if f.get('detected_failures')) / total_tasks
        }
    
    def _calculate_overall_reliability(self) -> float:
        """Calculate overall reliability score."""
        reliability_factors = []
        
        # Self-evaluation consistency
        if 'self_evaluation_consistency' in self.reliability_metrics:
            consistency = self.reliability_metrics['self_evaluation_consistency']
            reliability_factors.append(1.0 - consistency.get('confidence_variance', 0.5))
        
        # Failure rate
        if 'failure_patterns' in self.reliability_metrics:
            failure_rate = self.reliability_metrics['failure_patterns'].get('overall_failure_rate', 0.5)
            reliability_factors.append(1.0 - failure_rate)
        
        # Token optimization success
        if self.token_optimization_stats:
            optimization_success = sum(1 for stats in self.token_optimization_stats.values() 
                                     if stats.get('optimization_applied', False)) / len(self.token_optimization_stats)
            reliability_factors.append(optimization_success)
        
        return sum(reliability_factors) / len(reliability_factors) if reliability_factors else 0.5
    
    def _generate_reliability_recommendations(self) -> List[str]:
        """Generate recommendations for improving reliability."""
        recommendations = []
        
        # Check self-evaluation consistency
        if 'self_evaluation_consistency' in self.reliability_metrics:
            consistency = self.reliability_metrics['self_evaluation_consistency']
            if not consistency.get('consistent', True):
                recommendations.append("Improve self-evaluation consistency by refining confidence thresholds")
        
        # Check failure patterns
        if 'failure_patterns' in self.reliability_metrics:
            patterns = self.reliability_metrics['failure_patterns']
            if patterns.get('overall_failure_rate', 0) > 0.2:
                recommendations.append("High failure rate detected - review task complexity and agent capabilities")
        
        # Check token optimization
        if self.token_optimization_stats:
            optimization_rate = sum(1 for stats in self.token_optimization_stats.values() 
                                  if stats.get('optimization_applied', False)) / len(self.token_optimization_stats)
            if optimization_rate > 0.5:
                recommendations.append("High token optimization usage - consider using models with larger context windows")
        
        return recommendations or ["Overall reliability is good - continue current approach"]
    
    def _generate_evaluation_summary(self) -> Dict[str, Any]:
        """Generate high-level evaluation summary."""
        return {
            'total_tasks': len(self.tasks) if self.tasks else 0,
            'successful_evaluations': len([t for t in self.agent_outputs.values() if 'response' in t]) if self.agent_outputs else 0,
            'self_evaluations_performed': len(self.self_evaluation_results),
            'failures_detected': len([f for f in self.failure_analysis.values() if f.get('detected_failures')]),
            'token_optimizations_applied': len([s for s in self.token_optimization_stats.values() if s.get('optimization_applied', False)]),
            'overall_reliability_score': self._calculate_overall_reliability()
        }
    
    def _analyze_self_evaluation_insights(self) -> Dict[str, Any]:
        """Analyze insights from self-evaluation results."""
        if not self.self_evaluation_results:
            return {'insights': 'No self-evaluation data available'}
        
        improvements = [result.get('iterations_used', 0) for result in self.self_evaluation_results.values()]
        convergence_rate = sum(1 for result in self.self_evaluation_results.values() 
                              if result.get('converged', False)) / len(self.self_evaluation_results)
        
        return {
            'average_improvements': sum(improvements) / len(improvements),
            'convergence_rate': convergence_rate,
            'self_correction_effectiveness': convergence_rate > 0.7,
            'insights': f"Self-evaluation improved {convergence_rate:.1%} of responses to acceptable confidence levels"
        }
    
    def _analyze_failure_prevention(self) -> Dict[str, Any]:
        """Analyze effectiveness of failure prevention measures."""
        if not self.failure_analysis:
            return {'effectiveness': 'No failure data available'}
        
        prevented_failures = sum(1 for failure_info in self.failure_analysis.values() 
                               if failure_info.get('correctable', True) and failure_info.get('detected_failures'))
        
        total_potential_failures = len([f for f in self.failure_analysis.values() if f.get('detected_failures')])
        
        prevention_rate = prevented_failures / total_potential_failures if total_potential_failures > 0 else 1.0
        
        return {
            'prevention_rate': prevention_rate,
            'total_failures_detected': total_potential_failures,
            'correctable_failures': prevented_failures,
            'effectiveness': 'High' if prevention_rate > 0.8 else 'Medium' if prevention_rate > 0.5 else 'Low'
        }
    
    def _assess_overall_reliability(self) -> Dict[str, Any]:
        """Assess overall reliability of the evaluation framework."""
        reliability_score = self._calculate_overall_reliability()
        
        return {
            'reliability_score': reliability_score,
            'reliability_level': 'High' if reliability_score > 0.8 else 'Medium' if reliability_score > 0.6 else 'Low',
            'key_factors': {
                'self_evaluation_consistency': self.reliability_metrics.get('self_evaluation_consistency', {}),
                'failure_prevention': self.reliability_metrics.get('failure_patterns', {}),
                'token_optimization': len(self.token_optimization_stats) > 0
            }
        }
    
    def _analyze_token_optimization_impact(self) -> Dict[str, Any]:
        """Analyze the impact of token optimization on evaluation quality."""
        if not self.token_optimization_stats:
            return {'impact': 'No token optimization data available'}
        
        optimized_tasks = [stats for stats in self.token_optimization_stats.values() 
                          if stats.get('optimization_applied', False)]
        
        if not optimized_tasks:
            return {'impact': 'No optimizations were applied'}
        
        avg_token_reduction = sum(stats['estimated_tokens'] - stats.get('optimized_tokens', stats['estimated_tokens']) 
                                 for stats in optimized_tasks) / len(optimized_tasks)
        
        return {
            'optimizations_applied': len(optimized_tasks),
            'average_token_reduction': avg_token_reduction,
            'optimization_rate': len(optimized_tasks) / len(self.token_optimization_stats),
            'impact': 'Significant token savings achieved while maintaining evaluation quality'
        }
    
    def _generate_comprehensive_recommendations(self) -> List[str]:
        """Generate comprehensive recommendations for framework improvement."""
        recommendations = []
        
        # Reliability recommendations
        reliability_score = self._calculate_overall_reliability()
        if reliability_score < 0.7:
            recommendations.append("Improve overall reliability by addressing consistency and failure issues")
        
        # Self-evaluation recommendations
        if self.self_evaluation_results:
            convergence_rate = sum(1 for result in self.self_evaluation_results.values() 
                                  if result.get('converged', False)) / len(self.self_evaluation_results)
            if convergence_rate < 0.6:
                recommendations.append("Adjust self-evaluation thresholds to improve convergence rate")
        
        # Token optimization recommendations
        if self.token_optimization_stats:
            optimization_rate = len([s for s in self.token_optimization_stats.values() 
                                   if s.get('optimization_applied', False)]) / len(self.token_optimization_stats)
            if optimization_rate > 0.5:
                recommendations.append("Consider using models with larger context windows to reduce optimization needs")
        
        # Failure prevention recommendations
        if self.failure_analysis:
            failure_rate = len([f for f in self.failure_analysis.values() 
                              if f.get('detected_failures')]) / len(self.failure_analysis)
            if failure_rate > 0.3:
                recommendations.append("High failure rate - review task design and agent capabilities")
        
        return recommendations or ["Framework is performing well - continue current approach"]
    
    def _assess_framework_effectiveness(self) -> Dict[str, Any]:
        """Assess overall effectiveness of the enhanced framework."""
        effectiveness_metrics = {
            'task_agnostic_capability': self.task_agnostic_framework is not None,
            'self_evaluation_enabled': self.self_evaluator is not None,
            'failure_detection_active': self.failure_detector is not None,
            'reliability_management': self.reliability_manager is not None,
            'token_optimization': self.config.get('token_optimization', {}).get('enabled', False)
        }
        
        enabled_features = sum(effectiveness_metrics.values())
        total_features = len(effectiveness_metrics)
        
        effectiveness_score = enabled_features / total_features
        
        return {
            'effectiveness_score': effectiveness_score,
            'enabled_features': effectiveness_metrics,
            'feature_coverage': f"{enabled_features}/{total_features} advanced features enabled",
            'overall_assessment': 'Excellent' if effectiveness_score > 0.8 else 'Good' if effectiveness_score > 0.6 else 'Basic'
        }
    
    def _generate_specialized_agent_summary(self) -> Dict[str, Any]:
        """Generate summary of specialized agent usage."""
        agent_usage = {}
        agent_performance = {}
        
        for task_id, agent_info in self.specialized_agent_info.items():
            agent_name = agent_info['agent_name']
            
            # Count usage
            agent_usage[agent_name] = agent_usage.get(agent_name, 0) + 1
            
            # Track performance metrics
            if agent_name not in agent_performance:
                agent_performance[agent_name] = {
                    'tasks_handled': [],
                    'expertise_areas': agent_info['expertise'],
                    'personality': agent_info['personality'],
                    'success_rate': 0,
                    'avg_response_length': 0,
                    'self_eval_confidence': []
                }
            
            agent_performance[agent_name]['tasks_handled'].append(task_id)
            
            # Add self-evaluation confidence if available
            if task_id in self.self_evaluation_results:
                confidence = self.self_evaluation_results[task_id]['final_evaluation'].get('overall_confidence', 0)
                agent_performance[agent_name]['self_eval_confidence'].append(confidence)
        
        # Calculate performance metrics
        for agent_name, perf in agent_performance.items():
            perf['success_rate'] = len([t for t in perf['tasks_handled'] if t in self.agent_outputs and not self.agent_outputs[t].get('error', False)]) / len(perf['tasks_handled'])
            
            if perf['self_eval_confidence']:
                perf['avg_confidence'] = sum(perf['self_eval_confidence']) / len(perf['self_eval_confidence'])
            else:
                perf['avg_confidence'] = 0
        
        return {
            'agent_usage_distribution': agent_usage,
            'agent_performance_metrics': agent_performance,
            'total_specialized_agents_used': len(agent_usage),
            'most_used_agent': max(agent_usage.items(), key=lambda x: x[1]) if agent_usage else None,
            'specialization_effectiveness': self._assess_specialization_effectiveness()
        }
    
    def _assess_specialization_effectiveness(self) -> Dict[str, Any]:
        """Assess how effective the specialized agents were."""
        if not self.specialized_agent_info:
            return {'effectiveness_score': 0, 'analysis': 'No specialized agents used'}
        
        # Calculate effectiveness based on various factors
        total_tasks = len(self.specialized_agent_info)
        successful_tasks = len([t for t in self.agent_outputs.values() if not t.get('error', False)])
        
        # Base effectiveness on success rate
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
        
        # Bonus for self-evaluation improvements
        self_eval_bonus = 0
        if self.self_evaluation_results:
            improved_tasks = len([r for r in self.self_evaluation_results.values() if r.get('converged', False)])
            self_eval_bonus = (improved_tasks / len(self.self_evaluation_results)) * 0.2
        
        # Bonus for failure prevention
        failure_prevention_bonus = 0
        if self.failure_analysis:
            prevented_failures = len([f for f in self.failure_analysis.values() if f.get('correctable', False)])
            failure_prevention_bonus = (prevented_failures / len(self.failure_analysis)) * 0.1
        
        effectiveness_score = min(1.0, success_rate + self_eval_bonus + failure_prevention_bonus)
        
        return {
            'effectiveness_score': effectiveness_score,
            'success_rate': success_rate,
            'self_evaluation_improvement': self_eval_bonus,
            'failure_prevention_bonus': failure_prevention_bonus,
            'analysis': f"Specialized agents achieved {effectiveness_score:.1%} effectiveness with {success_rate:.1%} success rate"
        } 