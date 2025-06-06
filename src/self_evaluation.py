"""
Self-evaluation module for agents to assess their own work and reduce failures.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
from .utils import validate_json_response, normalize_score
from .judge import Judge

logger = logging.getLogger(__name__)

class SelfEvaluator:
    """Enables agents to self-evaluate their work before submission."""
    
    def __init__(self, agent_config: Dict[str, Any], cache=None):
        """Initialize self-evaluator with same model as agent."""
        self.agent_config = agent_config
        self.cache = cache
        
        # Create a judge instance using the agent's configuration
        self.self_judge = Judge("SelfEvaluator", agent_config, cache)
        
        # Self-evaluation thresholds
        self.confidence_threshold = 0.7
        self.retry_threshold = 0.5
        self.max_self_iterations = 3
        
    def evaluate_response(self, 
                         task: Dict[str, Any], 
                         response: str,
                         metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Have the agent evaluate its own response."""
        logger.info(f"Self-evaluating response for task {task.get('id', 'unknown')}")
        
        # Build self-evaluation prompt
        prompt = self._build_self_evaluation_prompt(task, response, metrics)
        
        try:
            # Get self-evaluation
            evaluation_response = self.self_judge.call(prompt)
            evaluation = validate_json_response(evaluation_response)
            
            # Process evaluation results
            processed_eval = self._process_self_evaluation(evaluation, metrics)
            
            logger.info(f"Self-evaluation completed. Confidence: {processed_eval['overall_confidence']:.3f}")
            return processed_eval
            
        except Exception as e:
            logger.error(f"Self-evaluation failed: {e}")
            return self._default_evaluation(metrics)
    
    def iterative_improvement(self, 
                            task: Dict[str, Any], 
                            initial_response: str,
                            metrics: List[Dict[str, Any]],
                            agent) -> Dict[str, Any]:
        """Iteratively improve response through self-evaluation."""
        logger.info(f"Starting iterative improvement for task {task.get('id', 'unknown')}")
        
        current_response = initial_response
        iteration = 0
        improvement_history = []
        
        while iteration < self.max_self_iterations:
            # Self-evaluate current response
            evaluation = self.evaluate_response(task, current_response, metrics)
            improvement_history.append({
                'iteration': iteration,
                'response': current_response,
                'evaluation': evaluation,
                'confidence': evaluation['overall_confidence']
            })
            
            # Check if response is good enough
            if evaluation['overall_confidence'] >= self.confidence_threshold:
                logger.info(f"Response meets confidence threshold after {iteration} iterations")
                break
            
            # Check if response is too poor to continue
            if evaluation['overall_confidence'] < self.retry_threshold:
                logger.warning(f"Response quality too low ({evaluation['overall_confidence']:.3f}), attempting improvement")
                
                # Generate improvement prompt
                improvement_prompt = self._build_improvement_prompt(task, current_response, evaluation)
                
                try:
                    # Generate improved response
                    improved_response_data = agent.generate_response(improvement_prompt, task.get('id', 'unknown'))
                    current_response = improved_response_data['response']
                    iteration += 1
                except Exception as e:
                    logger.error(f"Failed to generate improved response: {e}")
                    break
            else:
                break
        
        return {
            'final_response': current_response,
            'final_evaluation': evaluation,
            'improvement_history': improvement_history,
            'iterations_used': iteration,
            'converged': evaluation['overall_confidence'] >= self.confidence_threshold
        }
    
    def detect_failure_patterns(self, 
                               evaluation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze evaluation history to detect common failure patterns."""
        logger.info("Analyzing failure patterns from evaluation history")
        
        failure_patterns = {
            'low_confidence_tasks': [],
            'common_failure_metrics': {},
            'task_type_failures': {},
            'improvement_trends': {},
            'recommendations': []
        }
        
        # Analyze low confidence tasks
        for eval_data in evaluation_history:
            if eval_data.get('overall_confidence', 1.0) < self.confidence_threshold:
                failure_patterns['low_confidence_tasks'].append({
                    'task_id': eval_data.get('task_id'),
                    'confidence': eval_data.get('overall_confidence'),
                    'weak_metrics': [m for m, s in eval_data.get('metric_scores', {}).items() if s < 0.5]
                })
        
        # Analyze metric-specific failures
        metric_failures = {}
        for eval_data in evaluation_history:
            for metric, score in eval_data.get('metric_scores', {}).items():
                if metric not in metric_failures:
                    metric_failures[metric] = []
                metric_failures[metric].append(score)
        
        for metric, scores in metric_failures.items():
            avg_score = sum(scores) / len(scores)
            failure_rate = sum(1 for s in scores if s < 0.5) / len(scores)
            failure_patterns['common_failure_metrics'][metric] = {
                'average_score': avg_score,
                'failure_rate': failure_rate,
                'needs_attention': failure_rate > 0.3
            }
        
        # Generate recommendations
        failure_patterns['recommendations'] = self._generate_improvement_recommendations(failure_patterns)
        
        return failure_patterns
    
    def _build_self_evaluation_prompt(self, 
                                    task: Dict[str, Any], 
                                    response: str, 
                                    metrics: List[Dict[str, Any]]) -> str:
        """Build prompt for self-evaluation."""
        metrics_desc = []
        for metric in metrics:
            metrics_desc.append(f"- {metric['name']}: {metric['definition']} (Scale: {metric['scale']})")
        
        prompt = f"""You are evaluating your own response to a task. Be honest and critical in your assessment.

TASK: {task.get('prompt', 'No prompt provided')}

YOUR RESPONSE: {response}

EVALUATION METRICS:
{chr(10).join(metrics_desc)}

Please evaluate your response on each metric and provide:
1. A score for each metric (0-1 scale)
2. Your confidence in the response (0-1 scale)
3. Specific issues you identify
4. Suggestions for improvement

Output format (JSON):
{{
    "metric_scores": {{
        "MetricName1": 0.8,
        "MetricName2": 0.6,
        ...
    }},
    "overall_confidence": 0.7,
    "identified_issues": ["issue1", "issue2"],
    "improvement_suggestions": ["suggestion1", "suggestion2"],
    "reasoning": "Brief explanation of your evaluation"
}}

Return only valid JSON - no additional text."""
        
        return prompt
    
    def _build_improvement_prompt(self, 
                                task: Dict[str, Any], 
                                current_response: str, 
                                evaluation: Dict[str, Any]) -> str:
        """Build prompt for response improvement."""
        issues = evaluation.get('identified_issues', [])
        suggestions = evaluation.get('improvement_suggestions', [])
        
        prompt = f"""Your previous response needs improvement. Please provide a better response addressing the identified issues.

ORIGINAL TASK: {task.get('prompt', 'No prompt provided')}

YOUR PREVIOUS RESPONSE: {current_response}

IDENTIFIED ISSUES:
{chr(10).join(f"- {issue}" for issue in issues)}

IMPROVEMENT SUGGESTIONS:
{chr(10).join(f"- {suggestion}" for suggestion in suggestions)}

Please provide an improved response that addresses these issues while maintaining the core requirements of the task."""
        
        return prompt
    
    def _process_self_evaluation(self, 
                               evaluation: Dict[str, Any], 
                               metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process and normalize self-evaluation results."""
        processed = {
            'metric_scores': {},
            'overall_confidence': 0.0,
            'identified_issues': evaluation.get('identified_issues', []),
            'improvement_suggestions': evaluation.get('improvement_suggestions', []),
            'reasoning': evaluation.get('reasoning', '')
        }
        
        # Normalize metric scores
        metric_scores = evaluation.get('metric_scores', {})
        for metric in metrics:
            metric_name = metric['name']
            if metric_name in metric_scores:
                score = normalize_score(metric_scores[metric_name], metric['scale'])
                processed['metric_scores'][metric_name] = score
            else:
                processed['metric_scores'][metric_name] = 0.0
        
        # Calculate overall confidence
        confidence = evaluation.get('overall_confidence', 0.0)
        processed['overall_confidence'] = max(0.0, min(1.0, float(confidence)))
        
        return processed
    
    def _default_evaluation(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return default evaluation when self-evaluation fails."""
        return {
            'metric_scores': {metric['name']: 0.5 for metric in metrics},
            'overall_confidence': 0.5,
            'identified_issues': ['Self-evaluation failed'],
            'improvement_suggestions': ['Manual review recommended'],
            'reasoning': 'Self-evaluation process encountered an error'
        }
    
    def _generate_improvement_recommendations(self, 
                                           failure_patterns: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on failure patterns."""
        recommendations = []
        
        # Check for consistently failing metrics
        for metric, data in failure_patterns.get('common_failure_metrics', {}).items():
            if data.get('needs_attention', False):
                recommendations.append(
                    f"Focus on improving {metric} - current failure rate: {data['failure_rate']:.1%}"
                )
        
        # Check for low confidence tasks
        low_conf_count = len(failure_patterns.get('low_confidence_tasks', []))
        if low_conf_count > 0:
            recommendations.append(
                f"Review {low_conf_count} low-confidence tasks for common patterns"
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append("Overall performance is good - continue current approach")
        
        return recommendations

    def evaluate_and_improve_response(self, 
                                    prompt: str, 
                                    response: str, 
                                    task_id: str) -> Dict[str, Any]:
        """
        Evaluate and improve a response for adaptive evaluation.
        
        Args:
            prompt: The task prompt
            response: The agent's response
            task_id: The task identifier
            
        Returns:
            Dictionary with final_response and evaluation details
        """
        try:
            # Create a simplified task dict for compatibility
            task = {
                'id': task_id,
                'prompt': prompt
            }
            
            # Create basic metrics for self-evaluation
            basic_metrics = [
                {
                    'name': 'Task_Completion',
                    'definition': 'How well the response addresses the task requirements',
                    'scale': '0-1 (0=not addressed, 1=fully addressed)'
                },
                {
                    'name': 'Response_Quality', 
                    'definition': 'Overall quality and coherence of the response',
                    'scale': '0-1 (0=poor quality, 1=excellent quality)'
                },
                {
                    'name': 'Accuracy',
                    'definition': 'Factual correctness and logical soundness',
                    'scale': '0-1 (0=incorrect, 1=accurate)'
                }
            ]
            
            # Perform self-evaluation
            evaluation = self.evaluate_response(task, response, basic_metrics)
            
            # If confidence is low, attempt improvement
            if evaluation['overall_confidence'] < self.confidence_threshold:
                logger.info(f"Response confidence ({evaluation['overall_confidence']:.3f}) below threshold, attempting improvement")
                
                # For adaptive evaluation, we'll do a simplified improvement
                # since we don't have access to the original agent here
                improvement_suggestions = evaluation.get('improvement_suggestions', [])
                
                if improvement_suggestions:
                    # Build improved response prompt
                    improvement_prompt = self._build_improvement_prompt(task, response, evaluation)
                    
                    try:
                        # Try to get improved response from self-judge
                        improved_response_data = self.self_judge.call(improvement_prompt)
                        
                        # Extract improved response
                        if isinstance(improved_response_data, str):
                            improved_response = improved_response_data
                        else:
                            improved_response = improved_response_data.get('response', response)
                        
                        return {
                            'final_response': improved_response,
                            'original_response': response,
                            'evaluation': evaluation,
                            'improved': True,
                            'improvement_applied': True
                        }
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate improved response: {e}")
                        # Fall back to original response
                        return {
                            'final_response': response,
                            'original_response': response,
                            'evaluation': evaluation,
                            'improved': False,
                            'improvement_attempted': True,
                            'improvement_error': str(e)
                        }
            
            # Response is good enough, return as-is
            return {
                'final_response': response,
                'original_response': response,
                'evaluation': evaluation,
                'improved': False,
                'confidence': evaluation['overall_confidence']
            }
            
        except Exception as e:
            logger.error(f"Self-evaluation failed for {task_id}: {e}")
            # Return original response if evaluation fails
            return {
                'final_response': response,
                'original_response': response,
                'evaluation': None,
                'improved': False,
                'error': str(e)
            }

class FailureDetector:
    """Detects and prevents common failure cases in agent responses."""
    
    def __init__(self):
        self.failure_patterns = {
            'empty_response': r'^\\s*$',
            'error_message': r'(error|exception|failed|cannot|unable)',
            'incomplete_json': r'\\{[^}]*$',
            'truncated_response': r'\\.\\.\\.$',
            'repetitive_text': r'(.{10,})\\1{3,}',
        }
        
    def detect_failures(self, response: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential failures in agent response."""
        import re
        
        failures = {
            'detected_failures': [],
            'severity': 'none',
            'correctable': True,
            'recommendations': []
        }
        
        # Check for empty response
        if not response or response.strip() == '':
            failures['detected_failures'].append('empty_response')
            failures['severity'] = 'critical'
            failures['correctable'] = False
            
        # Check for error messages
        if re.search(self.failure_patterns['error_message'], response.lower()):
            failures['detected_failures'].append('error_message')
            failures['severity'] = 'high'
            
        # Check for incomplete JSON (if task expects JSON)
        if 'json' in task.get('prompt', '').lower():
            if re.search(self.failure_patterns['incomplete_json'], response):
                failures['detected_failures'].append('incomplete_json')
                failures['severity'] = 'high'
        
        # Check for truncated response
        if re.search(self.failure_patterns['truncated_response'], response):
            failures['detected_failures'].append('truncated_response')
            failures['severity'] = 'medium'
            
        # Check for repetitive text
        if re.search(self.failure_patterns['repetitive_text'], response):
            failures['detected_failures'].append('repetitive_text')
            failures['severity'] = 'medium'
        
        # Generate recommendations
        failures['recommendations'] = self._generate_failure_recommendations(failures)
        
        return failures
    
    def _generate_failure_recommendations(self, failures: Dict[str, Any]) -> List[str]:
        """Generate recommendations for detected failures."""
        recommendations = []
        
        for failure in failures['detected_failures']:
            if failure == 'empty_response':
                recommendations.append("Regenerate response - empty output detected")
            elif failure == 'error_message':
                recommendations.append("Review task requirements - error indicators found")
            elif failure == 'incomplete_json':
                recommendations.append("Complete JSON structure - incomplete format detected")
            elif failure == 'truncated_response':
                recommendations.append("Extend response - truncation detected")
            elif failure == 'repetitive_text':
                recommendations.append("Reduce repetition - repetitive patterns detected")
        
        return recommendations 