"""
Reliability and replicability module for long-running agentic tasks.
"""

import logging
import time
import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from .utils import save_json, load_json

logger = logging.getLogger(__name__)

class ReliabilityManager:
    """Manages reliability and replicability for long-running tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checkpoints_dir = "checkpoints"
        self.reliability_threshold = config.get('reliability', {}).get('threshold', 0.8)
        self.max_retries = config.get('reliability', {}).get('max_retries', 3)
        self.checkpoint_interval = config.get('reliability', {}).get('checkpoint_interval', 300)  # 5 minutes
        
        # Token optimization settings
        self.token_limits = {
            'gpt-4': 8192,
            'gpt-4-turbo': 128000,
            'claude-3': 200000,
            'gemini-pro': 32768,
            'gemini-flash': 1048576
        }
        
        # State tracking
        self.task_states = {}
        self.execution_history = []
        self.last_checkpoint = None
        
    def create_checkpoint(self, 
                         task_id: str, 
                         state: Dict[str, Any], 
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a checkpoint for task state."""
        checkpoint_id = self._generate_checkpoint_id(task_id, state)
        
        checkpoint_data = {
            'checkpoint_id': checkpoint_id,
            'task_id': task_id,
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'metadata': metadata or {},
            'state_hash': self._compute_state_hash(state)
        }
        
        # Save checkpoint
        checkpoint_path = f"{self.checkpoints_dir}/{checkpoint_id}.json"
        save_json(checkpoint_data, checkpoint_path)
        
        self.last_checkpoint = checkpoint_data
        logger.info(f"Created checkpoint {checkpoint_id} for task {task_id}")
        
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Restore task state from checkpoint."""
        try:
            checkpoint_path = f"{self.checkpoints_dir}/{checkpoint_id}.json"
            checkpoint_data = load_json(checkpoint_path)
            
            # Verify checkpoint integrity
            if self._verify_checkpoint_integrity(checkpoint_data):
                logger.info(f"Restored checkpoint {checkpoint_id}")
                return checkpoint_data
            else:
                logger.error(f"Checkpoint {checkpoint_id} integrity check failed")
                return None
                
        except Exception as e:
            logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
            return None
    
    def ensure_replicability(self, 
                           task: Dict[str, Any], 
                           agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure task execution is replicable."""
        # Create deterministic seed from task content
        task_hash = self._compute_task_hash(task)
        
        # Set deterministic parameters
        replicable_config = agent_config.copy()
        replicable_config['temperature'] = 0.0  # Deterministic output
        replicable_config['seed'] = int(task_hash[:8], 16)  # Deterministic seed
        
        # Add replicability metadata
        replicability_info = {
            'task_hash': task_hash,
            'config_hash': self._compute_config_hash(replicable_config),
            'timestamp': datetime.now().isoformat(),
            'replicable': True
        }
        
        return {
            'config': replicable_config,
            'replicability_info': replicability_info
        }
    
    def monitor_long_running_task(self, 
                                task_id: str, 
                                execution_func, 
                                *args, **kwargs) -> Dict[str, Any]:
        """Monitor and manage long-running task execution."""
        start_time = time.time()
        last_checkpoint_time = start_time
        
        execution_log = {
            'task_id': task_id,
            'start_time': start_time,
            'checkpoints': [],
            'errors': [],
            'performance_metrics': {},
            'completed': False
        }
        
        try:
            # Execute with monitoring
            result = None
            current_state = {'status': 'running', 'progress': 0.0}
            
            # Periodic checkpoint creation
            while not current_state.get('completed', False):
                current_time = time.time()
                
                # Create checkpoint if interval elapsed
                if current_time - last_checkpoint_time >= self.checkpoint_interval:
                    checkpoint_id = self.create_checkpoint(task_id, current_state)
                    execution_log['checkpoints'].append({
                        'checkpoint_id': checkpoint_id,
                        'timestamp': current_time,
                        'progress': current_state.get('progress', 0.0)
                    })
                    last_checkpoint_time = current_time
                
                # Execute next step
                try:
                    result = execution_func(*args, **kwargs)
                    current_state['completed'] = True
                    current_state['progress'] = 1.0
                    
                except Exception as e:
                    execution_log['errors'].append({
                        'timestamp': current_time,
                        'error': str(e),
                        'state': current_state.copy()
                    })
                    
                    # Attempt recovery
                    if len(execution_log['errors']) < self.max_retries:
                        logger.warning(f"Task {task_id} failed, attempting recovery")
                        time.sleep(2 ** len(execution_log['errors']))  # Exponential backoff
                        continue
                    else:
                        raise e
            
            # Final checkpoint
            final_checkpoint_id = self.create_checkpoint(task_id, {
                'status': 'completed',
                'result': result,
                'progress': 1.0
            })
            
            execution_log['completed'] = True
            execution_log['end_time'] = time.time()
            execution_log['duration'] = execution_log['end_time'] - start_time
            execution_log['final_checkpoint'] = final_checkpoint_id
            
            return {
                'result': result,
                'execution_log': execution_log,
                'reliable': len(execution_log['errors']) == 0
            }
            
        except Exception as e:
            execution_log['failed'] = True
            execution_log['final_error'] = str(e)
            execution_log['end_time'] = time.time()
            
            logger.error(f"Long-running task {task_id} failed: {e}")
            return {
                'result': None,
                'execution_log': execution_log,
                'reliable': False
            }
    
    def optimize_token_usage(self, 
                           prompt: str, 
                           model: str, 
                           max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Optimize token usage for model constraints."""
        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = len(prompt) // 4
        
        # Get model token limit
        model_limit = self._get_model_token_limit(model)
        effective_limit = min(model_limit, max_tokens or model_limit)
        
        optimization_result = {
            'original_prompt': prompt,
            'estimated_tokens': estimated_tokens,
            'model_limit': model_limit,
            'effective_limit': effective_limit,
            'needs_optimization': estimated_tokens > effective_limit * 0.8,  # 80% threshold
            'optimized_prompt': prompt,
            'optimization_applied': False
        }
        
        # Apply optimization if needed
        if optimization_result['needs_optimization']:
            optimized_prompt = self._optimize_prompt(prompt, effective_limit)
            optimization_result['optimized_prompt'] = optimized_prompt
            optimization_result['optimization_applied'] = True
            optimization_result['optimized_tokens'] = len(optimized_prompt) // 4
            
            logger.info(f"Token optimization applied: {estimated_tokens} -> {optimization_result['optimized_tokens']} tokens")
        
        return optimization_result
    
    def validate_consistency(self, 
                           results: List[Dict[str, Any]], 
                           tolerance: float = 0.1) -> Dict[str, Any]:
        """Validate consistency across multiple runs."""
        if len(results) < 2:
            return {'consistent': True, 'confidence': 1.0, 'analysis': 'Single result - no comparison possible'}
        
        consistency_metrics = {
            'score_variance': 0.0,
            'output_similarity': 0.0,
            'metric_consistency': {},
            'overall_consistency': 0.0
        }
        
        # Analyze score variance
        if all('final_performance' in result for result in results):
            scores = [result['final_performance'].get('overall_score', 0) for result in results]
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            consistency_metrics['score_variance'] = variance
        
        # Analyze output similarity
        if all('agent_outputs' in result for result in results):
            similarities = []
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    sim = self._compute_output_similarity(
                        results[i]['agent_outputs'], 
                        results[j]['agent_outputs']
                    )
                    similarities.append(sim)
            
            consistency_metrics['output_similarity'] = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Calculate overall consistency
        consistency_metrics['overall_consistency'] = (
            (1.0 - min(consistency_metrics['score_variance'], 1.0)) * 0.5 +
            consistency_metrics['output_similarity'] * 0.5
        )
        
        return {
            'consistent': consistency_metrics['overall_consistency'] >= self.reliability_threshold,
            'confidence': consistency_metrics['overall_consistency'],
            'metrics': consistency_metrics,
            'analysis': self._generate_consistency_analysis(consistency_metrics, tolerance)
        }
    
    def _generate_checkpoint_id(self, task_id: str, state: Dict[str, Any]) -> str:
        """Generate unique checkpoint ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_hash = self._compute_state_hash(state)[:8]
        return f"{task_id}_{timestamp}_{state_hash}"
    
    def _compute_state_hash(self, state: Dict[str, Any]) -> str:
        """Compute hash of task state for integrity checking."""
        state_str = json.dumps(state, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()
    
    def _compute_task_hash(self, task: Dict[str, Any]) -> str:
        """Compute deterministic hash of task for replicability."""
        # Use only stable task properties
        stable_task = {
            'id': task.get('id'),
            'prompt': task.get('prompt'),
            'type': task.get('type'),
            'tier': task.get('tier')
        }
        task_str = json.dumps(stable_task, sort_keys=True)
        return hashlib.sha256(task_str.encode()).hexdigest()
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute hash of configuration for replicability."""
        # Use only relevant config properties
        relevant_config = {
            'model': config.get('model'),
            'temperature': config.get('temperature'),
            'max_tokens': config.get('max_tokens'),
            'seed': config.get('seed')
        }
        config_str = json.dumps(relevant_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _verify_checkpoint_integrity(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Verify checkpoint data integrity."""
        try:
            expected_hash = checkpoint_data.get('state_hash')
            actual_hash = self._compute_state_hash(checkpoint_data.get('state', {}))
            return expected_hash == actual_hash
        except Exception:
            return False
    
    def _get_model_token_limit(self, model: str) -> int:
        """Get token limit for specific model."""
        # Check for exact match first
        if model in self.token_limits:
            return self.token_limits[model]
        
        # Check for partial matches
        for model_key, limit in self.token_limits.items():
            if model_key in model.lower():
                return limit
        
        # Default conservative limit
        return 4096
    
    def _optimize_prompt(self, prompt: str, target_tokens: int) -> str:
        """Optimize prompt to fit within token limits."""
        target_chars = target_tokens * 4  # Rough approximation
        
        if len(prompt) <= target_chars:
            return prompt
        
        # Simple truncation with preservation of structure
        lines = prompt.split('\n')
        optimized_lines = []
        current_length = 0
        
        # Preserve important sections (task description, metrics)
        important_keywords = ['TASK:', 'METRICS:', 'OUTPUT:', 'EVALUATION:']
        
        for line in lines:
            if any(keyword in line for keyword in important_keywords):
                # Always include important lines
                optimized_lines.append(line)
                current_length += len(line) + 1
            elif current_length + len(line) + 1 < target_chars * 0.9:  # Leave some buffer
                optimized_lines.append(line)
                current_length += len(line) + 1
            else:
                # Add truncation indicator
                if not any('...' in l for l in optimized_lines[-3:]):
                    optimized_lines.append("... [content truncated for token optimization] ...")
                break
        
        return '\n'.join(optimized_lines)
    
    def _compute_output_similarity(self, outputs1: Dict[str, Any], outputs2: Dict[str, Any]) -> float:
        """Compute similarity between two sets of outputs."""
        if not outputs1 or not outputs2:
            return 0.0
        
        # Simple similarity based on common task IDs and response length similarity
        common_tasks = set(outputs1.keys()) & set(outputs2.keys())
        if not common_tasks:
            return 0.0
        
        similarities = []
        for task_id in common_tasks:
            response1 = outputs1[task_id].get('response', '')
            response2 = outputs2[task_id].get('response', '')
            
            # Simple length-based similarity (can be enhanced with semantic similarity)
            len1, len2 = len(response1), len(response2)
            if len1 == 0 and len2 == 0:
                similarities.append(1.0)
            elif len1 == 0 or len2 == 0:
                similarities.append(0.0)
            else:
                length_sim = 1.0 - abs(len1 - len2) / max(len1, len2)
                similarities.append(length_sim)
        
        return sum(similarities) / len(similarities)
    
    def _generate_consistency_analysis(self, metrics: Dict[str, Any], tolerance: float) -> str:
        """Generate human-readable consistency analysis."""
        analysis_parts = []
        
        if metrics['score_variance'] <= tolerance:
            analysis_parts.append("Score variance within acceptable range")
        else:
            analysis_parts.append(f"High score variance detected ({metrics['score_variance']:.3f})")
        
        if metrics['output_similarity'] >= 0.8:
            analysis_parts.append("High output similarity across runs")
        elif metrics['output_similarity'] >= 0.6:
            analysis_parts.append("Moderate output similarity")
        else:
            analysis_parts.append("Low output similarity - potential reliability issues")
        
        overall = metrics['overall_consistency']
        if overall >= 0.9:
            analysis_parts.append("Excellent overall consistency")
        elif overall >= 0.7:
            analysis_parts.append("Good overall consistency")
        elif overall >= 0.5:
            analysis_parts.append("Moderate consistency - monitoring recommended")
        else:
            analysis_parts.append("Poor consistency - investigation required")
        
        return "; ".join(analysis_parts)

class TaskAgnosticFramework:
    """Framework for creating truly task-agnostic evaluation systems."""
    
    def __init__(self):
        self.universal_metrics = [
            {
                'name': 'Task_Completion',
                'definition': 'Whether the agent completed the requested task',
                'scale': 'binary',
                'weight': 0.3,
                'universal': True
            },
            {
                'name': 'Response_Quality',
                'definition': 'Overall quality and appropriateness of the response',
                'scale': 'continuous',
                'weight': 0.25,
                'universal': True
            },
            {
                'name': 'Instruction_Following',
                'definition': 'How well the agent followed the given instructions',
                'scale': 'continuous',
                'weight': 0.25,
                'universal': True
            },
            {
                'name': 'Error_Handling',
                'definition': 'How gracefully the agent handled errors or edge cases',
                'scale': 'continuous',
                'weight': 0.2,
                'universal': True
            }
        ]
        
        self.task_type_adaptations = {
            'reasoning': {'weight_adjustments': {'Response_Quality': 1.2}},
            'creative': {'weight_adjustments': {'Response_Quality': 1.3, 'Instruction_Following': 0.8}},
            'factual': {'weight_adjustments': {'Task_Completion': 1.2, 'Error_Handling': 1.1}},
            'coding': {'weight_adjustments': {'Task_Completion': 1.4, 'Error_Handling': 1.3}},
            'analysis': {'weight_adjustments': {'Response_Quality': 1.2, 'Instruction_Following': 1.1}}
        }
    
    def adapt_metrics_to_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Adapt universal metrics to specific task type."""
        task_type = task.get('type', 'general')
        adapted_metrics = []
        
        for metric in self.universal_metrics:
            adapted_metric = metric.copy()
            
            # Apply task-specific weight adjustments
            if task_type in self.task_type_adaptations:
                adjustments = self.task_type_adaptations[task_type].get('weight_adjustments', {})
                if metric['name'] in adjustments:
                    adapted_metric['weight'] *= adjustments[metric['name']]
            
            # Add task-specific context to definition
            adapted_metric['definition'] = self._contextualize_metric_definition(
                metric['definition'], task
            )
            
            adapted_metrics.append(adapted_metric)
        
        return adapted_metrics
    
    def _contextualize_metric_definition(self, definition: str, task: Dict[str, Any]) -> str:
        """Add task-specific context to metric definition."""
        task_type = task.get('type', 'general')
        task_context = task.get('prompt', '')[:100] + "..." if len(task.get('prompt', '')) > 100 else task.get('prompt', '')
        
        contextualized = f"{definition} (in the context of {task_type} task: '{task_context}')"
        return contextualized 