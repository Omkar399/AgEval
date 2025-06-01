"""
Judge implementation for the three-judge evaluation system.
"""

import openai
import anthropic
import json
import logging
from typing import Dict, Any, List, Optional
from .utils import ResponseCache, generate_cache_key, validate_json_response, retry_with_backoff
import google.generativeai as genai

logger = logging.getLogger(__name__)

class Judge:
    """Individual judge that uses an LLM to evaluate agent outputs."""
    
    def __init__(self, name: str, config: Dict[str, Any], cache: Optional[ResponseCache] = None):
        self.name = name
        self.config = config
        self.provider = config['provider']
        self.model = config['model']
        self.api_key = config['api_key']
        self.temperature = config.get('temperature', 0)
        self.max_tokens = config.get('max_tokens', 4000)
        self.cache = cache
        
        # Initialize API client
        if self.provider == 'openai':
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.provider == 'anthropic':
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == 'google':
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    @retry_with_backoff
    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """Make API call to the judge's LLM."""
        if self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        
        elif self.provider == 'anthropic':
            # Convert messages format for Anthropic
            system_message = ""
            user_messages = []
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    user_messages.append(msg)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_message,
                messages=user_messages
            )
            return response.content[0].text
        
        elif self.provider == 'google':
            # Convert messages to single prompt for Gemini
            prompt = ""
            for msg in messages:
                if msg['role'] == 'system':
                    prompt += f"System: {msg['content']}\n\n"
                elif msg['role'] == 'user':
                    prompt += f"User: {msg['content']}\n\n"
                elif msg['role'] == 'assistant':
                    prompt += f"Assistant: {msg['content']}\n\n"
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
            
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
    
    def call(self, prompt: str, use_cache: bool = True) -> str:
        """Call the judge with a prompt, optionally using cache."""
        # Generate cache key
        cache_key = generate_cache_key(self.name, self.model, prompt)
        
        # Try cache first
        if use_cache and self.cache:
            cached_response = self.cache.get(cache_key)
            if cached_response:
                logger.debug(f"Using cached response for {self.name}")
                return cached_response
        
        # Make API call
        messages = [{"role": "user", "content": prompt}]
        response = self._call_api(messages)
        
        # Cache response
        if use_cache and self.cache:
            self.cache.set(cache_key, response)
        
        logger.info(f"Judge {self.name} completed evaluation")
        return response
    
    def propose_metrics(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Propose five evaluation metrics based on the task suite."""
        task_descriptions = []
        for i, task in enumerate(tasks, 1):
            task_descriptions.append(f"{i}. [{task['id']}] \"{task['prompt']}\"")

        prompt = f"""You are evaluating Agent outputs across a diverse set of tasks. Each task is listed below with its ID and prompt. Your job is to propose exactly five evaluation metrics that apply across all tasks. For each metric, provide:

1. Name (short, e.g., "Response Accuracy")
2. Definition (1–2 sentences explaining how to assess it with confidence scoring)
3. Scale (Always use "Confidence [0.0-1.0]" for nuanced evaluation)

Focus on metrics that can be assessed with confidence levels rather than binary pass/fail. Consider aspects like:
- Quality and accuracy of responses
- Completeness and thoroughness  
- Relevance to the task requirements
- Clarity and coherence of explanations
- Technical correctness when applicable

The tasks:
{chr(10).join(task_descriptions)}

Output format (JSON array of five objects):
[
  {{
    "name": "Response Accuracy", 
    "definition": "Confidence in the factual correctness and logical soundness of the agent's response, considering accuracy of information, calculations, and reasoning.",
    "scale": "Confidence [0.0-1.0]"
  }},
  ...
]

Return only valid JSON - no additional text or markdown formatting."""
        
        response = self.call(prompt)
        try:
            metrics = validate_json_response(response)
            if not isinstance(metrics, list) or len(metrics) != 5:
                raise ValueError(f"Expected 5 metrics, got {len(metrics) if isinstance(metrics, list) else 'non-list'}")
            return metrics
        except Exception as e:
            logger.error(f"Judge {self.name} failed to propose valid metrics: {e}")
            raise
    
    def score_outputs(self, tasks_outputs: Dict[str, Dict[str, Any]], metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Score agent outputs using confidence-based evaluation."""
        # Build metrics description
        metrics_desc = []
        for i, metric in enumerate(metrics, 1):
            metrics_desc.append(f"{i}. Name: \"{metric['name']}\"\n   Definition: \"{metric['definition']}\"\n   Scale: {metric['scale']}")
        
        # Build tasks description
        tasks_desc = []
        for task_id, output_data in tasks_outputs.items():
            tasks_desc.append(f"\"{task_id}\": {{\n    \"prompt\": \"{output_data['prompt']}\",\n    \"response\": \"{output_data['response']}\"\n  }}")

        prompt = f"""You are an expert evaluator using confidence-based scoring. Below are evaluation metrics, each with a confidence scale from 0.0 to 1.0. For each task, evaluate the agent's response and provide your confidence score for each metric.

Confidence Scoring Guidelines:
- 1.0: Extremely confident - Exceptional quality, fully meets or exceeds expectations
- 0.8-0.9: Very confident - High quality with minor areas for improvement  
- 0.6-0.7: Moderately confident - Adequate quality with some notable issues
- 0.4-0.5: Low confidence - Below expectations with significant problems
- 0.2-0.3: Very low confidence - Poor quality with major deficiencies
- 0.0-0.1: No confidence - Completely inadequate or incorrect

Evaluation Metrics:
{chr(10).join(metrics_desc)}

Agent Outputs:
{{
  {',\n  '.join(tasks_desc)}
}}

For each task, evaluate the agent's response against each metric and provide a confidence score between 0.0 and 1.0. Consider the specific requirements of each task and be precise in your scoring.

Output (JSON):
Return an object {{ TaskID: {{ "MetricName": confidence_score, ... }} }} for every TaskID.
All scores must be floating point numbers between 0.0 and 1.0.

Return only valid JSON - no additional text or markdown formatting."""
        
        response = self.call(prompt)
        try:
            scores = validate_json_response(response)
            
            # Validate and normalize confidence scores
            normalized_scores = {}
            for task_id, task_scores in scores.items():
                normalized_scores[task_id] = {}
                for metric in metrics:
                    metric_name = metric['name']
                    if metric_name in task_scores:
                        score = float(task_scores[metric_name])
                        # Ensure score is within 0.0-1.0 range
                        normalized_scores[task_id][metric_name] = max(0.0, min(1.0, score))
                    else:
                        logger.warning(f"Missing score for {metric_name} in task {task_id}")
                        normalized_scores[task_id][metric_name] = 0.0
            
            return normalized_scores
        except Exception as e:
            logger.error(f"Judge {self.name} failed to score outputs: {e}")
            raise

class JudgeManager:
    """Manages multiple judges and coordinates their evaluations."""
    
    def __init__(self, judges_config: List[Dict[str, Any]], cache: Optional[ResponseCache] = None):
        self.judges = []
        for judge_config in judges_config:
            judge = Judge(judge_config['name'], judge_config, cache)
            self.judges.append(judge)
        
        logger.info(f"Initialized {len(self.judges)} judges: {[j.name for j in self.judges]}")
    
    def propose_metrics(self, tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Get metric proposals from all judges."""
        proposals = {}
        for judge in self.judges:
            try:
                metrics = judge.propose_metrics(tasks)
                proposals[judge.name] = metrics
                logger.info(f"Judge {judge.name} proposed {len(metrics)} metrics")
            except Exception as e:
                logger.error(f"Judge {judge.name} failed to propose metrics: {e}")
                proposals[judge.name] = []
        
        return proposals
    
    def score_outputs(self, tasks_outputs: Dict[str, Dict[str, Any]], metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get scores from all judges."""
        all_scores = {}
        for judge in self.judges:
            try:
                scores = judge.score_outputs(tasks_outputs, metrics)
                all_scores[judge.name] = scores
                logger.info(f"Judge {judge.name} scored {len(scores)} tasks")
            except Exception as e:
                logger.error(f"Judge {judge.name} failed to score outputs: {e}")
                all_scores[judge.name] = {}
        
        return all_scores
    
    def get_judge_names(self) -> List[str]:
        """Get list of judge names."""
        return [judge.name for judge in self.judges] 