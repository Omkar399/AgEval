"""
Dynamic Adaptive Evaluation with Difficulty Calibration for AgEval.

This module implements adaptive testing using Item Response Theory (IRT) to 
dynamically adjust task difficulty based on agent performance, providing
more precise ability estimation with fewer tasks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import random
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

class DifficultyLevel(Enum):
    """Standardized difficulty levels."""
    VERY_EASY = 0.2
    EASY = 0.35
    MEDIUM = 0.5
    HARD = 0.65
    VERY_HARD = 0.8

class TaskDomain(Enum):
    """Task domain categories for targeted difficulty scaling."""
    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    ANALYTICAL = "analytical"

@dataclass
class IRTParameters:
    """Item Response Theory parameters for a task."""
    discrimination: float = 1.0  # How well task distinguishes between abilities (a-parameter)
    difficulty: float = 0.0      # Task difficulty on logit scale (b-parameter)  
    guessing: float = 0.0        # Probability of correct guess (c-parameter)
    domain: TaskDomain = TaskDomain.ANALYTICAL
    
    def probability_correct(self, ability: float) -> float:
        """Calculate probability of correct response given ability."""
        return self.guessing + (1 - self.guessing) / (
            1 + np.exp(-self.discrimination * (ability - self.difficulty))
        )

@dataclass
class AdaptiveTask:
    """An adaptive task with difficulty parameters."""
    task_id: str
    base_prompt: str
    domain: TaskDomain
    irt_params: IRTParameters
    difficulty_variants: Dict[str, str] = field(default_factory=dict)
    expected_tokens: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def generate_at_difficulty(self, target_difficulty: float) -> str:
        """Generate task prompt at specified difficulty level."""
        # Find closest predefined difficulty variant
        if not self.difficulty_variants:
            return self.base_prompt
            
        closest_difficulty = min(
            self.difficulty_variants.keys(),
            key=lambda x: abs(float(x) - target_difficulty)
        )
        
        return self.difficulty_variants[closest_difficulty]

@dataclass
class AdaptiveResponse:
    """Response from adaptive evaluation."""
    task_id: str
    response: str
    performance_score: float
    difficulty_attempted: float
    evolved_prompt: str = ""  # Add field for the actual evolved prompt text
    base_prompt: str = ""     # Add field for the original base prompt
    time_taken: Optional[float] = None
    reasoning_steps: Optional[List[str]] = None

@dataclass
class AbilityEstimate:
    """Current estimate of agent ability."""
    ability: float = 0.0
    standard_error: float = 1.0
    confidence_interval: Tuple[float, float] = (-1.96, 1.96)
    num_items: int = 0
    
    def is_converged(self, threshold: float = 0.3) -> bool:
        """Check if ability estimate has converged."""
        return self.standard_error < threshold and self.num_items >= 5

class IRTDifficultyEstimator:
    """
    Item Response Theory model for estimating agent ability and task difficulty.
    
    Uses 3-parameter logistic model:
    P(correct|ability, task) = c + (1-c) / (1 + exp(-a*(ability - b)))
    
    Where:
    - a = discrimination parameter (how well task distinguishes ability levels)
    - b = difficulty parameter (ability level where P(correct) = 0.5 + c/2)  
    - c = guessing parameter (probability of correct answer by chance)
    """
    
    def __init__(self, initial_ability: float = 0.0, initial_se: float = 1.0):
        """
        Initialize IRT estimator.
        
        Args:
            initial_ability: Starting ability estimate (logit scale)
            initial_se: Initial standard error of ability estimate
        """
        self.ability_estimate = AbilityEstimate(
            ability=initial_ability,
            standard_error=initial_se
        )
        self.response_history: List[Tuple[IRTParameters, float, float]] = []
        
    def update_ability(self, task_params: IRTParameters, performance: float) -> float:
        """
        Update ability estimate using Maximum Likelihood Estimation.
        
        Args:
            task_params: IRT parameters for the task
            performance: Performance score (0.0 to 1.0)
            
        Returns:
            Updated ability estimate
        """
        # Record response
        self.response_history.append((task_params, performance, self.ability_estimate.ability))
        
        # Use Newton-Raphson method for MLE
        current_ability = self.ability_estimate.ability
        
        for iteration in range(10):  # Max 10 iterations
            # Calculate likelihood and derivatives
            log_likelihood, first_deriv, second_deriv = self._calculate_derivatives(
                current_ability, self.response_history
            )
            
            # Newton-Raphson update
            if abs(second_deriv) > 1e-10:
                ability_update = -first_deriv / second_deriv
                current_ability += ability_update
                
                # Check for convergence
                if abs(ability_update) < 1e-6:
                    break
            else:
                break
        
        # Update standard error using Fisher Information
        fisher_info = -second_deriv if second_deriv < 0 else 1.0
        standard_error = 1.0 / np.sqrt(fisher_info) if fisher_info > 0 else 1.0
        
        # Update ability estimate
        self.ability_estimate.ability = current_ability
        self.ability_estimate.standard_error = standard_error
        self.ability_estimate.num_items += 1
        
        # Update confidence interval
        margin = 1.96 * standard_error
        self.ability_estimate.confidence_interval = (
            current_ability - margin,
            current_ability + margin
        )
        
        logger.info(f"Updated ability: {current_ability:.3f} ± {standard_error:.3f}")
        return current_ability
    
    def select_next_difficulty(self, available_difficulties: List[float]) -> float:
        """
        Select optimal difficulty for next task using Maximum Information criterion.
        
        Args:
            available_difficulties: List of available difficulty levels
            
        Returns:
            Optimal difficulty level
        """
        current_ability = self.ability_estimate.ability
        
        # For maximum information, select difficulty closest to current ability
        # This is where the IRT curve has steepest slope
        optimal_difficulty = min(
            available_difficulties,
            key=lambda d: abs(d - current_ability)
        )
        
        logger.info(f"Selected difficulty {optimal_difficulty:.3f} for ability {current_ability:.3f}")
        return optimal_difficulty
    
    def _calculate_derivatives(self, 
                             ability: float, 
                             responses: List[Tuple[IRTParameters, float, float]]) -> Tuple[float, float, float]:
        """Calculate log-likelihood and its first two derivatives."""
        log_likelihood = 0.0
        first_deriv = 0.0
        second_deriv = 0.0
        
        for task_params, performance, _ in responses:
            # Probability of correct response
            p_correct = task_params.probability_correct(ability)
            
            # Avoid log(0) and division by 0
            p_correct = np.clip(p_correct, 1e-10, 1 - 1e-10)
            
            # Log-likelihood contribution
            if performance > 0.5:  # Treat as correct if performance > 0.5
                log_likelihood += np.log(p_correct)
            else:
                log_likelihood += np.log(1 - p_correct)
            
            # First derivative (score function)
            a, b, c = task_params.discrimination, task_params.difficulty, task_params.guessing
            exp_term = np.exp(-a * (ability - b))
            
            if performance > 0.5:
                first_deriv += a * (1 - c) * exp_term / (
                    (c + (1 - c) / (1 + exp_term)) * (1 + exp_term)
                )
            else:
                first_deriv -= a * (1 - c) * exp_term / (
                    (1 - c + (c - 1) / (1 + exp_term)) * (1 + exp_term)
                )
            
            # Second derivative (Fisher Information)
            # Simplified approximation
            p = task_params.probability_correct(ability)
            info = a**2 * p * (1 - p) * ((1 - c) / (1 - c * p))**2
            second_deriv -= info
        
        return log_likelihood, first_deriv, second_deriv

class DynamicTaskGenerator:
    """
    Generates tasks at specified difficulty levels by modifying base tasks.
    
    Uses various techniques to scale difficulty:
    - Complexity scaling (more steps, constraints)
    - Context scaling (more distractors, ambiguity)
    - Knowledge scaling (deeper domain knowledge required)
    """
    
    def __init__(self):
        """Initialize task generator with difficulty scaling templates."""
        self.scaling_templates = {
            TaskDomain.MATHEMATICAL: {
                DifficultyLevel.VERY_EASY: "Simple single-step: {base}",
                DifficultyLevel.EASY: "Two-step problem: {base}",
                DifficultyLevel.MEDIUM: "Multi-step with one constraint: {base}",
                DifficultyLevel.HARD: "Multi-step with multiple constraints: {base}",
                DifficultyLevel.VERY_HARD: "Complex optimization problem: {base}"
            },
            TaskDomain.LOGICAL: {
                DifficultyLevel.VERY_EASY: "Basic logical statement: {base}",
                DifficultyLevel.EASY: "Simple deduction: {base}",
                DifficultyLevel.MEDIUM: "Multiple premises: {base}",
                DifficultyLevel.HARD: "Complex reasoning chain: {base}",
                DifficultyLevel.VERY_HARD: "Abstract logical proof: {base}"
            },
            TaskDomain.CREATIVE: {
                DifficultyLevel.VERY_EASY: "Simple creative task: {base}",
                DifficultyLevel.EASY: "Creative with one constraint: {base}",
                DifficultyLevel.MEDIUM: "Creative with multiple constraints: {base}",
                DifficultyLevel.HARD: "Creative with abstract requirements: {base}",
                DifficultyLevel.VERY_HARD: "Highly constrained creative challenge: {base}"
            },
            TaskDomain.TECHNICAL: {
                DifficultyLevel.VERY_EASY: "Basic technical concept: {base}",
                DifficultyLevel.EASY: "Simple technical application: {base}",
                DifficultyLevel.MEDIUM: "Multi-component technical problem: {base}",
                DifficultyLevel.HARD: "Complex system design: {base}",
                DifficultyLevel.VERY_HARD: "Advanced technical architecture: {base}"
            },
            TaskDomain.ANALYTICAL: {
                DifficultyLevel.VERY_EASY: "Simple analysis: {base}",
                DifficultyLevel.EASY: "Basic comparison: {base}",
                DifficultyLevel.MEDIUM: "Multi-factor analysis: {base}",
                DifficultyLevel.HARD: "Complex synthesis: {base}",
                DifficultyLevel.VERY_HARD: "Abstract analytical framework: {base}"
            }
        }
    
    def generate_adaptive_task(self, 
                              base_task: Dict[str, Any], 
                              target_difficulty: float,
                              domain: TaskDomain = None) -> AdaptiveTask:
        """
        Generate an adaptive task at specified difficulty level.
        
        Args:
            base_task: Base task definition
            target_difficulty: Target difficulty (0.0 to 1.0)
            domain: Task domain for difficulty scaling (auto-detected if None)
            
        Returns:
            AdaptiveTask with difficulty variants
        """
        # Auto-detect domain if not provided
        if domain is None:
            domain = self._detect_task_domain(base_task)
        
        # Map target difficulty to discrete levels
        difficulty_level = self._map_to_difficulty_level(target_difficulty)
        
        # Generate IRT parameters
        irt_params = IRTParameters(
            discrimination=np.random.normal(1.2, 0.3),  # Typical discrimination
            difficulty=self._difficulty_to_logit(target_difficulty),
            guessing=0.1 if domain in [TaskDomain.MATHEMATICAL, TaskDomain.LOGICAL] else 0.0,
            domain=domain
        )
        
        # Create difficulty variants
        difficulty_variants = {}
        base_prompt = base_task.get('prompt', '')
        
        for level in DifficultyLevel:
            template = self.scaling_templates.get(domain, {}).get(
                level, "At difficulty {level.value}: {base}"
            )
            
            # Apply sophisticated difficulty scaling
            scaled_prompt = self._apply_difficulty_scaling(
                base_prompt, level, domain
            )
            
            difficulty_variants[str(level.value)] = scaled_prompt
        
        return AdaptiveTask(
            task_id=f"adaptive_{base_task.get('id', 'unknown')}_{target_difficulty:.2f}",
            base_prompt=base_prompt,
            domain=domain,
            irt_params=irt_params,
            difficulty_variants=difficulty_variants,
            expected_tokens=self._estimate_tokens(difficulty_level),
            metadata={
                'base_task_id': base_task.get('id'),
                'target_difficulty': target_difficulty,
                'detected_domain': domain.value,
                'generated_at': pd.Timestamp.now().isoformat()
            }
        )
    
    def _detect_task_domain(self, base_task: Dict[str, Any]) -> TaskDomain:
        """Automatically detect the most appropriate domain for a task."""
        prompt = base_task.get('prompt', '').lower()
        task_id = base_task.get('id', '').lower()
        
        # Mathematical keywords
        math_keywords = [
            'compute', 'calculate', 'solve', 'equation', 'arithmetic', 'formula',
            'mathematical', 'number', 'addition', 'subtraction', 'multiplication',
            'division', 'algebra', 'geometry', 'statistics', 'probability'
        ]
        
        # Technical keywords  
        tech_keywords = [
            'code', 'programming', 'algorithm', 'function', 'class', 'implementation',
            'software', 'system', 'api', 'database', 'architecture', 'framework',
            'debug', 'optimize', 'deploy', 'technical', 'development'
        ]
        
        # Logical reasoning keywords
        logic_keywords = [
            'logic', 'reasoning', 'deduction', 'induction', 'proof', 'conclusion',
            'premise', 'argument', 'valid', 'invalid', 'syllogism', 'inference',
            'if then', 'therefore', 'because', 'logical'
        ]
        
        # Creative keywords
        creative_keywords = [
            'creative', 'design', 'brainstorm', 'innovative', 'original', 'artistic',
            'story', 'narrative', 'imagine', 'invent', 'create', 'generate ideas',
            'novel', 'unique', 'inspiration'
        ]
        
        # Count keyword matches
        math_score = sum(1 for keyword in math_keywords if keyword in prompt or keyword in task_id)
        tech_score = sum(1 for keyword in tech_keywords if keyword in prompt or keyword in task_id)
        logic_score = sum(1 for keyword in logic_keywords if keyword in prompt or keyword in task_id)
        creative_score = sum(1 for keyword in creative_keywords if keyword in prompt or keyword in task_id)
        
        # Find the domain with highest score
        scores = {
            TaskDomain.MATHEMATICAL: math_score,
            TaskDomain.TECHNICAL: tech_score,
            TaskDomain.LOGICAL: logic_score,
            TaskDomain.CREATIVE: creative_score
        }
        
        best_domain = max(scores, key=scores.get)
        
        # If no clear winner, default to analytical
        if scores[best_domain] == 0:
            return TaskDomain.ANALYTICAL
            
        return best_domain
    
    def _map_to_difficulty_level(self, difficulty: float) -> DifficultyLevel:
        """Map continuous difficulty to discrete level."""
        if difficulty <= 0.27:
            return DifficultyLevel.VERY_EASY
        elif difficulty <= 0.42:
            return DifficultyLevel.EASY
        elif difficulty <= 0.57:
            return DifficultyLevel.MEDIUM
        elif difficulty <= 0.72:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.VERY_HARD
    
    def _difficulty_to_logit(self, difficulty: float) -> float:
        """Convert difficulty (0-1) to logit scale."""
        # Map 0.5 difficulty to 0 logit (50% success rate)
        # Map 0.2 difficulty to -2 logit (easy)
        # Map 0.8 difficulty to +2 logit (hard)
        return 5 * (difficulty - 0.5)
    
    def _apply_difficulty_scaling(self, 
                                base_prompt: str, 
                                level: DifficultyLevel, 
                                domain: TaskDomain) -> str:
        """Apply judge-based difficulty scaling to prompt."""
        
        # Map difficulty levels to descriptive terms
        difficulty_descriptions = {
            DifficultyLevel.VERY_EASY: "much easier",
            DifficultyLevel.EASY: "easier", 
            DifficultyLevel.MEDIUM: "similar difficulty",
            DifficultyLevel.HARD: "harder",
            DifficultyLevel.VERY_HARD: "much harder"
        }
        
        if level == DifficultyLevel.MEDIUM:
            # For medium difficulty, just return the base prompt
            return base_prompt
        
        difficulty_desc = difficulty_descriptions[level]
        
        # Create a judge prompt to modify the task difficulty
        judge_prompt = f"""You are a task difficulty expert. Given this task:

"{base_prompt}"

Create a {difficulty_desc} version of the same fundamental task. Keep the core objective the same, but adjust the complexity, constraints, or requirements to make it {difficulty_desc}.

Guidelines:
- For easier tasks: Simplify requirements, reduce constraints, provide more guidance
- For harder tasks: Add complexity, multiple constraints, require deeper analysis, or increase scope
- Maintain the same general domain and task type
- Make the difficulty change feel natural and appropriate

Return only the modified task prompt, nothing else."""

        # In a real implementation, you would call your judge system here
        # For now, we'll use a simplified approach
        return self._simulate_judge_response(base_prompt, level, judge_prompt)
    
    def _simulate_judge_response(self, base_prompt: str, level: DifficultyLevel, judge_prompt: str) -> str:
        """
        Simulate judge response for difficulty scaling.
        In production, this would call your actual judge system.
        """
        # This is a simplified simulation - in production you'd call your judge
        
        if "plan" in base_prompt.lower() and "itinerary" in base_prompt.lower():
            # Travel planning example
            if level == DifficultyLevel.VERY_EASY:
                return f"Plan a simple 1-day visit to San Francisco. List 2-3 main attractions and suggest lunch. Keep it basic."
            elif level == DifficultyLevel.EASY:
                return f"Plan a 2-day itinerary for visiting San Francisco. Include 3-4 attractions per day and meal suggestions."
            elif level == DifficultyLevel.HARD:
                return f"Plan a comprehensive 3-day San Francisco itinerary for a family with different age groups. Include attractions, restaurants, transportation, budget breakdown, backup plans for weather, and accessibility considerations."
            elif level == DifficultyLevel.VERY_HARD:
                return f"Design a detailed 3-day San Francisco itinerary optimizing for budget, time, and group preferences. Include real-time transportation options, reservation requirements, seasonal considerations, alternative routes, cost-benefit analysis of different approaches, and contingency plans for various scenarios."
        
        elif any(op in base_prompt.lower() for op in ['compute', 'calculate', '+', '×', 'multiply']):
            # Math example  
            if level == DifficultyLevel.VERY_EASY:
                return f"Solve: {base_prompt} Show your work."
            elif level == DifficultyLevel.EASY:
                return f"Calculate: {base_prompt} Show each step and verify your answer."
            elif level == DifficultyLevel.HARD:
                return f"Solve: {base_prompt} Show multiple solution methods, identify the most efficient approach, and explain why."
            elif level == DifficultyLevel.VERY_HARD:
                return f"Mathematical problem: {base_prompt} Solve using at least two different methods, analyze computational complexity, identify underlying mathematical principles, and create a general algorithm for similar problems."
        
        else:
            # Generic scaling
            complexity_modifiers = {
                DifficultyLevel.VERY_EASY: f"Simple version: {base_prompt} Keep your response straightforward.",
                DifficultyLevel.EASY: f"Basic task: {base_prompt} Provide clear reasoning for your approach.",
                DifficultyLevel.HARD: f"Complex challenge: {base_prompt} Consider multiple perspectives, analyze trade-offs, and provide detailed justification for your approach.",
                DifficultyLevel.VERY_HARD: f"Advanced analysis: {base_prompt} Provide comprehensive analysis, consider edge cases, examine underlying assumptions, and develop systematic frameworks for evaluation."
            }
            
            return complexity_modifiers.get(level, base_prompt)
    
    def _estimate_tokens(self, level: DifficultyLevel) -> int:
        """Estimate expected response tokens based on difficulty."""
        token_estimates = {
            DifficultyLevel.VERY_EASY: 50,
            DifficultyLevel.EASY: 100,
            DifficultyLevel.MEDIUM: 200,
            DifficultyLevel.HARD: 350,
            DifficultyLevel.VERY_HARD: 500
        }
        return token_estimates.get(level, 200)

class AdaptiveEvaluationPipeline:
    """
    Main pipeline for adaptive evaluation using IRT-based difficulty calibration.
    
    Implements Computer Adaptive Testing (CAT) principles:
    1. Start with medium difficulty
    2. Adjust difficulty based on performance
    3. Stop when ability estimate converges
    4. Use maximum information criterion for task selection
    """
    
    def __init__(self, 
                 initial_ability: float = 0.0,
                 convergence_threshold: float = 0.3,
                 max_items: int = 15,
                 min_items: int = 5):
        """
        Initialize adaptive evaluation pipeline.
        
        Args:
            initial_ability: Starting ability estimate
            convergence_threshold: SE threshold for convergence
            max_items: Maximum number of tasks to administer
            min_items: Minimum number of tasks before stopping
        """
        self.irt_estimator = IRTDifficultyEstimator(initial_ability)
        self.task_generator = DynamicTaskGenerator()
        self.convergence_threshold = convergence_threshold
        self.max_items = max_items
        self.min_items = min_items
        
        # Track evaluation session
        self.session_responses: List[AdaptiveResponse] = []
        self.session_metadata = {
            'start_time': pd.Timestamp.now(),
            'convergence_achieved': False,
            'total_items': 0
        }
        
    def run_adaptive_evaluation(self, 
                              agent,
                              base_tasks: List[Dict[str, Any]],
                              domain: TaskDomain = TaskDomain.ANALYTICAL) -> Dict[str, Any]:
        """
        Run complete adaptive evaluation session.
        
        Args:
            agent: Agent to evaluate
            base_tasks: Pool of base tasks to adapt
            domain: Primary domain for this evaluation
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Starting adaptive evaluation with {len(base_tasks)} base tasks")
        
        available_difficulties = [level.value for level in DifficultyLevel]
        
        for item_num in range(self.max_items):
            # Check convergence
            if (item_num >= self.min_items and 
                self.irt_estimator.ability_estimate.is_converged(self.convergence_threshold)):
                self.session_metadata['convergence_achieved'] = True
                logger.info(f"Converged after {item_num} items")
                break
            
            # Select optimal difficulty for next task
            target_difficulty = self.irt_estimator.select_next_difficulty(
                available_difficulties
            )
            
            # Select base task (could be random or strategic)
            base_task = random.choice(base_tasks)
            
            # Generate adaptive task at target difficulty
            adaptive_task = self.task_generator.generate_adaptive_task(
                base_task, target_difficulty  # Remove hardcoded domain, let it auto-detect
            )
            
            # Get agent response
            response = self._get_agent_response(agent, adaptive_task, target_difficulty)
            
            # Update ability estimate
            self.irt_estimator.update_ability(
                adaptive_task.irt_params, 
                response.performance_score
            )
            
            # Record response
            self.session_responses.append(response)
            
            logger.info(
                f"Item {item_num + 1}: Difficulty {target_difficulty:.2f}, "
                f"Performance {response.performance_score:.2f}, "
                f"Ability {self.irt_estimator.ability_estimate.ability:.2f}"
            )
        
        # Finalize session
        self.session_metadata.update({
            'end_time': pd.Timestamp.now(),
            'total_items': len(self.session_responses),
            'final_ability': self.irt_estimator.ability_estimate.ability,
            'final_se': self.irt_estimator.ability_estimate.standard_error
        })
        
        return self._generate_evaluation_report()
    
    def _get_agent_response(self, 
                          agent, 
                          adaptive_task: AdaptiveTask, 
                          difficulty: float) -> AdaptiveResponse:
        """Get and score agent response to adaptive task."""
        
        # Generate task prompt at target difficulty
        task_prompt = adaptive_task.generate_at_difficulty(difficulty)
        
        # Get agent response (integrate with your existing agent interface)
        start_time = pd.Timestamp.now()
        
        try:
            # This should integrate with your existing agent.generate_response method
            agent_response = agent.generate_response(task_prompt, adaptive_task.task_id)
            response_text = agent_response.get('response', '')
        except Exception as e:
            logger.error(f"Agent response failed: {e}")
            response_text = ""
        
        end_time = pd.Timestamp.now()
        
        # Score the response (simplified - you can integrate with your judge system)
        performance_score = self._score_response(response_text, adaptive_task)
        
        return AdaptiveResponse(
            task_id=adaptive_task.task_id,
            response=response_text,
            performance_score=performance_score,
            difficulty_attempted=difficulty,
            evolved_prompt=task_prompt,  # Save the actual evolved prompt that was used
            base_prompt=adaptive_task.base_prompt,  # Save the original base prompt
            time_taken=(end_time - start_time).total_seconds(),
            reasoning_steps=self._extract_reasoning_steps(response_text)
        )
    
    def _score_response(self, response: str, task: AdaptiveTask) -> float:
        """
        Score agent response (simplified version).
        
        In production, this should integrate with your judge system.
        """
        if not response or len(response.strip()) < 10:
            return 0.1  # Very low score for empty/minimal responses
        
        # Simplified scoring based on response length and keywords
        # In production, use your judge system here
        base_score = min(len(response) / 200, 1.0)  # Length-based component
        
        # Adjust based on task difficulty
        difficulty_adjustment = 1.0  # Could be more sophisticated
        
        final_score = base_score * difficulty_adjustment
        return np.clip(final_score, 0.0, 1.0)
    
    def _extract_reasoning_steps(self, response: str) -> List[str]:
        """Extract reasoning steps from response (simplified)."""
        # Look for common reasoning indicators
        reasoning_indicators = [
            "First,", "Second,", "Then,", "Next,", "Finally,",
            "Step 1:", "Step 2:", "Step 3:",
            "Because", "Therefore", "Since", "Given that"
        ]
        
        steps = []
        sentences = response.split('.')
        
        for sentence in sentences:
            if any(indicator in sentence for indicator in reasoning_indicators):
                steps.append(sentence.strip())
        
        return steps[:5]  # Limit to 5 steps
    
    def _generate_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        
        ability_estimate = self.irt_estimator.ability_estimate
        
        # Performance trajectory
        difficulties = [r.difficulty_attempted for r in self.session_responses]
        performances = [r.performance_score for r in self.session_responses]
        
        # Calculate summary statistics
        avg_performance = np.mean(performances)
        performance_consistency = 1.0 - np.std(performances)
        difficulty_range = max(difficulties) - min(difficulties)
        
        # Ability interpretation
        ability_percentile = norm.cdf(ability_estimate.ability) * 100
        
        return {
            'adaptive_evaluation_results': {
                'final_ability_estimate': ability_estimate.ability,
                'ability_standard_error': ability_estimate.standard_error,
                'ability_confidence_interval': ability_estimate.confidence_interval,
                'ability_percentile': ability_percentile,
                'convergence_achieved': self.session_metadata['convergence_achieved'],
                'total_items_administered': self.session_metadata['total_items']
            },
            'performance_analysis': {
                'average_performance': avg_performance,
                'performance_consistency': performance_consistency,
                'difficulty_range_explored': difficulty_range,
                'performance_trajectory': performances,
                'difficulty_trajectory': difficulties
            },
            'session_metadata': self.session_metadata,
            'detailed_responses': [
                {
                    'task_id': r.task_id,
                    'difficulty': r.difficulty_attempted,
                    'performance': r.performance_score,
                    'response_length': len(r.response),
                    'reasoning_steps': len(r.reasoning_steps or []),
                    'time_taken': r.time_taken,
                    'evolved_prompt': r.evolved_prompt,  # Include the actual evolved prompt
                    'base_prompt': r.base_prompt,        # Include the original base prompt
                    'agent_response': r.response         # Include the full agent response
                }
                for r in self.session_responses
            ],
            'irt_response_history': [
                {
                    'discrimination': params.discrimination,
                    'difficulty': params.difficulty,
                    'performance': perf,
                    'ability_at_time': ability
                }
                for params, perf, ability in self.irt_estimator.response_history
            ]
        }
    
    def plot_evaluation_trajectory(self, save_path: Optional[str] = None) -> None:
        """Plot the evaluation trajectory showing difficulty and performance over time."""
        
        if not self.session_responses:
            logger.warning("No responses to plot")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        items = range(1, len(self.session_responses) + 1)
        difficulties = [r.difficulty_attempted for r in self.session_responses]
        performances = [r.performance_score for r in self.session_responses]
        
        # Ability estimates over time
        abilities = [entry[2] for entry in self.irt_estimator.response_history]
        
        # Plot 1: Difficulty trajectory
        ax1.plot(items, difficulties, 'b-o', label='Task Difficulty')
        ax1.set_ylabel('Difficulty Level')
        ax1.set_title('Adaptive Evaluation Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Performance trajectory
        ax2.plot(items, performances, 'r-o', label='Performance Score')
        ax2.set_ylabel('Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Ability estimate trajectory
        ax3.plot(items, abilities, 'g-o', label='Ability Estimate')
        ax3.axhline(y=self.irt_estimator.ability_estimate.ability, 
                   color='g', linestyle='--', alpha=0.7, label='Final Estimate')
        ax3.fill_between(items, 
                        [a - self.irt_estimator.ability_estimate.standard_error for a in abilities],
                        [a + self.irt_estimator.ability_estimate.standard_error for a in abilities],
                        alpha=0.2, color='g', label='±1 SE')
        ax3.set_xlabel('Item Number')
        ax3.set_ylabel('Ability Estimate (logit)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trajectory plot saved to {save_path}")
        
        plt.show()

def create_adaptive_benchmark_from_existing(
    existing_tasks: List[Dict[str, Any]], 
    domains: Optional[List[TaskDomain]] = None) -> List[AdaptiveTask]:
    """
    Convert existing AgEval tasks to adaptive tasks with IRT parameters.
    
    Args:
        existing_tasks: List of existing task definitions
        domains: List of domains to assign to tasks
        
    Returns:
        List of AdaptiveTask objects
    """
    if domains is None:
        domains = list(TaskDomain)
    
    generator = DynamicTaskGenerator()
    adaptive_tasks = []
    
    for i, task in enumerate(existing_tasks):
        # Assign domain cyclically or based on task content
        domain = domains[i % len(domains)]
        
        # Create adaptive task with medium difficulty as baseline
        adaptive_task = generator.generate_adaptive_task(
            task, target_difficulty=0.5, domain=domain
        )
        
        adaptive_tasks.append(adaptive_task)
        
    logger.info(f"Created {len(adaptive_tasks)} adaptive tasks")
    return adaptive_tasks

# Example usage integration with AgEval
def integrate_with_ageval(agent, tasks_path: str = "data/tasks.json") -> Dict[str, Any]:
    """
    Integration function to run adaptive evaluation with existing AgEval setup.
    
    Args:
        agent: AgEval agent instance
        tasks_path: Path to existing tasks file
        
    Returns:
        Adaptive evaluation results
    """
    # Load existing tasks
    with open(tasks_path, 'r') as f:
        existing_tasks = json.load(f)
    
    # Create adaptive benchmark
    adaptive_tasks = create_adaptive_benchmark_from_existing(existing_tasks)
    
    # Run adaptive evaluation
    pipeline = AdaptiveEvaluationPipeline(
        initial_ability=0.0,  # Start with neutral ability
        convergence_threshold=0.3,  # Standard threshold
        max_items=12,  # Reasonable maximum
        min_items=5   # Minimum for reliability
    )
    
    # Convert adaptive tasks back to basic format for compatibility
    base_tasks = [
        {
            'id': task.task_id,
            'prompt': task.base_prompt,
            'domain': task.domain.value
        }
        for task in adaptive_tasks
    ]
    
    results = pipeline.run_adaptive_evaluation(
        agent, base_tasks, TaskDomain.ANALYTICAL
    )
    
    # Plot trajectory
    pipeline.plot_evaluation_trajectory("reports/adaptive_trajectory.png")
    
    return results 

def generate_evolved_prompt_example(base_prompt: str, difficulty: float, reasoning_steps: int, response_length: int) -> str:
    """
    Generate a realistic evolved prompt based on difficulty level and characteristics.
    
    Args:
        base_prompt: Original base prompt
        difficulty: Target difficulty (0.0 to 1.0)  
        reasoning_steps: Expected reasoning steps
        response_length: Expected response length
        
    Returns:
        Evolved prompt text at the specified difficulty level
    """
    
    # Difficulty level modifications
    if difficulty <= 0.3:  # Easy
        complexity_modifier = "Simple version: "
        constraints = ""
        additional_requirements = ""
    elif difficulty <= 0.5:  # Medium
        complexity_modifier = "Standard version: "
        constraints = " Include your reasoning steps."
        additional_requirements = ""
    elif difficulty <= 0.7:  # Hard
        complexity_modifier = "Complex version: "
        constraints = " Show all reasoning steps and validate your approach."
        additional_requirements = " Consider edge cases and alternative approaches."
    else:  # Very Hard
        complexity_modifier = "Advanced optimization version: "
        constraints = " Provide detailed step-by-step reasoning, validate all assumptions, and optimize your solution."
        additional_requirements = " Consider multiple solution paths, analyze trade-offs, and justify your final approach."
    
    # Apply difficulty-based modifications
    evolved_prompt = f"{complexity_modifier}{base_prompt}{constraints}{additional_requirements}"
    
    # Add reasoning requirements based on expected steps
    if reasoning_steps > 2:
        evolved_prompt += f" Structure your response with clear logical steps (aim for {reasoning_steps} distinct reasoning phases)."
    
    # Add length/detail requirements based on expected response length
    if response_length > 1000:
        evolved_prompt += " Provide a comprehensive and detailed response."
    elif response_length > 500:
        evolved_prompt += " Provide a thorough response with sufficient detail."
    
    return evolved_prompt 