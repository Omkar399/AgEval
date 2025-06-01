"""
Hard Thinking: Multi-LLM Ensemble System
Implementation of the advanced reasoning framework with task decomposition,
parallel model execution, and intelligent response synthesis.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class EnsembleStrategy(Enum):
    VOTING = "voting"
    WEIGHTED = "weighted"
    BEST = "best"
    CONSENSUS = "consensus"

@dataclass
class Subtask:
    id: int
    task: str
    complexity: TaskComplexity
    context: Optional[str] = None
    dependencies: List[int] = None

@dataclass
class ModelResponse:
    model_name: str
    subtask_id: int
    response: str
    confidence: float
    tokens: int
    latency: float
    metadata: Dict[str, Any] = None

@dataclass
class ResponseScore:
    confidence_score: float
    consistency_score: float
    learned_weight: float
    final_score: float

@dataclass
class EnsembleResult:
    query: str
    best_model: str
    final_answer: str
    confidence_score: float
    consensus_level: float
    processing_time: float
    total_tokens: int
    model_breakdown: Dict[str, Any]
    decomposition: List[Subtask]
    strategy_used: EnsembleStrategy

class TaskDecomposer:
    """Breaks complex tasks into manageable subtasks"""
    
    def __init__(self):
        self.decomposition_patterns = {
            TaskComplexity.SIMPLE: self._simple_decomposition,
            TaskComplexity.MODERATE: self._moderate_decomposition,
            TaskComplexity.COMPLEX: self._complex_decomposition
        }
    
    def decompose_task(self, query: str, complexity: TaskComplexity, problem_type: str = "general") -> List[Subtask]:
        """Decompose a task based on complexity level"""
        decompose_func = self.decomposition_patterns[complexity]
        return decompose_func(query, problem_type)
    
    def _simple_decomposition(self, query: str, problem_type: str) -> List[Subtask]:
        """Simple tasks don't need decomposition"""
        return [Subtask(id=1, task=query, complexity=TaskComplexity.SIMPLE)]
    
    def _moderate_decomposition(self, query: str, problem_type: str) -> List[Subtask]:
        """Break into 2-4 logical steps"""
        if problem_type == "math":
            return [
                Subtask(1, f"Understand the mathematical problem: {query}", TaskComplexity.MODERATE),
                Subtask(2, "Identify the appropriate mathematical approach and formulas", TaskComplexity.MODERATE),
                Subtask(3, "Solve step by step with calculations", TaskComplexity.MODERATE),
                Subtask(4, "Verify the solution and provide final answer", TaskComplexity.MODERATE)
            ]
        elif problem_type == "code":
            return [
                Subtask(1, f"Analyze the coding requirement: {query}", TaskComplexity.MODERATE),
                Subtask(2, "Design the algorithm and data structures", TaskComplexity.MODERATE),
                Subtask(3, "Implement the solution with proper coding practices", TaskComplexity.MODERATE)
            ]
        else:
            return [
                Subtask(1, f"Analyze the problem: {query}", TaskComplexity.MODERATE),
                Subtask(2, "Gather relevant information and context", TaskComplexity.MODERATE),
                Subtask(3, "Synthesize the solution", TaskComplexity.MODERATE)
            ]
    
    def _complex_decomposition(self, query: str, problem_type: str) -> List[Subtask]:
        """Hierarchical breakdown for complex problems"""
        base_subtasks = [
            Subtask(1, f"Break down the complex problem: {query}", TaskComplexity.COMPLEX),
            Subtask(2, "Research and gather comprehensive information", TaskComplexity.COMPLEX),
            Subtask(3, "Apply multiple reasoning frameworks", TaskComplexity.COMPLEX),
            Subtask(4, "Cross-validate the approach from different angles", TaskComplexity.COMPLEX),
            Subtask(5, "Integrate findings into a comprehensive solution", TaskComplexity.COMPLEX)
        ]
        
        # Add problem-type specific subtasks
        if problem_type == "analysis":
            base_subtasks.extend([
                Subtask(6, "Perform statistical or quantitative analysis", TaskComplexity.COMPLEX),
                Subtask(7, "Identify patterns and insights", TaskComplexity.COMPLEX)
            ])
        elif problem_type == "reasoning":
            base_subtasks.extend([
                Subtask(6, "Apply logical deduction and inference", TaskComplexity.COMPLEX),
                Subtask(7, "Check for logical consistency and validity", TaskComplexity.COMPLEX)
            ])
        
        return base_subtasks

class LLMEnsemble:
    """Manages parallel execution across multiple LLMs"""
    
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.models = {
            "GPT-4": self._query_openai,
            "Claude": self._query_anthropic,
            "Gemini": self._query_google
        }
        self.model_weights = {
            "GPT-4": {"math": 0.9, "code": 0.85, "reasoning": 0.9, "general": 0.85},
            "Claude": {"math": 0.85, "code": 0.9, "reasoning": 0.85, "general": 0.9},
            "Gemini": {"math": 0.8, "code": 0.8, "reasoning": 0.8, "general": 0.8}
        }
    
    async def process_subtask(self, subtask: Subtask, problem_type: str, attempts: int = 3) -> List[ModelResponse]:
        """Process a subtask with all available models"""
        tasks = []
        for model_name, query_func in self.models.items():
            tasks.append(self._execute_with_retries(query_func, model_name, subtask, attempts))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid responses
        valid_responses = [r for r in responses if isinstance(r, ModelResponse)]
        return valid_responses
    
    async def _execute_with_retries(self, query_func, model_name: str, subtask: Subtask, attempts: int) -> ModelResponse:
        """Execute query with retry logic"""
        for attempt in range(attempts):
            try:
                start_time = time.time()
                response_text, confidence, tokens = await query_func(subtask.task)
                latency = time.time() - start_time
                
                return ModelResponse(
                    model_name=model_name,
                    subtask_id=subtask.id,
                    response=response_text,
                    confidence=confidence,
                    tokens=tokens,
                    latency=latency
                )
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {model_name}: {e}")
                if attempt == attempts - 1:
                    # Return a fallback response on final failure
                    return ModelResponse(
                        model_name=model_name,
                        subtask_id=subtask.id,
                        response=f"Model {model_name} unavailable",
                        confidence=0.0,
                        tokens=0,
                        latency=0.0
                    )
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def _query_openai(self, prompt: str) -> Tuple[str, float, int]:
        """Query OpenAI GPT-4 (placeholder implementation)"""
        # Simulate API call
        await asyncio.sleep(0.5 + np.random.uniform(0, 1))
        response = f"GPT-4 response to: {prompt[:100]}..."
        confidence = 0.8 + np.random.uniform(0, 0.2)
        tokens = len(prompt.split()) * 2 + np.random.randint(50, 200)
        return response, confidence, tokens
    
    async def _query_anthropic(self, prompt: str) -> Tuple[str, float, int]:
        """Query Anthropic Claude (placeholder implementation)"""
        # Simulate API call
        await asyncio.sleep(0.7 + np.random.uniform(0, 1))
        response = f"Claude response to: {prompt[:100]}..."
        confidence = 0.75 + np.random.uniform(0, 0.25)
        tokens = len(prompt.split()) * 2 + np.random.randint(40, 180)
        return response, confidence, tokens
    
    async def _query_google(self, prompt: str) -> Tuple[str, float, int]:
        """Query Google Gemini (placeholder implementation)"""
        # Simulate API call
        await asyncio.sleep(0.4 + np.random.uniform(0, 0.8))
        response = f"Gemini response to: {prompt[:100]}..."
        confidence = 0.7 + np.random.uniform(0, 0.3)
        tokens = len(prompt.split()) * 2 + np.random.randint(30, 150)
        return response, confidence, tokens

class ResponseScorer:
    """Evaluates and scores model responses"""
    
    def __init__(self, problem_type: str):
        self.problem_type = problem_type
        self.consistency_threshold = 0.7
    
    def score_responses(self, responses: List[ModelResponse], strategy: EnsembleStrategy) -> Dict[str, ResponseScore]:
        """Score all responses from different models"""
        scores = {}
        
        for response in responses:
            scores[response.model_name] = self._calculate_score(response, responses, strategy)
        
        return scores
    
    def _calculate_score(self, response: ModelResponse, all_responses: List[ModelResponse], strategy: EnsembleStrategy) -> ResponseScore:
        """Calculate comprehensive score for a single response"""
        confidence_score = response.confidence
        consistency_score = self._calculate_consistency(response, all_responses)
        learned_weight = self._get_learned_weight(response.model_name)
        
        # Calculate final score based on strategy
        if strategy == EnsembleStrategy.WEIGHTED:
            final_score = (
                0.4 * confidence_score +
                0.3 * consistency_score +
                0.3 * learned_weight
            )
        elif strategy == EnsembleStrategy.CONSENSUS:
            final_score = (
                0.2 * confidence_score +
                0.6 * consistency_score +
                0.2 * learned_weight
            )
        else:
            final_score = confidence_score
        
        return ResponseScore(
            confidence_score=confidence_score,
            consistency_score=consistency_score,
            learned_weight=learned_weight,
            final_score=final_score
        )
    
    def _calculate_consistency(self, response: ModelResponse, all_responses: List[ModelResponse]) -> float:
        """Calculate consistency with other model responses"""
        if len(all_responses) <= 1:
            return 0.8  # Default consistency for single response
        
        # Simplified consistency calculation (in practice, would use semantic similarity)
        same_subtask_responses = [r for r in all_responses if r.subtask_id == response.subtask_id]
        if len(same_subtask_responses) <= 1:
            return 0.8
        
        # Simulate consistency scoring based on response length similarity
        response_lengths = [len(r.response) for r in same_subtask_responses]
        avg_length = np.mean(response_lengths)
        length_variance = np.var(response_lengths) / (avg_length + 1)
        
        # Lower variance = higher consistency
        consistency = max(0.3, 1.0 - length_variance / 1000)
        return min(1.0, consistency)
    
    def _get_learned_weight(self, model_name: str) -> float:
        """Get learned weight for the model based on historical performance"""
        # In practice, this would be learned from historical data
        weights = {
            "GPT-4": 0.85,
            "Claude": 0.88,
            "Gemini": 0.75
        }
        return weights.get(model_name, 0.7)

class HarderThinkingSystem:
    """Main orchestrator for the multi-LLM ensemble system"""
    
    def __init__(self, api_keys: Dict[str, str]):
        self.decomposer = TaskDecomposer()
        self.ensemble = LLMEnsemble(api_keys)
        self.start_time = None
    
    async def process_query(self, 
                          query: str, 
                          problem_type: str = "general",
                          complexity: TaskComplexity = TaskComplexity.MODERATE,
                          strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED) -> EnsembleResult:
        """Process a query through the complete hard thinking pipeline"""
        
        self.start_time = time.time()
        
        # Step 1: Decompose the task
        subtasks = self.decomposer.decompose_task(query, complexity, problem_type)
        
        # Step 2: Process each subtask with ensemble
        all_responses = []
        for subtask in subtasks:
            responses = await self.ensemble.process_subtask(subtask, problem_type)
            all_responses.extend(responses)
        
        # Step 3: Score responses
        scorer = ResponseScorer(problem_type)
        scores = scorer.score_responses(all_responses, strategy)
        
        # Step 4: Integrate results
        final_result = self._integrate_results(
            query, subtasks, all_responses, scores, strategy
        )
        
        return final_result
    
    def _integrate_results(self, 
                          query: str,
                          subtasks: List[Subtask],
                          responses: List[ModelResponse],
                          scores: Dict[str, ResponseScore],
                          strategy: EnsembleStrategy) -> EnsembleResult:
        """Integrate all results into final answer"""
        
        # Find best model overall
        best_model = max(scores.keys(), key=lambda m: scores[m].final_score)
        
        # Calculate consensus level
        consensus = np.mean([score.final_score for score in scores.values()])
        
        # Calculate total tokens
        total_tokens = sum(r.tokens for r in responses)
        
        # Generate final answer
        final_answer = self._generate_final_answer(query, best_model, responses, scores, strategy)
        
        # Prepare model breakdown
        model_breakdown = {}
        for model_name in scores.keys():
            model_responses = [r for r in responses if r.model_name == model_name]
            model_breakdown[model_name] = {
                "responses": [{"response": r.response, "confidence": r.confidence, "tokens": r.tokens} for r in model_responses],
                "overall_confidence": scores[model_name].confidence_score,
                "consistency_score": scores[model_name].consistency_score,
                "final_score": scores[model_name].final_score
            }
        
        return EnsembleResult(
            query=query,
            best_model=best_model,
            final_answer=final_answer,
            confidence_score=scores[best_model].final_score,
            consensus_level=consensus * 100,
            processing_time=time.time() - self.start_time,
            total_tokens=total_tokens,
            model_breakdown=model_breakdown,
            decomposition=subtasks,
            strategy_used=strategy
        )
    
    def _generate_final_answer(self, 
                              query: str,
                              best_model: str,
                              responses: List[ModelResponse],
                              scores: Dict[str, ResponseScore],
                              strategy: EnsembleStrategy) -> str:
        """Generate the final synthesized answer"""
        
        best_responses = [r for r in responses if r.model_name == best_model]
        confidence_pct = scores[best_model].final_score * 100
        consensus_pct = np.mean([score.final_score for score in scores.values()]) * 100
        
        # Create a comprehensive answer combining insights
        answer_parts = []
        answer_parts.append(f"Based on multi-LLM ensemble analysis using {strategy.value} strategy:")
        answer_parts.append(f"\nPrimary Answer (from {best_model} - {confidence_pct:.1f}% confidence):")
        
        if best_responses:
            # Use the first best response as the primary answer
            answer_parts.append(best_responses[0].response)
        
        answer_parts.append(f"\nEnsemble Consensus: {consensus_pct:.1f}%")
        answer_parts.append(f"Models consulted: {', '.join(scores.keys())}")
        
        # Add validation note
        if consensus_pct > 80:
            answer_parts.append("\n✅ High consensus achieved across models")
        elif consensus_pct > 60:
            answer_parts.append("\n⚠️  Moderate consensus - consider multiple perspectives")
        else:
            answer_parts.append("\n❌ Low consensus - high uncertainty in results")
        
        return "\n".join(answer_parts)

# Example usage and testing
async def main():
    """Example usage of the Hard Thinking system"""
    
    # Mock API keys (in practice, these would be real API keys)
    api_keys = {
        "openai": "mock_openai_key",
        "anthropic": "mock_anthropic_key",
        "google": "mock_google_key"
    }
    
    # Initialize the system
    hard_thinking = HarderThinkingSystem(api_keys)
    
    # Test query
    query = "What is the optimal solution for the traveling salesman problem with 20 cities?"
    
    # Process the query
    result = await hard_thinking.process_query(
        query=query,
        problem_type="reasoning",
        complexity=TaskComplexity.COMPLEX,
        strategy=EnsembleStrategy.WEIGHTED
    )
    
    # Print results
    print(f"Query: {result.query}")
    print(f"Best Model: {result.best_model}")
    print(f"Confidence: {result.confidence_score:.3f}")
    print(f"Consensus: {result.consensus_level:.1f}%")
    print(f"Processing Time: {result.processing_time:.2f}s")
    print(f"Total Tokens: {result.total_tokens}")
    print(f"\nFinal Answer:\n{result.final_answer}")

if __name__ == "__main__":
    asyncio.run(main())