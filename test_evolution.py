#!/usr/bin/env python3
"""Test script for sophisticated prompt evolution system."""

from src.adaptive_evaluation import DynamicTaskGenerator, TaskDomain
import json

def test_prompt_evolution():
    """Test the new sophisticated prompt evolution."""
    generator = DynamicTaskGenerator()

    # Test mathematical task
    math_task = {'id': 'test_math', 'prompt': 'Compute 47 × 382 + 129.'}
    adaptive_math = generator.generate_adaptive_task(math_task, 0.8)
    print('=== MATHEMATICAL TASK EVOLUTION ===')
    print(f'Detected Domain: {adaptive_math.domain.value}')
    print(f'Base: {adaptive_math.base_prompt}')
    print(f'Very Easy: {adaptive_math.difficulty_variants.get("0.2", "N/A")}')
    print(f'Very Hard: {adaptive_math.difficulty_variants.get("0.8", "N/A")}')
    print()

    # Test technical task  
    tech_task = {'id': 'test_tech', 'prompt': 'Write a function to implement a binary search algorithm.'}
    adaptive_tech = generator.generate_adaptive_task(tech_task, 0.8)
    print('=== TECHNICAL TASK EVOLUTION ===')
    print(f'Detected Domain: {adaptive_tech.domain.value}')
    print(f'Base: {adaptive_tech.base_prompt}')
    print(f'Very Easy: {adaptive_tech.difficulty_variants.get("0.2", "N/A")}')
    print(f'Very Hard: {adaptive_tech.difficulty_variants.get("0.8", "N/A")}')
    print()

    # Test logical task
    logic_task = {'id': 'test_logic', 'prompt': 'If all cats are mammals and some mammals are dogs, what can we conclude?'}
    adaptive_logic = generator.generate_adaptive_task(logic_task, 0.8)
    print('=== LOGICAL TASK EVOLUTION ===')
    print(f'Detected Domain: {adaptive_logic.domain.value}')
    print(f'Base: {adaptive_logic.base_prompt}')
    print(f'Very Easy: {adaptive_logic.difficulty_variants.get("0.2", "N/A")}')
    print(f'Very Hard: {adaptive_logic.difficulty_variants.get("0.8", "N/A")}')
    print()

    # Test creative task
    creative_task = {'id': 'test_creative', 'prompt': 'Design a creative solution for reducing food waste in restaurants.'}
    adaptive_creative = generator.generate_adaptive_task(creative_task, 0.8)
    print('=== CREATIVE TASK EVOLUTION ===')
    print(f'Detected Domain: {adaptive_creative.domain.value}')
    print(f'Base: {adaptive_creative.base_prompt}')
    print(f'Very Easy: {adaptive_creative.difficulty_variants.get("0.2", "N/A")}')
    print(f'Very Hard: {adaptive_creative.difficulty_variants.get("0.8", "N/A")}')
    print()

    # Test analytical task (default)
    analytical_task = {'id': 'test_analytical', 'prompt': 'Analyze the pros and cons of remote work.'}
    adaptive_analytical = generator.generate_adaptive_task(analytical_task, 0.8)
    print('=== ANALYTICAL TASK EVOLUTION ===')
    print(f'Detected Domain: {adaptive_analytical.domain.value}')
    print(f'Base: {adaptive_analytical.base_prompt}')
    print(f'Very Easy: {adaptive_analytical.difficulty_variants.get("0.2", "N/A")}')
    print(f'Very Hard: {adaptive_analytical.difficulty_variants.get("0.8", "N/A")}')

if __name__ == "__main__":
    test_prompt_evolution() 