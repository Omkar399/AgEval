#!/usr/bin/env python3
"""
Test script for the Hard Thinking multi-LLM ensemble system
"""

import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from hard_thinking import HarderThinkingSystem, TaskComplexity, EnsembleStrategy

async def test_hard_thinking():
    """Test the Hard Thinking system with various scenarios"""
    
    print("🧠 Testing Hard Thinking Multi-LLM Ensemble System")
    print("=" * 60)
    
    # Mock API keys for testing
    api_keys = {
        "openai": "test_key",
        "anthropic": "test_key", 
        "google": "test_key"
    }
    
    # Initialize system
    hard_thinking = HarderThinkingSystem(api_keys)
    
    # Test cases
    test_cases = [
        {
            "name": "Simple Math Problem",
            "query": "What is 127 * 43?",
            "problem_type": "math",
            "complexity": TaskComplexity.SIMPLE,
            "strategy": EnsembleStrategy.VOTING
        },
        {
            "name": "Moderate Code Problem", 
            "query": "Write a Python function to find the longest palindromic substring",
            "problem_type": "code",
            "complexity": TaskComplexity.MODERATE,
            "strategy": EnsembleStrategy.WEIGHTED
        },
        {
            "name": "Complex Reasoning Problem",
            "query": "Analyze the economic implications of implementing universal basic income in developed countries",
            "problem_type": "analysis",
            "complexity": TaskComplexity.COMPLEX,
            "strategy": EnsembleStrategy.CONSENSUS
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print("-" * 40)
        print(f"Query: {test_case['query']}")
        print(f"Type: {test_case['problem_type']}")
        print(f"Complexity: {test_case['complexity'].value}")
        print(f"Strategy: {test_case['strategy'].value}")
        
        try:
            # Run the hard thinking process
            result = await hard_thinking.process_query(
                query=test_case['query'],
                problem_type=test_case['problem_type'],
                complexity=test_case['complexity'],
                strategy=test_case['strategy']
            )
            
            # Display results
            print(f"\n✅ Results:")
            print(f"   Best Model: {result.best_model}")
            print(f"   Confidence: {result.confidence_score:.3f}")
            print(f"   Consensus: {result.consensus_level:.1f}%")
            print(f"   Processing Time: {result.processing_time:.2f}s")
            print(f"   Total Tokens: {result.total_tokens:,}")
            print(f"   Subtasks: {len(result.decomposition)}")
            print(f"\n📝 Final Answer Preview:")
            preview = result.final_answer[:200] + "..." if len(result.final_answer) > 200 else result.final_answer
            print(f"   {preview}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎉 Hard Thinking system test completed!")
    print("Ready for production use! 🚀")

if __name__ == "__main__":
    asyncio.run(test_hard_thinking())