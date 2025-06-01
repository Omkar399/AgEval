#!/usr/bin/env python3
"""
Demo script to show confidence-based scoring with varied agent responses
"""

import json
from src.judge import Judge
from src.utils import ResponseCache

# Sample tasks with varied response quality
test_cases = [
    {
        "task_id": "math_test",
        "prompt": "Compute 47 × 382 + 129.",
        "responses": {
            "excellent": "Let me calculate this step by step:\n47 × 382 = 17,954\n17,954 + 129 = 18,083\nTherefore, the answer is 18,083.",
            "good": "47 × 382 + 129 = 18083",
            "poor": "I think it's around 18000 or so",
            "terrible": "Math is hard. Maybe 5000?"
        }
    }
]

# Initialize a judge
judge_config = {
    'provider': 'google',
    'model': 'gemini-2.0-flash-lite', 
    'api_key': 'AIzaSyA_-qYhkttlbhbzU_0-c9Ck5LqJRDdFx0k',
    'temperature': 0
}

print("🔬 Testing Confidence-Based Scoring System")
print("=" * 50)

judge = Judge("TestJudge", judge_config, ResponseCache())

# Define confidence-based metrics
metrics = [
    {
        "name": "Response Accuracy",
        "definition": "Confidence in the factual correctness and logical soundness of the agent's response, considering accuracy of information, calculations, and reasoning.",
        "scale": "Confidence [0.0-1.0]"
    },
    {
        "name": "Response Clarity",
        "definition": "Confidence in how clear, well-structured, and easy to understand the response is. Higher scores for responses with logical flow and good organization.",
        "scale": "Confidence [0.0-1.0]"
    }
]

for test_case in test_cases:
    print(f"\n📝 Task: {test_case['prompt']}")
    print("-" * 30)
    
    for quality, response in test_case['responses'].items():
        print(f"\n🤖 {quality.title()} Response: {response}")
        
        # Create task output format
        task_outputs = {
            test_case['task_id']: {
                'prompt': test_case['prompt'],
                'response': response
            }
        }
        
        try:
            # Get judge scores
            scores = judge.score_outputs(task_outputs, metrics)
            task_scores = scores[test_case['task_id']]
            
            print(f"⚖️ Judge Confidence Scores:")
            for metric_name, score in task_scores.items():
                confidence_level = (
                    "High Confidence ✅" if score >= 0.8 else
                    "Moderate Confidence ⚠️" if score >= 0.6 else  
                    "Low Confidence ❌" if score >= 0.4 else
                    "Very Low Confidence 🔴"
                )
                print(f"   • {metric_name}: {score:.2f} ({confidence_level})")
                
        except Exception as e:
            print(f"❌ Error: {e}")

print("\n" + "=" * 50)
print("✅ Confidence scoring test complete!") 