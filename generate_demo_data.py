#!/usr/bin/env python3
"""
Generate demo data for FastAPI dashboard testing
"""
import json
import random
import numpy as np
from datetime import datetime, timedelta
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def generate_agent_trajectory(agent_id, num_tasks=8):
    """Generate a realistic agent trajectory"""
    # Start with random initial ability
    initial_ability = random.uniform(-2, 2)
    trajectory = []
    ability = initial_ability
    
    for i in range(num_tasks):
        # Generate task difficulty based on current ability estimate
        if i == 0:
            difficulty = 0.3  # Start easy
        else:
            # Adaptive difficulty selection
            difficulty = max(0.1, min(0.9, ability / 3.0 + random.uniform(-0.2, 0.2)))
        
        # Determine performance based on ability and difficulty
        probability = 1 / (1 + np.exp(-(ability - difficulty)))
        outcome = bool(random.random() < probability)
        
        # Update ability estimate (simplified IRT)
        if outcome:
            ability += 0.3 * (difficulty - ability + 1) / (i + 2)
        else:
            ability -= 0.3 * (ability - difficulty + 1) / (i + 2)
        
        # Calculate uncertainty (decreases over time)
        uncertainty = max(0.1, 2.0 / np.sqrt(i + 1))
        
        trajectory.append({
            "step": i + 1,
            "task_id": f"adaptive_task_{i+1}_{difficulty:.2f}",
            "task_difficulty": round(difficulty, 3),
            "ability_estimate": round(ability, 3),
            "uncertainty": round(uncertainty, 3),
            "outcome": outcome,
            "response_time": random.uniform(5, 30)
        })
    
    # Determine convergence
    final_uncertainty = trajectory[-1]["uncertainty"]
    converged = final_uncertainty < 0.3
    convergence_step = None
    
    if converged:
        # Find when convergence was achieved
        for i, step in enumerate(trajectory):
            if step["uncertainty"] < 0.3:
                convergence_step = i + 1
                break
    
    return {
        "trajectory": trajectory,
        "final_ability": round(ability, 3),
        "confidence_interval": [
            round(ability - 1.96 * final_uncertainty, 3),
            round(ability + 1.96 * final_uncertainty, 3)
        ],
        "converged": converged,
        "convergence_step": convergence_step,
        "total_tasks": len(trajectory)
    }

def generate_static_performance(agent_id, num_tasks=15):
    """Generate static evaluation performance"""
    # Random overall ability
    ability = random.uniform(-1.5, 2.5)
    
    # Generate task outcomes
    tasks_completed = 0
    tasks_passed = 0
    category_scores = {}
    
    categories = ["atomic", "compositional", "end2end"]
    
    for category in categories:
        cat_tasks = random.randint(3, 6)
        cat_passed = 0
        
        for _ in range(cat_tasks):
            # Slightly different difficulty for different categories
            if category == "atomic":
                difficulty = 0.3
            elif category == "compositional":
                difficulty = 0.6
            else:  # end2end
                difficulty = 0.8
            
            probability = 1 / (1 + np.exp(-(ability - difficulty)))
            if bool(random.random() < probability):
                cat_passed += 1
            
            tasks_completed += 1
        
        category_scores[category] = {
            "passed": cat_passed,
            "total": cat_tasks,
            "score": round(cat_passed / cat_tasks, 3)
        }
        tasks_passed += cat_passed
    
    overall_score = round(tasks_passed / tasks_completed, 3)
    
    return {
        "overall_performance": {
            "score": overall_score,
            "total_tasks": tasks_completed,
            "passed_tasks": tasks_passed
        },
        "category_performance": category_scores
    }

def generate_adaptive_results():
    """Generate comprehensive adaptive evaluation results"""
    agents = [
        "GPT-4-Turbo", "Claude-3-Opus", "Gemini-Pro", "Llama-3-70B", "Mixtral-8x7B",
        "CodeLlama-34B", "PaLM-2", "GPT-3.5-Turbo", "Claude-3-Sonnet", "Vicuna-33B"
    ]
    
    agent_results = {}
    
    for agent_id in agents:
        # Generate adaptive trajectory
        num_tasks = random.randint(5, 12)  # Adaptive uses fewer tasks
        agent_results[agent_id] = generate_agent_trajectory(agent_id, num_tasks)
    
    # Calculate summary statistics
    total_evaluations = len(agent_results)
    avg_tasks = np.mean([r["total_tasks"] for r in agent_results.values()])
    convergence_rate = sum(1 for r in agent_results.values() if r["converged"]) / total_evaluations * 100
    
    return {
        "metadata": {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_agents": total_evaluations,
            "average_tasks_per_agent": round(avg_tasks, 1),
            "convergence_rate": round(convergence_rate, 1)
        },
        "agent_results": agent_results
    }

def generate_comprehensive_analysis():
    """Generate comprehensive analysis matching adaptive agents"""
    agents = [
        "GPT-4-Turbo", "Claude-3-Opus", "Gemini-Pro", "Llama-3-70B", "Mixtral-8x7B",
        "CodeLlama-34B", "PaLM-2", "GPT-3.5-Turbo", "Claude-3-Sonnet", "Vicuna-33B"
    ]
    
    agent_data = {}
    
    for agent_id in agents:
        agent_data[agent_id] = generate_static_performance(agent_id)
    
    return {
        "evaluation_metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(agents),
            "evaluation_type": "comprehensive_static"
        },
        "agents": agent_data
    }

def generate_tasks():
    """Generate task definitions"""
    tasks = []
    
    # Atomic tasks
    for i in range(3):
        tasks.append({
            "id": f"atomic_{i+1}",
            "tier": "atomic",
            "prompt": f"Solve this mathematical reasoning problem step by step: Problem {i+1}",
            "gold_answer": f"Solution for atomic problem {i+1}",
            "description": f"Mathematical reasoning task {i+1}",
            "difficulty": round(0.2 + i * 0.1, 1)
        })
    
    # Compositional tasks
    for i in range(3):
        tasks.append({
            "id": f"compositional_{i+1}",
            "tier": "compositional",
            "prompt": f"Design and implement a complex system that: Requirements {i+1}",
            "gold_answer": f"Implementation for compositional problem {i+1}",
            "description": f"System design task {i+1}",
            "difficulty": round(0.5 + i * 0.1, 1)
        })
    
    # End-to-end tasks
    for i in range(3):
        tasks.append({
            "id": f"end2end_{i+1}",
            "tier": "end2end", 
            "prompt": f"Complete end-to-end project: Project {i+1}",
            "gold_answer": f"Full solution for end2end problem {i+1}",
            "description": f"Complete project task {i+1}",
            "difficulty": round(0.7 + i * 0.1, 1)
        })
    
    return tasks

def generate_all_demo_data():
    """Generate all demo data files"""
    
    print("🎯 Generating demo data for FastAPI dashboard...")
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Generate adaptive evaluation results
    print("📊 Generating adaptive evaluation results...")
    adaptive_results = generate_adaptive_results()
    with open("data/adaptive_evaluation_results.json", "w") as f:
        json.dump(adaptive_results, f, indent=2, cls=NumpyEncoder)
    
    # Generate comprehensive analysis
    print("📈 Generating comprehensive analysis...")
    comp_analysis = generate_comprehensive_analysis()
    with open("data/comprehensive_analysis.json", "w") as f:
        json.dump(comp_analysis, f, indent=2, cls=NumpyEncoder)
    
    # Generate tasks
    print("📝 Generating tasks...")
    tasks = generate_tasks()
    with open("data/tasks.json", "w") as f:
        json.dump(tasks, f, indent=2, cls=NumpyEncoder)
    
    # Generate some additional files
    print("📁 Generating additional data files...")
    
    # Detailed adaptive results
    detailed_adaptive = {
        "evaluation_details": adaptive_results,
        "statistical_analysis": {
            "efficiency_metrics": {
                "average_task_reduction": 67.3,
                "time_saved_percent": 58.9,
                "convergence_reliability": 89.2
            }
        }
    }
    with open("data/detailed_adaptive_results.json", "w") as f:
        json.dump(detailed_adaptive, f, indent=2, cls=NumpyEncoder)
    
    # Performance summary
    performance_summary = {
        "summary_statistics": {
            "total_evaluations": len(adaptive_results["agent_results"]),
            "avg_performance": 0.73,
            "top_performer": max(adaptive_results["agent_results"].keys(), 
                               key=lambda x: adaptive_results["agent_results"][x]["final_ability"]),
            "evaluation_efficiency": 68.5
        }
    }
    with open("data/performance_summary.json", "w") as f:
        json.dump(performance_summary, f, indent=2, cls=NumpyEncoder)
    
    # Empty files for missing data types
    empty_files = [
        "enhanced_results.json",
        "failure_adjusted_scores.json", 
        "calibrated_scores.json",
        "raw_scores.json",
        "final_performance.json",
        "static_evaluation_results.json",
        "adaptive_comprehensive_analysis.json",
        "adaptive_base_tasks.json"
    ]
    
    for filename in empty_files:
        filepath = f"data/{filename}"
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                json.dump({}, f)
    
    print("✅ Demo data generation complete!")
    print(f"📊 Generated data for {len(adaptive_results['agent_results'])} agents")
    print(f"📈 Adaptive evaluation shows {adaptive_results['metadata']['convergence_rate']}% convergence rate")
    print(f"⚡ Average {adaptive_results['metadata']['average_tasks_per_agent']} tasks per agent")
    print("\n🚀 You can now run the FastAPI dashboard with realistic demo data!")

if __name__ == "__main__":
    generate_all_demo_data()