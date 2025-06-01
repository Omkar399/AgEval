#!/usr/bin/env python3
"""
Demonstration of specialized agents with distinct personalities.
Each agent uses the same LLM but with specialized system prompts.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from src.specialized_agents import SpecializedAgentFactory
from src.utils import ResponseCache
from src.pipeline import EvaluationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demonstrate_specialized_agents():
    """Demonstrate how specialized agents work with different personalities."""
    print("🤖 AgEval Specialized Agents Demonstration")
    print("=" * 60)
    
    # Load configuration and tasks using pipeline
    pipeline = EvaluationPipeline('config/judges_config.yaml')
    tasks = pipeline.load_tasks('data/tasks.json')
    config = pipeline.config
    
    # Initialize cache and agent factory
    cache = ResponseCache('data/cache')
    agent_factory = SpecializedAgentFactory()
    
    # Agent configuration
    agent_config = config['agent']
    
    print(f"\n📋 Loaded {len(tasks)} tasks")
    print(f"🔧 Using {agent_config['model']} as base LLM")
    print(f"🎭 Creating {len(agent_factory.get_all_agent_types())} specialized agent personalities")
    
    # Demonstrate each specialized agent
    results = {}
    start_time = time.time()
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"Task {i}/{len(tasks)}: {task['id']}")
        print(f"{'='*60}")
        
        # Get specialized agent for this task
        specialized_agent = agent_factory.get_agent_for_task(task, agent_config)
        
        print(f"🎭 Agent: {specialized_agent.specialized_name}")
        print(f"🧠 Expertise: {', '.join(specialized_agent.expertise)}")
        print(f"💭 Personality: {specialized_agent.personality}")
        print(f"📝 Task: {task['prompt'][:100]}{'...' if len(task['prompt']) > 100 else ''}")
        
        # Generate response
        print(f"\n⏳ Generating response...")
        task_start = time.time()
        
        try:
            response_data = specialized_agent.generate_response(
                task['prompt'], 
                task['id'], 
                use_cache=config['optimization']['cache_responses']
            )
            
            task_time = time.time() - task_start
            
            # Store results
            results[task['id']] = {
                'task': task,
                'agent_info': specialized_agent.get_info(),
                'response': response_data,
                'execution_time': task_time
            }
            
            print(f"✅ Response generated in {task_time:.2f}s")
            print(f"📊 Response length: {len(response_data['response'])} characters")
            
            # Show a preview of the response
            response_preview = response_data['response'][:200]
            if len(response_data['response']) > 200:
                response_preview += "..."
            
            print(f"\n📄 Response Preview:")
            print(f"   {response_preview}")
            
            # Show agent-specific metadata
            if response_data.get('specialized'):
                print(f"\n🎯 Agent Specialization:")
                print(f"   Type: {response_data.get('agent_type', 'Unknown')}")
                print(f"   Expertise: {', '.join(response_data.get('expertise', []))}")
                print(f"   Personality: {response_data.get('personality', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to generate response for task {task['id']}: {e}")
            print(f"❌ Error: {e}")
            results[task['id']] = {
                'task': task,
                'agent_info': specialized_agent.get_info(),
                'error': str(e),
                'execution_time': time.time() - task_start
            }
    
    total_time = time.time() - start_time
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("📊 SPECIALIZED AGENTS SUMMARY")
    print(f"{'='*60}")
    
    successful_tasks = [r for r in results.values() if 'response' in r]
    failed_tasks = [r for r in results.values() if 'error' in r]
    
    print(f"✅ Successful tasks: {len(successful_tasks)}/{len(tasks)}")
    print(f"❌ Failed tasks: {len(failed_tasks)}")
    print(f"⏱️  Total execution time: {total_time:.2f}s")
    print(f"📈 Average time per task: {total_time/len(tasks):.2f}s")
    
    # Show agent type distribution
    print(f"\n🎭 Agent Type Distribution:")
    agent_types = {}
    for result in results.values():
        if 'agent_info' in result:
            agent_name = result['agent_info'].get('specialized_name', 'Unknown')
            agent_types[agent_name] = agent_types.get(agent_name, 0) + 1
    
    for agent_name, count in agent_types.items():
        print(f"   {agent_name}: {count} task(s)")
    
    # Show response characteristics by agent type
    print(f"\n📊 Response Characteristics by Agent:")
    for agent_name in agent_types.keys():
        agent_responses = [
            r['response']['response'] for r in results.values() 
            if 'response' in r and r['agent_info'].get('specialized_name') == agent_name
        ]
        
        if agent_responses:
            avg_length = sum(len(resp) for resp in agent_responses) / len(agent_responses)
            print(f"   {agent_name}: Avg {avg_length:.0f} chars per response")
    
    # Save detailed results
    output_file = f"data/specialized_agents_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Prepare results for JSON serialization
    json_results = {}
    for task_id, result in results.items():
        json_result = {
            'task': result['task'],
            'agent_info': result['agent_info'],
            'execution_time': result['execution_time']
        }
        
        if 'response' in result:
            json_result['response'] = result['response']
        if 'error' in result:
            json_result['error'] = result['error']
            
        json_results[task_id] = json_result
    
    # Add summary metadata
    summary_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_tasks': len(tasks),
            'successful_tasks': len(successful_tasks),
            'failed_tasks': len(failed_tasks),
            'total_execution_time': total_time,
            'average_time_per_task': total_time / len(tasks),
            'base_model': agent_config['model'],
            'agent_types_used': agent_types
        },
        'results': json_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Show example of personality differences
    print(f"\n🎭 PERSONALITY SHOWCASE")
    print(f"{'='*60}")
    
    # Find examples of different agent types
    example_agents = {}
    for result in results.values():
        if 'response' in result:
            agent_name = result['agent_info'].get('specialized_name', 'Unknown')
            if agent_name not in example_agents:
                example_agents[agent_name] = result
    
    for agent_name, result in list(example_agents.items())[:3]:  # Show first 3 examples
        print(f"\n🤖 {agent_name}")
        print(f"   Personality: {result['agent_info'].get('personality', 'Unknown')}")
        print(f"   Task: {result['task']['id']}")
        
        response_sample = result['response']['response'][:150]
        if len(result['response']['response']) > 150:
            response_sample += "..."
        print(f"   Response Style: {response_sample}")
    
    print(f"\n🎉 Specialized Agents Demonstration Complete!")
    print(f"   Each agent used the same LLM ({agent_config['model']}) but with")
    print(f"   distinct personalities, expertise, and behavioral patterns.")
    
    return results

if __name__ == "__main__":
    try:
        results = demonstrate_specialized_agents()
        print(f"\n✅ Demonstration completed successfully!")
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"❌ Error: {e}")
        exit(1) 