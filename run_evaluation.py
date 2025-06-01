#!/usr/bin/env python3
"""
Main script to run the three-judge evaluation pipeline.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import setup_logging
from src.pipeline import EvaluationPipeline

def main():
    parser = argparse.ArgumentParser(description="Run the three-judge AI evaluation system")
    parser.add_argument("--config", default="config/judges_config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--tasks", default="data/tasks.json",
                       help="Path to tasks file")
    parser.add_argument("--anchors", default="data/anchors.json", 
                       help="Path to anchor tasks file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", default="logs/evaluation.log",
                       help="Log file path")
    parser.add_argument("--phase", type=int, choices=range(1, 9),
                       help="Run only a specific phase (1-8)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    try:
        # Initialize pipeline
        pipeline = EvaluationPipeline(args.config)
        
        if args.phase:
            # Run specific phase
            print(f"Running Phase {args.phase}...")
            
            if args.phase == 1:
                pipeline.phase_1_task_suite(args.tasks)
            elif args.phase == 2:
                pipeline.phase_2_configure_judges()
            elif args.phase == 3:
                pipeline.load_tasks(args.tasks)
                pipeline.phase_3_metric_proposal()
            elif args.phase == 4:
                from src.utils import load_json
                proposals = load_json("data/metric_proposals.json")
                pipeline.phase_4_metric_consolidation(proposals)
            elif args.phase == 5:
                pipeline.load_tasks(args.tasks)
                pipeline.phase_5_generate_outputs()
            elif args.phase == 6:
                pipeline.load_tasks(args.tasks)
                from src.utils import load_json
                pipeline.canonical_metrics = load_json("data/canonical_metrics.json")
                pipeline.agent_outputs = load_json("data/agent_outputs.json")
                pipeline.phase_6_scoring()
            elif args.phase == 7:
                from src.utils import load_json
                pipeline.canonical_metrics = load_json("data/canonical_metrics.json")
                pipeline.raw_scores = load_json("data/raw_scores.json")
                pipeline.anchors = load_json(args.anchors)
                try:
                    pipeline.anchor_scores = load_json("data/anchor_scores.json")
                except FileNotFoundError:
                    print("Warning: No anchor scores found, skipping bias calibration")
                pipeline.phase_7_calibration()
            elif args.phase == 8:
                from src.utils import load_json
                pipeline.canonical_metrics = load_json("data/canonical_metrics.json")
                pipeline.calibrated_scores = load_json("data/calibrated_scores.json")
                pipeline.tasks = load_json(args.tasks)
                pipeline.phase_8_aggregation()
            
            print(f"Phase {args.phase} completed successfully!")
            
        else:
            # Run full evaluation
            print("Starting full three-judge evaluation...")
            report = pipeline.run_full_evaluation(args.tasks, args.anchors)
            
            print("\n" + "="*60)
            print("EVALUATION COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Evaluation ID: {report['evaluation_id']}")
            print(f"Duration: {report['duration_seconds']:.1f} seconds")
            print(f"Agent: {report['agent_info']['name']} ({report['agent_info']['model']})")
            print(f"Tasks: {report['num_tasks']}")
            print(f"Metrics: {report['num_metrics']}")
            print(f"Judges: {', '.join(report['judge_names'])}")
            
            print("\nFinal Performance:")
            for metric_name, score in report['final_performance'].items():
                print(f"  {metric_name}: {score:.3f}")
            
            print(f"\nDetailed results saved to: data/")
            print(f"Evaluation report: data/evaluation_report.json")
            print(f"Performance summary: data/performance_summary.json")
            
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        print("Make sure you have:")
        print("1. Copied config/judges_config.yaml.example to config/judges_config.yaml")
        print("2. Added your API keys to the configuration file")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 