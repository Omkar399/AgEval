#!/usr/bin/env python3
"""
Quick test of the new Adaptive Evaluation system.

This script runs a focused test of the adaptive evaluation to verify it's working correctly.
"""

import os
import sys
import logging
from datetime import datetime
from src.enhanced_pipeline import EnhancedEvaluationPipeline
from src.adaptive_evaluation import TaskDomain
from src.utils import setup_logging

def quick_adaptive_test():
    """Run a quick test of adaptive evaluation."""
    print("=" * 60)
    print("🎯 QUICK ADAPTIVE EVALUATION TEST")
    print("=" * 60)
    print()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize pipeline
        print("🔧 Initializing Adaptive Evaluation Pipeline...")
        pipeline = EnhancedEvaluationPipeline("config/judges_config.yaml")
        
        # Verify adaptive pipeline is enabled
        if not pipeline.adaptive_pipeline:
            print("❌ Adaptive pipeline not initialized!")
            return False
        
        print("✅ Adaptive pipeline initialized successfully")
        print(f"   • Max items: {pipeline.adaptive_pipeline.max_items}")
        print(f"   • Min items: {pipeline.adaptive_pipeline.min_items}")
        print(f"   • Convergence threshold: {pipeline.adaptive_pipeline.convergence_threshold}")
        
        # Run adaptive evaluation
        print("\n🚀 Running Adaptive Evaluation...")
        
        results = pipeline.run_enhanced_evaluation(
            tasks_path="data/tasks.json",
            anchors_path="data/anchors.json",
            enable_self_eval=False,  # Disable for speed
            enable_reliability=False,  # Disable for speed
            enable_adaptive=True,
            adaptive_domain=TaskDomain.ANALYTICAL
        )
        
        # Check results
        if 'adaptive_evaluation_results' in results:
            adaptive_results = results['adaptive_evaluation_results']['adaptive_evaluation_results']
            
            print("\n✅ ADAPTIVE EVALUATION COMPLETED!")
            print(f"   • Final Ability: {adaptive_results['final_ability_estimate']:.3f}")
            print(f"   • Percentile: {adaptive_results['ability_percentile']:.1f}%")
            print(f"   • Standard Error: ±{adaptive_results['ability_standard_error']:.3f}")
            print(f"   • Items Used: {adaptive_results['total_items_administered']}")
            print(f"   • Converged: {'Yes' if adaptive_results['convergence_achieved'] else 'No'}")
            
            # Check efficiency
            efficiency = results.get('evaluation_efficiency', 0)
            print(f"   • Efficiency: {efficiency:.3f}")
            
            # Check if trajectory plot was created
            if os.path.exists("reports/adaptive_evaluation_trajectory.png"):
                print("   • 📊 Trajectory plot created!")
            
            return True
        else:
            print("❌ No adaptive results found!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_adaptive_test()
    print("\n" + "=" * 60)
    if success:
        print("🎉 ADAPTIVE EVALUATION TEST PASSED!")
        print("   Your AgEval system is now using dynamic difficulty calibration!")
    else:
        print("❌ TEST FAILED - Check the error messages above")
    print("=" * 60)
    
    sys.exit(0 if success else 1) 