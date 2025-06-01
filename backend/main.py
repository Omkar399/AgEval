"""
FastAPI backend for AgEval Dashboard
"""

import sys
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Dict, List, Any, Optional
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import math

# Add parent directory to path for src imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Set up logging to capture background task errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(parent_dir / "backend.log"))
    ]
)

try:
    from src.pipeline import EvaluationPipeline
    from src.utils import load_json, save_json
    USING_REAL_PIPELINE = True
    print("✅ Successfully imported real EvaluationPipeline")
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Looking for src in: {parent_dir}")
    USING_REAL_PIPELINE = False
    # Fallback - create minimal versions if src not available
    class EvaluationPipeline:
        def run_full_evaluation(self):
            return {"status": "error", "message": "AgEval pipeline not available"}
    
    def load_json(path):
        import json
        with open(path, 'r') as f:
            return json.load(f)
    
    def save_json(data, path):
        import json
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

app = FastAPI(title="AgEval API", description="API for AgEval Dashboard", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for data caching
_cached_data = {}
_last_refresh = None

def clean_nan_values(obj):
    """Recursively replace NaN values with None in nested data structures."""
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif isinstance(obj, np.ndarray):
        return clean_nan_values(obj.tolist())
    elif hasattr(obj, 'item'):  # numpy scalar
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    else:
        return obj

def load_evaluation_data() -> Dict[str, Any]:
    """Load all evaluation data from files."""
    global _cached_data, _last_refresh
    
    # Check if we need to refresh (cache for 30 seconds)
    if _last_refresh and (datetime.now() - _last_refresh).seconds < 30:
        return _cached_data
    
    data = {}
    data_dir = Path(__file__).parent.parent / "data"
    
    try:
        # Core evaluation files
        if (data_dir / "evaluation_report.json").exists():
            data["evaluation_report"] = load_json(str(data_dir / "evaluation_report.json"))
        
        if (data_dir / "performance_summary.json").exists():
            data["performance_summary"] = load_json(str(data_dir / "performance_summary.json"))
        
        if (data_dir / "final_performance.json").exists():
            data["final_performance"] = load_json(str(data_dir / "final_performance.json"))
        
        if (data_dir / "calibration_report.json").exists():
            data["calibration_report"] = load_json(str(data_dir / "calibration_report.json"))
        
        if (data_dir / "aggregated_scores.json").exists():
            data["aggregated_scores"] = load_json(str(data_dir / "aggregated_scores.json"))
        
        if (data_dir / "canonical_metrics.json").exists():
            data["canonical_metrics"] = load_json(str(data_dir / "canonical_metrics.json"))
        
        if (data_dir / "tasks.json").exists():
            data["tasks"] = load_json(str(data_dir / "tasks.json"))
        
        # Adaptive evaluation files
        adaptive_files = [
            "adaptive_evaluation_results.json",
            "detailed_adaptive_results.json", 
            "adaptive_comprehensive_analysis.json",
            "adaptive_base_tasks.json"
        ]
        
        for file in adaptive_files:
            if (data_dir / file).exists():
                data[file.replace('.json', '')] = load_json(str(data_dir / file))
        
        _cached_data = data
        _last_refresh = datetime.now()
        
    except Exception as e:
        logging.error(f"Error loading evaluation data: {e}")
        
    return data

@app.get("/")
async def root():
    return {"message": "AgEval API is running"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/evaluation/overview")
async def get_evaluation_overview():
    """Get evaluation overview data."""
    data = load_evaluation_data()
    
    overview = {
        "has_data": bool(data),
        "available_reports": list(data.keys()),
        "last_updated": _last_refresh.isoformat() if _last_refresh else None
    }
    
    if "evaluation_report" in data:
        report = data["evaluation_report"]
        overview.update({
            "evaluation_id": report.get("evaluation_id"),
            "status": report.get("status"),
            "duration": report.get("duration_seconds"),
            "num_tasks": report.get("num_tasks"),
            "num_metrics": report.get("num_metrics"),
            "agent_info": report.get("agent_info")
        })
    
    return overview

@app.get("/api/evaluation/performance")
async def get_performance_data():
    """Get performance metrics and scores."""
    data = load_evaluation_data()
    
    if not data:
        raise HTTPException(status_code=404, detail="No evaluation data found")
    
    response = {}
    
    if "final_performance" in data:
        response["final_performance"] = data["final_performance"]
    
    if "performance_summary" in data:
        response["performance_summary"] = data["performance_summary"]
    
    if "aggregated_scores" in data:
        response["aggregated_scores"] = data["aggregated_scores"]
    
    if "canonical_metrics" in data:
        response["canonical_metrics"] = data["canonical_metrics"]
    
    return response

@app.get("/api/evaluation/calibration")
async def get_calibration_data():
    """Get calibration and reliability analysis data."""
    data = load_evaluation_data()
    
    if "calibration_report" not in data:
        raise HTTPException(status_code=404, detail="No calibration data found")
    
    return clean_nan_values(data["calibration_report"])

@app.get("/api/evaluation/tasks")
async def get_tasks_data():
    """Get tasks and their details."""
    data = load_evaluation_data()
    
    response = {}
    
    if "tasks" in data:
        response["tasks"] = data["tasks"]
    
    if "aggregated_scores" in data:
        # Add performance data per task
        scores = data["aggregated_scores"]
        task_performance = {}
        for task_id, metrics in scores.items():
            task_performance[task_id] = {
                "overall_score": np.mean(list(metrics.values())) if metrics else 0,
                "metrics": metrics
            }
        response["task_performance"] = task_performance
    
    return response

@app.get("/api/adaptive/overview")
async def get_adaptive_overview():
    """Get adaptive evaluation overview."""
    data = load_evaluation_data()
    
    adaptive_keys = [
        "adaptive_evaluation_results",
        "detailed_adaptive_results", 
        "adaptive_comprehensive_analysis",
        "adaptive_base_tasks"
    ]
    
    adaptive_data = {key: data.get(key) for key in adaptive_keys if key in data}
    
    if not adaptive_data:
        raise HTTPException(status_code=404, detail="No adaptive evaluation data found")
    
    overview = {
        "has_adaptive_data": True,
        "available_reports": list(adaptive_data.keys())
    }
    
    if "adaptive_evaluation_results" in adaptive_data:
        report = adaptive_data["adaptive_evaluation_results"]
        overview.update({
            "total_iterations": len(report.get("iterations", [])),
            "convergence_achieved": report.get("convergence_achieved", False),
            "final_accuracy": report.get("final_accuracy")
        })
    
    return {**overview, "data": adaptive_data}

@app.get("/api/charts/performance-radar")
async def get_performance_radar_data():
    """Get data for performance radar chart."""
    data = load_evaluation_data()
    
    if "final_performance" not in data or "canonical_metrics" not in data:
        raise HTTPException(status_code=404, detail="Performance data not available")
    
    performance = data["final_performance"]
    metrics = data["canonical_metrics"]
    
    chart_data = {
        "metrics": [metric["name"] for metric in metrics],
        "scores": [performance.get(metric["name"], 0) for metric in metrics],
        "descriptions": [metric.get("definition", "") for metric in metrics]
    }
    
    return chart_data

@app.get("/api/charts/calibration-analysis")
async def get_calibration_chart_data():
    """Get data for calibration analysis charts."""
    data = load_evaluation_data()
    
    if "calibration_report" not in data:
        raise HTTPException(status_code=404, detail="Calibration data not available")
    
    calibration = data["calibration_report"]
    
    return clean_nan_values({
        "bias_offsets": calibration.get("bias_offsets", {}),
        "agreement_analysis": calibration.get("agreement_analysis", {}),
        "reliability_metrics": calibration.get("reliability_metrics", {})
    })

@app.post("/api/evaluation/run")
async def run_evaluation(background_tasks: BackgroundTasks, config: Optional[Dict] = None):
    """Run a complete evaluation (standard + adaptive) in the background."""
    
    def run_pipeline():
        error_report = None
        # Change working directory to project root for correct relative paths
        original_cwd = os.getcwd()
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)
        
        try:
            logging.info(f"🚀 Starting evaluation pipeline - Using {'REAL' if USING_REAL_PIPELINE else 'DUMMY'} EvaluationPipeline")
            
            # Clear any existing cache to force fresh evaluation
            global _cached_data, _last_refresh
            _cached_data = {}
            _last_refresh = None
            
            # Run standard evaluation
            logging.info("📊 Starting standard evaluation...")
            config_path = str(Path(__file__).parent.parent / "config" / "judges_config.yaml")
            
            # Ensure absolute paths for data files
            data_dir = Path(__file__).parent.parent / "data"
            tasks_path = str(data_dir / "tasks.json")
            anchors_path = str(data_dir / "anchors.json")
            
            logging.info(f"📁 Using config: {config_path}")
            logging.info(f"📁 Using tasks: {tasks_path}")
            logging.info(f"📁 Using anchors: {anchors_path}")
            
            pipeline = EvaluationPipeline(config_path)
            result = pipeline.run_full_evaluation(tasks_path, anchors_path)
            logging.info(f"✅ Standard evaluation completed: {result.get('status', 'unknown')}")
            
            # Also try adaptive evaluation (but don't fail if it doesn't work)
            adaptive_success = False
            try:
                logging.info("🧠 Starting adaptive evaluation...")
                from src.enhanced_pipeline import EnhancedEvaluationPipeline
                from src.adaptive_evaluation import TaskDomain
                
                enhanced_pipeline = EnhancedEvaluationPipeline(config_path)
                adaptive_results = enhanced_pipeline.run_enhanced_evaluation(
                    tasks_path=tasks_path,
                    anchors_path=anchors_path,
                    enable_self_eval=True,
                    enable_reliability=True,
                    enable_adaptive=True,
                    adaptive_domain=TaskDomain.ANALYTICAL
                )
                logging.info("✅ Adaptive evaluation completed")
                adaptive_success = True
            except Exception as adaptive_error:
                logging.warning(f"⚠️ Adaptive evaluation failed (continuing): {adaptive_error}")
                # Continue even if adaptive fails
            
            return {
                "message": "Evaluation completed successfully",
                "status": "completed",
                "standard_evaluation": result,
                "adaptive_evaluation_success": adaptive_success,
                "evaluation_id": result.get("evaluation_id", "unknown")
            }
            
        except Exception as e:
            logging.error(f"❌ Evaluation failed: {e}")
            logging.error(f"Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            logging.error(f"Traceback:\n{traceback.format_exc()}")
            
            # Save error details
            error_report = {
                "evaluation_id": f"eval_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            }
            try:
                save_json(error_report, str(Path(__file__).parent.parent / "data" / "evaluation_report.json"))
                logging.info("💾 Error report saved to evaluation_report.json")
            except Exception as save_error:
                logging.error(f"Failed to save error report: {save_error}")
            
            # Re-raise to ensure proper error handling
            raise e
        finally:
            # Always restore original working directory
            os.chdir(original_cwd)
    
    # Run the pipeline in background
    background_tasks.add_task(run_pipeline)
    
    return {
        "message": "Evaluation started in background", 
        "status": "running",
        "info": "Check /api/evaluation/status for progress"
    }

@app.post("/api/evaluation/run-adaptive")
async def run_adaptive_evaluation(config: Optional[Dict] = None):
    """Run only an adaptive evaluation."""
    try:
        # Import enhanced pipeline for adaptive evaluation
        from src.enhanced_pipeline import EnhancedEvaluationPipeline
        from src.adaptive_evaluation import TaskDomain
        
        logging.info("Starting adaptive evaluation...")
        config_path = str(Path(__file__).parent.parent / "config" / "judges_config.yaml")
        
        # Ensure absolute paths for data files
        data_dir = Path(__file__).parent.parent / "data"
        tasks_path = str(data_dir / "tasks.json")
        anchors_path = str(data_dir / "anchors.json")
        
        pipeline = EnhancedEvaluationPipeline(config_path)
        
        # Run enhanced evaluation with adaptive mode enabled
        results = pipeline.run_enhanced_evaluation(
            tasks_path=tasks_path,
            anchors_path=anchors_path,
            enable_self_eval=True,
            enable_reliability=True,
            enable_adaptive=True,
            adaptive_domain=TaskDomain.ANALYTICAL
        )
        
        # Clear cache to force refresh
        global _cached_data, _last_refresh
        _cached_data = {}
        _last_refresh = None
        
        logging.info("Adaptive evaluation completed")
        return {
            "message": "Adaptive evaluation completed",
            "status": "completed",
            "results": results
        }
    except Exception as e:
        logging.error(f"Adaptive evaluation failed: {e}")
        return {
            "message": f"Adaptive evaluation failed: {str(e)}",
            "status": "failed",
            "error": str(e)
        }

@app.get("/api/evaluation/status")
async def get_evaluation_status():
    """Get detailed evaluation status."""
    data_dir = Path(__file__).parent.parent / "data"
    
    status_info = {
        "status": "idle",
        "has_data": False,
        "last_evaluation": None,
        "files_present": {},
        "error": None
    }
    
    # Check what files exist
    required_files = [
        "evaluation_report.json",
        "final_performance.json", 
        "calibration_report.json",
        "tasks.json",
        "anchors.json"
    ]
    
    for file in required_files:
        file_path = data_dir / file
        status_info["files_present"][file] = file_path.exists()
    
    # Check evaluation report for status
    evaluation_report_path = data_dir / "evaluation_report.json"
    if evaluation_report_path.exists():
        try:
            report = load_json(str(evaluation_report_path))
            status_info["status"] = report.get("status", "unknown")
            status_info["last_evaluation"] = {
                "id": report.get("evaluation_id"),
                "start_time": report.get("start_time"),
                "end_time": report.get("end_time"),
                "duration": report.get("duration_seconds"),
                "agent_info": report.get("agent_info")
            }
            
            if report.get("error"):
                status_info["error"] = report.get("error")
            
            # Check if we have a complete successful evaluation
            if (status_info["status"] == "completed" and 
                status_info["files_present"]["final_performance.json"] and
                status_info["files_present"]["calibration_report.json"]):
                status_info["has_data"] = True
                
        except Exception as e:
            status_info["error"] = f"Error reading evaluation report: {str(e)}"
            status_info["status"] = "error"
    
    return status_info

@app.get("/api/test/pipeline")
async def test_pipeline_components():
    """Test that all pipeline components can be imported and initialized."""
    test_results = {
        "pipeline_available": USING_REAL_PIPELINE,
        "components": {},
        "files": {},
        "config": {}
    }
    
    # Test basic imports
    try:
        from src.pipeline import EvaluationPipeline
        test_results["components"]["EvaluationPipeline"] = "✅ Available"
    except Exception as e:
        test_results["components"]["EvaluationPipeline"] = f"❌ Error: {str(e)}"
    
    # Test config loading
    try:
        config_path = Path(__file__).parent.parent / "config" / "judges_config.yaml"
        if config_path.exists():
            test_results["config"]["judges_config"] = "✅ Found"
            # Test pipeline initialization
            try:
                config_path = str(Path(__file__).parent.parent / "config" / "judges_config.yaml")
                pipeline = EvaluationPipeline(config_path)
                test_results["components"]["Pipeline_Init"] = "✅ Initialized"
                
                # Test judge manager
                try:
                    judges = pipeline.judge_manager.get_judge_names()
                    test_results["components"]["Judges"] = f"✅ {len(judges)} judges: {judges}"
                except Exception as e:
                    test_results["components"]["Judges"] = f"❌ Error: {str(e)}"
                    
            except Exception as e:
                test_results["components"]["Pipeline_Init"] = f"❌ Error: {str(e)}"
        else:
            test_results["config"]["judges_config"] = "❌ Not found"
    except Exception as e:
        test_results["config"]["judges_config"] = f"❌ Error: {str(e)}"
    
    # Test data files
    data_dir = Path(__file__).parent.parent / "data"
    data_files = ["tasks.json", "anchors.json"]
    for file in data_files:
        file_path = data_dir / file
        if file_path.exists():
            try:
                data = load_json(str(file_path))
                if isinstance(data, list):
                    test_results["files"][file] = f"✅ {len(data)} items"
                else:
                    test_results["files"][file] = "✅ Available"
            except Exception as e:
                test_results["files"][file] = f"❌ Error reading: {str(e)}"
        else:
            test_results["files"][file] = "❌ Not found"
    
    return test_results

@app.post("/api/evaluation/run-fresh")
async def run_fresh_evaluation(background_tasks: BackgroundTasks):
    """Run a completely fresh evaluation by clearing existing data files first."""
    
    def run_fresh_pipeline():
        # Change working directory to project root for correct relative paths
        original_cwd = os.getcwd()
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)
        
        try:
            logging.info("🧹 Clearing existing evaluation data for fresh run...")
            
            # Clear existing evaluation data files (but keep tasks and anchors)
            data_dir = Path(__file__).parent.parent / "data"
            files_to_clear = [
                "evaluation_report.json",
                "performance_summary.json", 
                "final_performance.json",
                "calibration_report.json",
                "bias_offsets.json",
                "calibrated_scores.json",
                "raw_scores.json",
                "anchor_scores.json",
                "agent_outputs.json",
                "anchor_outputs.json",
                "canonical_metrics.json",
                "metric_proposals.json",
                "aggregated_scores.json"
            ]
            
            cleared_count = 0
            for file in files_to_clear:
                file_path = data_dir / file
                if file_path.exists():
                    file_path.unlink()
                    logging.info(f"🗑️ Deleted {file}")
                    cleared_count += 1
            
            logging.info(f"🧹 Cleared {cleared_count} files")
            
            # Clear any existing cache
            global _cached_data, _last_refresh
            _cached_data = {}
            _last_refresh = None
            
            logging.info("🚀 Starting fresh evaluation pipeline...")
            
            # Run fresh standard evaluation
            config_path = str(Path(__file__).parent.parent / "config" / "judges_config.yaml")
            
            # Force reload of tasks and anchors
            tasks_path = str(data_dir / "tasks.json")
            anchors_path = str(data_dir / "anchors.json")
            
            logging.info(f"📁 Using config: {config_path}")
            logging.info(f"📁 Using tasks: {tasks_path}")
            logging.info(f"📁 Using anchors: {anchors_path}")
            
            pipeline = EvaluationPipeline(config_path)
            result = pipeline.run_full_evaluation(tasks_path, anchors_path)
            logging.info(f"✅ Fresh evaluation completed: {result.get('status', 'unknown')}")
            
            return {
                "message": "Fresh evaluation completed successfully",
                "status": "completed",
                "evaluation": result,
                "evaluation_id": result.get("evaluation_id", "unknown"),
                "files_cleared": cleared_count
            }
            
        except Exception as e:
            logging.error(f"❌ Fresh evaluation failed: {e}")
            logging.error(f"Exception details: {type(e).__name__}: {str(e)}")
            import traceback
            logging.error(f"Traceback:\n{traceback.format_exc()}")
            
            # Save error details
            error_report = {
                "evaluation_id": f"fresh_eval_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            }
            try:
                save_json(error_report, str(Path(__file__).parent.parent / "data" / "evaluation_report.json"))
                logging.info("💾 Error report saved to evaluation_report.json")
            except Exception as save_error:
                logging.error(f"Failed to save error report: {save_error}")
                
            raise e
        finally:
            # Always restore original working directory
            os.chdir(original_cwd)
    
    # Run the pipeline in background
    background_tasks.add_task(run_fresh_pipeline)
    
    return {
        "message": "Fresh evaluation started (clearing existing data)", 
        "status": "running",
        "info": "Check /api/evaluation/status for progress"
    }

# Serve React build files in production
frontend_path = Path(__file__).parent.parent / "frontend" / "ageval-dashboard" / "build"
if frontend_path.exists() and (frontend_path / "static").exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path / "static")), name="static")
    
    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(frontend_path / "index.html"))
    
    @app.get("/{path:path}")
    async def serve_frontend_routes(path: str):
        file_path = frontend_path / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(frontend_path / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)