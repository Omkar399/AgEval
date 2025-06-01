import json
import os
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import subprocess
import asyncio
from typing import Dict, List, Any, Optional
import threading
import queue
import re
import time

app = FastAPI(title="AgEval Adaptive Dashboard API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Enhanced global state for running evaluations
evaluation_state = {
    "running": False,
    "progress": 0,
    "current_task": "",
    "current_step": "",
    "current_agent_response": "",
    "thinking_process": [],
    "evaluation_steps": [],
    "agent_trajectory": [],
    "current_ability": 0.0,
    "current_uncertainty": 0.0,
    "tasks_completed": 0,
    "results": [],
    "logs": [],
    "detailed_logs": [],
    "start_time": None,
    "end_time": None,
    "current_task_id": "",
    "current_difficulty": 0.0,
    "stage": "waiting",  # waiting, initializing, evaluating, analyzing, complete
    "substage": "",
    "irt_updates": []
}

def load_evaluation_data():
    """Load all evaluation data files including adaptive evaluation results"""
    data = {}
    
    # List of data files to load (including new adaptive evaluation files)
    files = {
        # Original files
        'enhanced_results': 'data/enhanced_evaluation_results.json',
        'comprehensive_analysis': 'data/comprehensive_analysis.json',
        'performance_summary': 'data/performance_summary.json',
        'failure_scores': 'data/failure_adjusted_scores.json',
        'calibrated_scores': 'data/calibrated_scores.json',
        'raw_scores': 'data/raw_scores.json',
        'tasks': 'data/tasks.json',
        'anchors': 'data/anchors.json',
        'final_performance': 'data/final_performance.json',
        
        # New adaptive evaluation files
        'adaptive_results': 'data/adaptive_evaluation_results.json',
        'detailed_adaptive': 'data/detailed_adaptive_results.json',
        'adaptive_analysis': 'data/adaptive_comprehensive_analysis.json',
        'adaptive_base_tasks': 'data/adaptive_base_tasks.json',
        
        # Static evaluation results for comparison
        'static_results': 'data/static_evaluation_results.json'
    }
    
    for key, filepath in files.items():
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data[key] = json.load(f)
            except Exception as e:
                data[key] = {"error": str(e)}
        else:
            data[key] = {}
    
    return data

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/data")
async def get_all_data():
    """Get all evaluation data"""
    try:
        data = load_evaluation_data()
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/{data_type}")
async def get_specific_data(data_type: str):
    """Get specific type of evaluation data"""
    try:
        data = load_evaluation_data()
        if data_type in data:
            return JSONResponse(content=data[data_type])
        else:
            raise HTTPException(status_code=404, detail=f"Data type '{data_type}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/adaptive/overview")
async def get_adaptive_overview():
    """Get adaptive evaluation overview metrics"""
    try:
        data = load_evaluation_data()
        adaptive_data = data.get('adaptive_results', {})
        
        if not adaptive_data:
            return JSONResponse(content={"status": "no_data"})
        
        # Calculate overview metrics
        overview = {
            "total_evaluations": len(adaptive_data.get("agent_results", {})),
            "average_tasks_per_agent": 0,
            "convergence_rate": 0,
            "efficiency_gain": 0,
            "total_tasks_saved": 0,
            "average_convergence_steps": 0
        }
        
        agent_results = adaptive_data.get("agent_results", {})
        if agent_results:
            task_counts = []
            convergence_steps = []
            
            for agent_id, result in agent_results.items():
                trajectory = result.get("trajectory", [])
                task_counts.append(len(trajectory))
                
                if result.get("converged", False):
                    convergence_steps.append(result.get("convergence_step", len(trajectory)))
            
            if task_counts:
                overview["average_tasks_per_agent"] = sum(task_counts) / len(task_counts)
                overview["convergence_rate"] = sum(1 for r in agent_results.values() if r.get("converged", False)) / len(agent_results) * 100
                
                # Compare with static evaluation
                static_task_count = len(data.get("tasks", {}))
                if static_task_count > 0:
                    overview["efficiency_gain"] = (1 - overview["average_tasks_per_agent"] / static_task_count) * 100
                    overview["total_tasks_saved"] = (static_task_count - overview["average_tasks_per_agent"]) * len(agent_results)
            
            if convergence_steps:
                overview["average_convergence_steps"] = sum(convergence_steps) / len(convergence_steps)
        
        return JSONResponse(content=overview)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/adaptive/trajectory/{agent_id}")
async def get_agent_trajectory(agent_id: str):
    """Get trajectory data for a specific agent"""
    try:
        data = load_evaluation_data()
        adaptive_data = data.get('adaptive_results', {})
        agent_results = adaptive_data.get("agent_results", {})
        
        if agent_id not in agent_results:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        
        result = agent_results[agent_id]
        trajectory = result.get("trajectory", [])
        
        # Prepare trajectory data for plotting
        trajectory_data = {
            "steps": list(range(1, len(trajectory) + 1)),
            "abilities": [t.get("ability_estimate", 0) for t in trajectory],
            "difficulties": [t.get("task_difficulty", 0) for t in trajectory],
            "uncertainties": [t.get("uncertainty", 0) for t in trajectory],
            "outcomes": [t.get("outcome", False) for t in trajectory],
            "task_ids": [t.get("task_id", "") for t in trajectory],
            "converged": result.get("converged", False),
            "convergence_step": result.get("convergence_step", None),
            "final_ability": result.get("final_ability", 0),
            "confidence_interval": result.get("confidence_interval", [0, 0])
        }
        
        return JSONResponse(content=trajectory_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/list")
async def get_agents_list():
    """Get list of all agents"""
    try:
        data = load_evaluation_data()
        agents = set()
        
        # Collect agents from different data sources
        for source in ['adaptive_results', 'enhanced_results', 'comprehensive_analysis']:
            if source in data and isinstance(data[source], dict):
                if 'agent_results' in data[source]:
                    agents.update(data[source]['agent_results'].keys())
                elif 'agents' in data[source]:
                    agents.update(data[source]['agents'].keys())
        
        return JSONResponse(content={"agents": sorted(list(agents))})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/{agent_id}/performance")
async def get_agent_performance(agent_id: str):
    """Get detailed performance data for a specific agent"""
    try:
        data = load_evaluation_data()
        
        # Collect performance data from various sources
        performance = {
            "agent_id": agent_id,
            "adaptive_performance": {},
            "static_performance": {},
            "comparison_metrics": {}
        }
        
        # Get adaptive performance
        adaptive_data = data.get('adaptive_results', {})
        if 'agent_results' in adaptive_data and agent_id in adaptive_data['agent_results']:
            agent_result = adaptive_data['agent_results'][agent_id]
            performance['adaptive_performance'] = {
                "final_ability": agent_result.get("final_ability", 0),
                "confidence_interval": agent_result.get("confidence_interval", [0, 0]),
                "converged": agent_result.get("converged", False),
                "tasks_completed": len(agent_result.get("trajectory", [])),
                "convergence_step": agent_result.get("convergence_step", None)
            }
        
        # Get static performance from comprehensive analysis
        comp_analysis = data.get('comprehensive_analysis', {})
        if 'agents' in comp_analysis and agent_id in comp_analysis['agents']:
            agent_data = comp_analysis['agents'][agent_id]
            performance['static_performance'] = {
                "overall_score": agent_data.get("overall_performance", {}).get("score", 0),
                "total_tasks": agent_data.get("overall_performance", {}).get("total_tasks", 0),
                "category_scores": agent_data.get("category_performance", {})
            }
        
        # Calculate comparison metrics
        if performance['adaptive_performance'] and performance['static_performance']:
            static_tasks = performance['static_performance'].get('total_tasks', 0)
            adaptive_tasks = performance['adaptive_performance'].get('tasks_completed', 0)
            
            if static_tasks > 0:
                performance['comparison_metrics'] = {
                    "efficiency_gain": ((static_tasks - adaptive_tasks) / static_tasks) * 100,
                    "tasks_saved": static_tasks - adaptive_tasks,
                    "convergence_achieved": performance['adaptive_performance'].get('converged', False)
                }
        
        return JSONResponse(content=performance)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluation/run")
async def run_evaluation(request: Request):
    """Run a new evaluation"""
    global evaluation_state
    
    if evaluation_state["running"]:
        raise HTTPException(status_code=400, detail="Evaluation already running")
    
    try:
        body = await request.json()
        evaluation_type = body.get("type", "adaptive")
        num_agents = body.get("num_agents", 5)
        
        # Start evaluation in background
        evaluation_state["running"] = True
        evaluation_state["progress"] = 0
        evaluation_state["current_task"] = "Initializing..."
        evaluation_state["results"] = []
        evaluation_state["logs"] = []
        evaluation_state["start_time"] = datetime.now().isoformat()
        
        # Run evaluation asynchronously
        asyncio.create_task(run_evaluation_async(evaluation_type, num_agents))
        
        return JSONResponse(content={"status": "started", "message": "Evaluation started successfully"})
    except Exception as e:
        evaluation_state["running"] = False
        raise HTTPException(status_code=500, detail=str(e))

async def parse_evaluation_output(line_text: str):
    """Parse evaluation output and extract structured information"""
    global evaluation_state
    
    # Parse different types of log messages
    if "Starting adaptive evaluation" in line_text:
        evaluation_state["stage"] = "initializing"
        evaluation_state["substage"] = "Setting up adaptive evaluation"
        
    elif "Selected difficulty" in line_text:
        # Extract difficulty value
        match = re.search(r"Selected difficulty (\d+\.\d+)", line_text)
        if match:
            evaluation_state["current_difficulty"] = float(match.group(1))
            evaluation_state["stage"] = "evaluating"
            evaluation_state["substage"] = f"Selected task difficulty: {match.group(1)}"
            
    elif "task_id" in line_text and "adaptive_" in line_text:
        # Extract task ID
        match = re.search(r"(adaptive_\w+_\d+_\d+\.\d+)", line_text)
        if match:
            evaluation_state["current_task_id"] = match.group(1)
            evaluation_state["substage"] = f"Executing task: {match.group(1)}"
            
    elif "Agent generated response" in line_text:
        evaluation_state["substage"] = "Agent thinking and responding..."
        evaluation_state["thinking_process"].append({
            "timestamp": datetime.now().isoformat(),
            "step": "Agent Response Generated",
            "details": line_text
        })
        
    elif "Updated ability:" in line_text:
        # Extract ability and uncertainty
        match = re.search(r"Updated ability: ([-\d\.]+) ± ([-\d\.]+)", line_text)
        if match:
            ability = float(match.group(1))
            uncertainty = float(match.group(2))
            evaluation_state["current_ability"] = ability
            evaluation_state["current_uncertainty"] = uncertainty
            evaluation_state["tasks_completed"] += 1
            
            # Add to trajectory
            evaluation_state["agent_trajectory"].append({
                "step": evaluation_state["tasks_completed"],
                "ability": ability,
                "uncertainty": uncertainty,
                "difficulty": evaluation_state["current_difficulty"],
                "task_id": evaluation_state["current_task_id"],
                "timestamp": datetime.now().isoformat()
            })
            
            # Add IRT update
            evaluation_state["irt_updates"].append({
                "timestamp": datetime.now().isoformat(),
                "ability_estimate": ability,
                "uncertainty": uncertainty,
                "convergence_metric": uncertainty,
                "step": evaluation_state["tasks_completed"]
            })
            
            evaluation_state["substage"] = f"IRT Update: θ={ability:.3f} ± {uncertainty:.3f}"
            
    elif "Item" in line_text and "Difficulty" in line_text and "Performance" in line_text:
        # Extract performance information
        evaluation_state["thinking_process"].append({
            "timestamp": datetime.now().isoformat(),
            "step": "Performance Analysis",
            "details": line_text
        })
        
    elif "Progress:" in line_text:
        try:
            progress_value = int(line_text.split(":")[-1].strip().replace("%", ""))
            evaluation_state["progress"] = progress_value
        except:
            pass
            
    elif "Demonstration completed" in line_text:
        evaluation_state["stage"] = "complete"
        evaluation_state["substage"] = "Evaluation finished successfully"
        evaluation_state["progress"] = 100

async def run_evaluation_async(evaluation_type: str, num_agents: int):
    """Run evaluation in background with detailed progress tracking"""
    global evaluation_state
    
    try:
        # Reset state
        evaluation_state.update({
            "stage": "initializing",
            "substage": "Starting evaluation process...",
            "thinking_process": [],
            "evaluation_steps": [],
            "agent_trajectory": [],
            "current_ability": 0.0,
            "current_uncertainty": 0.0,
            "tasks_completed": 0,
            "detailed_logs": [],
            "irt_updates": []
        })
        
        # Broadcast initial state
        await manager.broadcast({"type": "evaluation_update", "data": evaluation_state})
        
        # Determine which script to run
        if evaluation_type == "adaptive":
            # Check if we have demo mode enabled or use real evaluation
            if os.path.exists("demo_evaluation.py"):
                script = "demo_evaluation.py"  # Use demo for faster testing
            else:
                script = "run_enhanced_evaluation.py"
        else:
            script = "run_evaluation.py"
        
        # Run the evaluation script
        process = await asyncio.create_subprocess_exec(
            "python", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Monitor progress with detailed parsing
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            
            line_text = line.decode().strip()
            if line_text:  # Only process non-empty lines
                evaluation_state["logs"].append(line_text)
                evaluation_state["detailed_logs"].append({
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "message": line_text
                })
                
                # Parse and update structured information
                await parse_evaluation_output(line_text)
                
                # Update current task display
                evaluation_state["current_task"] = line_text[:200]  # Show more detail
                
                # Broadcast updates to connected clients
                await manager.broadcast({
                    "type": "evaluation_update", 
                    "data": evaluation_state
                })
                
                # Add small delay to make updates visible
                await asyncio.sleep(0.1)
        
        # Handle stderr
        stderr_line = await process.stderr.read()
        if stderr_line:
            stderr_text = stderr_line.decode().strip()
            if stderr_text:
                evaluation_state["detailed_logs"].append({
                    "timestamp": datetime.now().isoformat(),
                    "level": "ERROR",
                    "message": stderr_text
                })
        
        await process.wait()
        
        # Mark as complete
        evaluation_state["running"] = False
        evaluation_state["progress"] = 100
        evaluation_state["stage"] = "complete"
        evaluation_state["substage"] = "Evaluation completed successfully"
        evaluation_state["current_task"] = "Evaluation complete - Processing results..."
        evaluation_state["end_time"] = datetime.now().isoformat()
        
        # Final broadcast
        await manager.broadcast({
            "type": "evaluation_complete", 
            "data": evaluation_state
        })
        
    except Exception as e:
        evaluation_state["running"] = False
        evaluation_state["stage"] = "error"
        evaluation_state["substage"] = f"Error: {str(e)}"
        evaluation_state["current_task"] = f"Error: {str(e)}"
        evaluation_state["logs"].append(f"Error: {str(e)}")
        evaluation_state["detailed_logs"].append({
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "message": str(e)
        })
        
        # Broadcast error
        await manager.broadcast({
            "type": "evaluation_error", 
            "data": evaluation_state
        })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming WebSocket messages if needed
            if data == "get_status":
                await websocket.send_json(evaluation_state)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/evaluation/status")
async def get_evaluation_status():
    """Get current evaluation status"""
    return JSONResponse(content=evaluation_state)

@app.get("/api/plots/trajectory/{agent_id}")
async def get_trajectory_plot(agent_id: str):
    """Generate trajectory plot for an agent"""
    try:
        data = load_evaluation_data()
        adaptive_data = data.get('adaptive_results', {})
        agent_results = adaptive_data.get("agent_results", {})
        
        if agent_id not in agent_results:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        
        result = agent_results[agent_id]
        trajectory = result.get("trajectory", [])
        
        # Create the plot
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=("Ability Estimation Trajectory", "Task Difficulty vs Performance"),
            vertical_spacing=0.15
        )
        
        # Trajectory plot
        steps = list(range(1, len(trajectory) + 1))
        abilities = [t.get("ability_estimate", 0) for t in trajectory]
        uncertainties = [t.get("uncertainty", 0) for t in trajectory]
        
        # Add ability line
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=abilities,
                mode='lines+markers',
                name='Ability Estimate',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Add uncertainty bands
        upper_bound = [a + u for a, u in zip(abilities, uncertainties)]
        lower_bound = [a - u for a, u in zip(abilities, uncertainties)]
        
        fig.add_trace(
            go.Scatter(
                x=steps + steps[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(0,100,255,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # Mark convergence point
        if result.get("converged", False):
            convergence_step = result.get("convergence_step", len(trajectory))
            fig.add_vline(
                x=convergence_step,
                line_dash="dash",
                line_color="green",
                annotation_text="Converged",
                row=1, col=1
            )
        
        # Difficulty vs Performance plot
        difficulties = [t.get("task_difficulty", 0) for t in trajectory]
        outcomes = [1 if t.get("outcome", False) else 0 for t in trajectory]
        
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=difficulties,
                mode='markers',
                name='Task Difficulty',
                marker=dict(
                    size=10,
                    color=outcomes,
                    colorscale=['red', 'green'],
                    showscale=False
                ),
                text=[f"{'Success' if o else 'Failure'}" for o in outcomes],
                hovertemplate='Step: %{x}<br>Difficulty: %{y:.2f}<br>%{text}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Evaluation Step", row=2, col=1)
        fig.update_yaxes(title_text="Ability Estimate", row=1, col=1)
        fig.update_yaxes(title_text="Task Difficulty", row=2, col=1)
        
        fig.update_layout(
            title=f"Adaptive Evaluation Trajectory - {agent_id}",
            height=700,
            showlegend=True
        )
        
        return JSONResponse(content=fig.to_json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/plots/comparison")
async def get_comparison_plot():
    """Generate comparison plot between adaptive and static evaluation"""
    try:
        data = load_evaluation_data()
        
        # Prepare comparison data
        agents = []
        adaptive_tasks = []
        static_tasks = []
        efficiency_gains = []
        
        adaptive_data = data.get('adaptive_results', {})
        comp_analysis = data.get('comprehensive_analysis', {})
        
        if 'agent_results' in adaptive_data and 'agents' in comp_analysis:
            for agent_id in adaptive_data['agent_results'].keys():
                if agent_id in comp_analysis['agents']:
                    agents.append(agent_id)
                    
                    # Adaptive tasks
                    trajectory = adaptive_data['agent_results'][agent_id].get('trajectory', [])
                    adaptive_tasks.append(len(trajectory))
                    
                    # Static tasks
                    static_count = comp_analysis['agents'][agent_id].get('overall_performance', {}).get('total_tasks', 0)
                    static_tasks.append(static_count)
                    
                    # Efficiency gain
                    if static_count > 0:
                        efficiency_gains.append(((static_count - len(trajectory)) / static_count) * 100)
                    else:
                        efficiency_gains.append(0)
        
        # Create comparison plot
        fig = go.Figure()
        
        # Add bars for task counts
        fig.add_trace(go.Bar(
            name='Static Evaluation',
            x=agents,
            y=static_tasks,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Adaptive Evaluation',
            x=agents,
            y=adaptive_tasks,
            marker_color='darkblue'
        ))
        
        # Add efficiency gain as line
        fig.add_trace(go.Scatter(
            name='Efficiency Gain (%)',
            x=agents,
            y=efficiency_gains,
            mode='lines+markers',
            yaxis='y2',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ))
        
        # Update layout
        fig.update_layout(
            title='Adaptive vs Static Evaluation Comparison',
            xaxis_title='Agents',
            yaxis_title='Number of Tasks',
            yaxis2=dict(
                title='Efficiency Gain (%)',
                overlaying='y',
                side='right',
                range=[0, 100]
            ),
            barmode='group',
            height=600,
            hovermode='x unified'
        )
        
        return JSONResponse(content=fig.to_json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/hardthinking/run")
async def run_hard_thinking(request: Request):
    """Run hard thinking multi-LLM ensemble"""
    try:
        body = await request.json()
        query = body.get("query", "")
        problem_type = body.get("problem_type", "general")
        complexity = body.get("complexity", "moderate")
        strategy = body.get("strategy", "weighted")
        
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Initialize Hard Thinking system with mock API keys
        # In production, these would come from environment variables or config
        api_keys = {
            "openai": "demo_key",  # Mock for demo
            "anthropic": "demo_key",  # Mock for demo
            "google": "demo_key"  # Mock for demo
        }
        
        # Convert string parameters to enums
        try:
            from src.hard_thinking import HarderThinkingSystem, TaskComplexity, EnsembleStrategy
            
            complexity_enum = TaskComplexity(complexity)
            strategy_enum = EnsembleStrategy(strategy)
            
            # Run the hard thinking process
            hard_thinking = HarderThinkingSystem(api_keys)
            result = await hard_thinking.process_query(
                query=query,
                problem_type=problem_type,
                complexity=complexity_enum,
                strategy=strategy_enum
            )
            
            # Convert result to dict for JSON response
            result_dict = {
                "query": result.query,
                "best_model": result.best_model,
                "final_answer": result.final_answer,
                "confidence_score": result.confidence_score,
                "consensus_level": result.consensus_level,
                "processing_time": result.processing_time,
                "total_tokens": result.total_tokens,
                "model_breakdown": result.model_breakdown,
                "decomposition": [
                    {
                        "id": subtask.id,
                        "task": subtask.task,
                        "complexity": subtask.complexity.value
                    } for subtask in result.decomposition
                ],
                "strategy_used": result.strategy_used.value
            }
            
            return JSONResponse(content={
                "status": "completed",
                "message": "Hard thinking process completed successfully",
                "result": result_dict
            })
            
        except ImportError as e:
            # Fallback to simulation if import fails
            return JSONResponse(content={
                "status": "started",
                "message": "Hard thinking process initiated (simulation mode)",
                "query": query,
                "config": {
                    "problem_type": problem_type,
                    "complexity": complexity,
                    "strategy": strategy
                }
            })
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)