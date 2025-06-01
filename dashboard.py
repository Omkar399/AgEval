import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import os
from PIL import Image
import base64

# Page configuration
st.set_page_config(
    page_title="AgEval Adaptive Dashboard - Evolution & Results",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .adaptive-header {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .adaptive-card {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3);
    }
    
    .irt-card {
        background: linear-gradient(135deg, #A8E6CF 0%, #FF8B94 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(168, 230, 207, 0.3);
    }
    
    .success-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(255, 152, 0, 0.3);
    }
    
    .info-card {
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(33, 150, 243, 0.3);
    }
    
    .agent-card {
        background: linear-gradient(135deg, #9C27B0 0%, #673AB7 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(156, 39, 176, 0.3);
    }
    
    .evolved-prompt-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #FF6B6B;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #212529;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.5;
        box-shadow: 0 8px 16px rgba(255, 107, 107, 0.2);
    }
    
    .task-prompt-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.4;
        color: #212529;
    }
    
    .adaptive-prompt-text {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #212529;
        font-weight: 500;
    }
    
    .evolved-characteristic {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
        color: #0d47a1;
        font-weight: 500;
    }
    
    .response-container {
        background-color: #f0f8f0;
        border: 1px solid #d4edda;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .trajectory-insight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .convergence-status {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem;
    }
    
    .converged {
        background-color: #4CAF50;
        color: white;
    }
    
    .not-converged {
        background-color: #FF5722;
        color: white;
    }
    
    .metric-excellent {
        color: #4CAF50;
        font-weight: bold;
    }
    
    .metric-good {
        color: #ff9800;
        font-weight: bold;
    }
    
    .metric-poor {
        color: #f44336;
        font-weight: bold;
    }
    
    .expandable-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .expandable-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    .agent-rank-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    
    .rank-gold {
        background-color: #ffd700;
        color: #333;
    }
    
    .rank-silver {
        background-color: #c0c0c0;
        color: #333;
    }
    
    .rank-bronze {
        background-color: #cd7f32;
        color: white;
    }
    
    .rank-other {
        background-color: #6c757d;
        color: white;
    }
    
    .feature-badge {
        display: inline-block;
        padding: 0.2rem 0.4rem;
        border-radius: 6px;
        font-size: 0.75rem;
        margin: 0.1rem;
    }
    
    .feature-enabled {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .feature-disabled {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .adaptive-evolution-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .judge-score-container {
        background-color: #f8f9fa;
        border-radius: 6px;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    
    .self-eval-container {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.5rem 0;
    }
    
    .failure-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.75rem;
        border-radius: 0 6px 6px 0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
                st.warning(f"Could not load {filepath}: {e}")
                data[key] = {}
        else:
            data[key] = {}
    
    return data

def create_adaptive_evaluation_overview(adaptive_data):
    """Create overview of adaptive evaluation results"""
    if not adaptive_data:
        st.warning("No adaptive evaluation data available.")
        return
    
    st.markdown('<h2 class="adaptive-header">🎯 Adaptive Evaluation Results</h2>', unsafe_allow_html=True)
    
    # Extract key metrics
    eval_results = adaptive_data.get('adaptive_evaluation_results', {})
    if isinstance(eval_results, dict) and 'adaptive_evaluation_results' in eval_results:
        eval_results = eval_results['adaptive_evaluation_results']
    
    ability_estimate = eval_results.get('final_ability_estimate', 0)
    ability_percentile = eval_results.get('ability_percentile', 0)
    convergence = eval_results.get('convergence_achieved', False)
    total_items = eval_results.get('total_items_administered', 0)
    ability_se = eval_results.get('ability_standard_error', 0)
    
    # Performance analysis
    perf_analysis = adaptive_data.get('performance_analysis', {})
    avg_performance = perf_analysis.get('average_performance', 0)
    consistency = perf_analysis.get('performance_consistency', 0)
    difficulty_range = perf_analysis.get('difficulty_range_explored', 0)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="adaptive-card">
            <h3>🎯 Final Ability</h3>
            <h2>{ability_estimate:.2f}</h2>
            <p><strong>{ability_percentile:.1f}th</strong> Percentile</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        convergence_class = "converged" if convergence else "not-converged"
        convergence_text = "✅ Converged" if convergence else "⏳ Continuing"
        st.markdown(f"""
        <div class="adaptive-card">
            <h3>🔄 Convergence</h3>
            <div class="convergence-status {convergence_class}">{convergence_text}</div>
            <p><strong>{total_items}</strong> Items Used</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        precision_score = 1.0 / (1.0 + ability_se) if ability_se > 0 else 1.0
        st.markdown(f"""
        <div class="adaptive-card">
            <h3>📊 Performance</h3>
            <h2>{avg_performance:.3f}</h2>
            <p><strong>{consistency:.3f}</strong> Consistency</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        efficiency = 1.0 - (total_items / 15.0) if total_items > 0 else 0
        st.markdown(f"""
        <div class="adaptive-card">
            <h3>⚡ Efficiency</h3>
            <h2>{efficiency:.3f}</h2>
            <p><strong>{difficulty_range:.3f}</strong> Range Explored</p>
        </div>
        """, unsafe_allow_html=True)

def create_adaptive_trajectory_plot(adaptive_data):
    """Create adaptive evaluation trajectory visualization"""
    if not adaptive_data:
        return None
    
    perf_analysis = adaptive_data.get('performance_analysis', {})
    detailed_responses = adaptive_data.get('detailed_responses', [])
    irt_history = adaptive_data.get('irt_response_history', [])
    
    if not detailed_responses and not irt_history:
        return None
    
    # Create trajectory plot
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Ability Evolution', 'Difficulty Progression', 
                       'Performance Trajectory', 'Response Time Evolution',
                       'Reasoning Complexity', 'IRT Discrimination'),
        vertical_spacing=0.1,
        specs=[[{}, {}], [{}, {}], [{}, {}]]
    )
    
    # Ability evolution from IRT history
    if irt_history:
        abilities = [entry.get('ability_at_time', 0) for entry in irt_history]
        items = list(range(1, len(abilities) + 1))
        
        fig.add_trace(
            go.Scatter(x=items, y=abilities, mode='lines+markers',
                      name='Ability Estimate', line=dict(color='#FF6B6B', width=3)),
            row=1, col=1
        )
    
    # Difficulty progression
    if detailed_responses:
        difficulties = [r.get('difficulty', 0) for r in detailed_responses]
        items = list(range(1, len(difficulties) + 1))
        
        fig.add_trace(
            go.Scatter(x=items, y=difficulties, mode='lines+markers',
                      name='Task Difficulty', line=dict(color='#4ECDC4', width=3)),
            row=1, col=2
        )
    
    # Performance trajectory
    performance_traj = perf_analysis.get('performance_trajectory', [])
    if performance_traj:
        items = list(range(1, len(performance_traj) + 1))
        fig.add_trace(
            go.Scatter(x=items, y=performance_traj, mode='lines+markers',
                      name='Performance', line=dict(color='#4CAF50', width=3)),
            row=2, col=1
        )
    
    # Response time evolution
    if detailed_responses:
        times = [r.get('time_taken', 0) for r in detailed_responses]
        items = list(range(1, len(times) + 1))
        
        fig.add_trace(
            go.Scatter(x=items, y=times, mode='lines+markers',
                      name='Response Time', line=dict(color='#FF9800', width=3)),
            row=2, col=2
        )
    
    # Reasoning complexity
    if detailed_responses:
        reasoning_steps = [r.get('reasoning_steps', 0) for r in detailed_responses]
        items = list(range(1, len(reasoning_steps) + 1))
        
        fig.add_trace(
            go.Bar(x=items, y=reasoning_steps, name='Reasoning Steps',
                  marker=dict(color='#9C27B0')),
            row=3, col=1
        )
    
    # IRT discrimination parameters
    if irt_history:
        discriminations = [entry.get('discrimination', 0) for entry in irt_history]
        items = list(range(1, len(discriminations) + 1))
        
        fig.add_trace(
            go.Scatter(x=items, y=discriminations, mode='lines+markers',
                      name='Discrimination', line=dict(color='#795548', width=3)),
            row=3, col=2
        )
    
    fig.update_layout(height=900, title_text="🎯 Adaptive Evaluation Trajectory Analysis",
                     showlegend=False)
    
    return fig

def create_evolved_prompts_section(adaptive_data):
    """Display evolved prompts and their progression"""
    if not adaptive_data:
        return
    
    st.markdown('<h2 class="adaptive-header">🧬 Evolved Prompts & Task Generation</h2>', unsafe_allow_html=True)
    
    detailed_responses = adaptive_data.get('detailed_responses', [])
    
    if not detailed_responses:
        st.warning("No evolved prompt data available.")
        return
    
    # Group by task type and show evolution
    task_groups = {}
    for response in detailed_responses:
        task_id = response.get('task_id', '')
        task_type = 'atomic' if 'atomic' in task_id else 'compositional' if 'compositional' in task_id else 'end2end'
        
        if task_type not in task_groups:
            task_groups[task_type] = []
        task_groups[task_type].append(response)
    
    # Display each task type group
    for task_type, responses in task_groups.items():
        st.subheader(f"🎯 {task_type.title()} Task Evolution")
        
        # Show difficulty progression for this task type
        difficulties = [r.get('difficulty', 0) for r in responses]
        performances = [r.get('performance', 0) for r in responses]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create mini progression chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(difficulties) + 1)),
                y=difficulties,
                mode='lines+markers',
                name='Difficulty',
                line=dict(color='#FF6B6B', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(performances) + 1)),
                y=performances,
                mode='lines+markers',
                name='Performance',
                line=dict(color='#4ECDC4', width=3),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title=f"{task_type.title()} Difficulty-Performance Evolution",
                xaxis_title="Task Sequence",
                yaxis_title="Difficulty Level",
                yaxis2=dict(title="Performance", overlaying='y', side='right'),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Task type statistics
            avg_difficulty = np.mean(difficulties)
            avg_performance = np.mean(performances)
            total_tasks = len(responses)
            avg_time = np.mean([r.get('time_taken', 0) for r in responses])
            
            st.markdown(f"""
            <div class="trajectory-insight">
                <h4>📊 {task_type.title()} Stats</h4>
                <p><strong>Tasks:</strong> {total_tasks}</p>
                <p><strong>Avg Difficulty:</strong> {avg_difficulty:.3f}</p>
                <p><strong>Avg Performance:</strong> {avg_performance:.3f}</p>
                <p><strong>Avg Time:</strong> {avg_time:.2f}s</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show sample evolved prompts
        with st.expander(f"📝 View {task_type.title()} Task Details"):
            for i, response in enumerate(responses[:3]):  # Show first 3 tasks
                task_id = response.get('task_id', '')
                difficulty = response.get('difficulty', 0)
                performance = response.get('performance', 0)
                reasoning_steps = response.get('reasoning_steps', 0)
                time_taken = response.get('time_taken', 0)
                
                st.markdown(f"""
                <div class="evolved-prompt-container">
                    <h4>🎯 Task: {task_id}</h4>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
                        <span class="adaptive-evolution-badge">Difficulty: {difficulty:.2f}</span>
                        <span class="adaptive-evolution-badge">Performance: {performance:.2f}</span>
                        <span class="adaptive-evolution-badge">Reasoning: {reasoning_steps} steps</span>
                        <span class="adaptive-evolution-badge">Time: {time_taken:.2f}s</span>
                    </div>
                    <p><strong>Evolved Characteristics:</strong></p>
                    <ul>
                        <li>Adaptive difficulty calibration: {difficulty:.3f}</li>
                        <li>IRT-based complexity scaling</li>
                        <li>Dynamic reasoning requirement: {reasoning_steps} steps</li>
                        <li>Response optimization: {response.get('response_length', 0)} chars</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

def create_irt_analysis_section(adaptive_data):
    """Create Item Response Theory analysis section"""
    if not adaptive_data:
        return
    
    st.markdown('<h2 class="adaptive-header">📊 Item Response Theory Analysis</h2>', unsafe_allow_html=True)
    
    irt_history = adaptive_data.get('irt_response_history', [])
    
    if not irt_history:
        st.warning("No IRT analysis data available.")
        return
    
    # Extract IRT parameters
    discriminations = [entry.get('discrimination', 0) for entry in irt_history]
    difficulties = [entry.get('difficulty', 0) for entry in irt_history]
    abilities = [entry.get('ability_at_time', 0) for entry in irt_history]
    performances = [entry.get('performance', 0) for entry in irt_history]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # IRT Parameter Evolution
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Discrimination Parameters', 'Ability vs Difficulty'))
        
        # Discrimination evolution
        fig.add_trace(
            go.Scatter(x=list(range(1, len(discriminations) + 1)), 
                      y=discriminations,
                      mode='lines+markers',
                      name='Discrimination',
                      line=dict(color='#FF6B6B', width=3)),
            row=1, col=1
        )
        
        # Ability vs Difficulty scatter
        fig.add_trace(
            go.Scatter(x=difficulties, y=abilities,
                      mode='markers',
                      marker=dict(size=10, color=performances, colorscale='RdYlGn',
                                colorbar=dict(title="Performance")),
                      name='Ability-Difficulty'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="IRT Parameter Analysis")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # IRT Statistics
        avg_discrimination = np.mean(discriminations)
        difficulty_range = max(difficulties) - min(difficulties)
        ability_growth = abilities[-1] - abilities[0] if len(abilities) > 1 else 0
        
        st.markdown(f"""
        <div class="irt-card">
            <h3>📈 IRT Model Statistics</h3>
            <p><strong>Avg Discrimination:</strong> {avg_discrimination:.3f}</p>
            <p><strong>Difficulty Range:</strong> {difficulty_range:.3f}</p>
            <p><strong>Ability Growth:</strong> {ability_growth:.3f}</p>
            <p><strong>Model Quality:</strong> {'Excellent' if difficulty_range > 1.0 else 'Good'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # IRT Model Validation
        st.subheader("🔬 Model Validation")
        
        # Calculate model fit metrics
        correlation = np.corrcoef(abilities[1:], performances[1:])[0, 1] if len(abilities) > 1 else 0
        
        validation_metrics = {
            "Discrimination Quality": "High" if avg_discrimination > 1.0 else "Medium",
            "Difficulty Spread": "Excellent" if difficulty_range > 2.0 else "Good",
            "Ability-Performance Correlation": f"{correlation:.3f}",
            "Convergence Trend": "Positive" if ability_growth > 0 else "Stable"
        }
        
        for metric, value in validation_metrics.items():
            st.write(f"**{metric}:** {value}")

def display_trajectory_plot():
    """Display the adaptive evaluation trajectory plot if available"""
    trajectory_path = "reports/adaptive_evaluation_trajectory.png"
    
    if os.path.exists(trajectory_path):
        st.markdown('<h2 class="adaptive-header">📈 Adaptive Evaluation Trajectory</h2>', unsafe_allow_html=True)
        
        try:
            image = Image.open(trajectory_path)
            st.image(image, caption="Adaptive Evaluation Trajectory - Real-time ability estimation and difficulty calibration", 
                    use_container_width=True)
            
            st.markdown("""
            <div class="trajectory-insight">
                <h4>🎯 Trajectory Insights</h4>
                <p>This plot shows the real-time evolution of:</p>
                <ul>
                    <li><strong>Ability Estimation:</strong> How the agent's estimated ability evolved</li>
                    <li><strong>Difficulty Calibration:</strong> Dynamic task difficulty adjustments</li>
                    <li><strong>Performance Tracking:</strong> Success rate across different difficulty levels</li>
                    <li><strong>Convergence Analysis:</strong> Statistical certainty in ability measurement</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading trajectory plot: {e}")
    else:
        st.info("Trajectory plot not available. Run an adaptive evaluation to generate the plot.")

def create_comparison_analysis(data):
    """Compare adaptive vs static evaluation if both are available"""
    adaptive_data = data.get('adaptive_results', {})
    static_data = data.get('static_results', {})
    
    if not adaptive_data and not static_data:
        return
    
    st.markdown('<h2 class="adaptive-header">⚖️ Adaptive vs Static Evaluation Comparison</h2>', unsafe_allow_html=True)
    
    if adaptive_data and static_data:
        # Both available - show comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Adaptive Evaluation")
            adaptive_metrics = adaptive_data.get('adaptive_evaluation_results', {})
            if isinstance(adaptive_metrics, dict) and 'adaptive_evaluation_results' in adaptive_metrics:
                adaptive_metrics = adaptive_metrics['adaptive_evaluation_results']
            
            adaptive_ability = adaptive_metrics.get('final_ability_estimate', 0)
            adaptive_items = adaptive_metrics.get('total_items_administered', 0)
            adaptive_convergence = adaptive_metrics.get('convergence_achieved', False)
            
            st.markdown(f"""
            <div class="adaptive-card">
                <h4>📊 Results</h4>
                <p><strong>Ability:</strong> {adaptive_ability:.2f}</p>
                <p><strong>Items Used:</strong> {adaptive_items}</p>
                <p><strong>Converged:</strong> {'✅ Yes' if adaptive_convergence else '❌ No'}</p>
                <p><strong>Efficiency:</strong> {1.0 - adaptive_items/15:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Static Evaluation")
            static_results = static_data.get('evaluation_results', {})
            
            if static_results:
                static_performance = np.mean(list(static_results.values())) if static_results else 0
                static_items = 15  # Assumed static evaluation uses all tasks
                
                st.markdown(f"""
                <div class="info-card">
                    <h4>📊 Results</h4>
                    <p><strong>Performance:</strong> {static_performance:.3f}</p>
                    <p><strong>Items Used:</strong> {static_items}</p>
                    <p><strong>Converged:</strong> N/A (Fixed)</p>
                    <p><strong>Efficiency:</strong> 0.000 (Baseline)</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Static evaluation results not available")
        
        # Efficiency comparison chart
        if adaptive_data and static_data:
            comparison_data = {
                'Evaluation Type': ['Adaptive', 'Static'],
                'Items Used': [adaptive_items, 15],
                'Efficiency': [1.0 - adaptive_items/15, 0.0],
                'Precision': [1.0 if adaptive_convergence else 0.5, 0.7]  # Assumed precision
            }
            
            df = pd.DataFrame(comparison_data)
            
            fig = px.bar(df, x='Evaluation Type', y=['Items Used', 'Efficiency', 'Precision'],
                        title="Adaptive vs Static Evaluation Comparison",
                        barmode='group')
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif adaptive_data:
        st.info("🎯 Only adaptive evaluation results available. Run static evaluation for comparison.")
    elif static_data:
        st.info("📊 Only static evaluation results available. Run adaptive evaluation to see the evolution!")

def get_agent_performance_data(data):
    """Transform task-based data into agent-based performance data"""
    # Try multiple data sources
    failure_scores = data.get('failure_scores', {})
    if not failure_scores:
        failure_scores = data.get('calibrated_scores', {})
    if not failure_scores:
        failure_scores = data.get('raw_scores', {})
        
    tasks_data = data.get('tasks', [])
    task_lookup = {task['id']: task for task in tasks_data}
    
    # Also check for adaptive tasks
    adaptive_base_tasks = data.get('adaptive_base_tasks', [])
    if adaptive_base_tasks:
        for task in adaptive_base_tasks:
            task_lookup[task['id']] = task
    
    # Get adaptive evaluation data for enhanced agent info - FIX: Check both possible locations
    adaptive_data = data.get('detailed_adaptive', {})
    if not adaptive_data:
        adaptive_data = data.get('adaptive_results', {})
    
    detailed_responses = adaptive_data.get('detailed_responses', [])
    
    # Debug print to console
    print(f"DEBUG: Found {len(detailed_responses)} detailed responses in adaptive data")
    if detailed_responses:
        print(f"DEBUG: Sample adaptive task ID: {detailed_responses[0].get('task_id', 'NO_ID')}")
    
    # Create lookup for adaptive responses by task_id
    adaptive_lookup = {}
    for response in detailed_responses:
        task_id = response.get('task_id', '')
        # Map adaptive task IDs to base task IDs - handle format like "adaptive_atomic_3_0.20"
        if 'adaptive_' in task_id:
            # Extract base task ID: "adaptive_atomic_3_0.20" -> "atomic_3"
            parts = task_id.replace('adaptive_', '').split('_')
            if len(parts) >= 2:
                base_task_id = f"{parts[0]}_{parts[1]}"  # e.g., "atomic_3"
            else:
                base_task_id = task_id.replace('adaptive_', '')
        else:
            base_task_id = task_id
            
        if base_task_id not in adaptive_lookup:
            adaptive_lookup[base_task_id] = []
        adaptive_lookup[base_task_id].append(response)
    
    # Debug print the mapping
    print(f"DEBUG: Adaptive lookup keys: {list(adaptive_lookup.keys())}")
    
    # Define better agent names based on task content
    def generate_agent_name(task_id, task_info):
        """Generate descriptive agent names based on task content"""
        task_descriptions = {
            'atomic_1': '🧮 Math Calculator',
            'atomic_2': '📄 JSON Parser', 
            'atomic_3': '🌡️ Unit Converter',
            'compositional_1': '🌤️ Weather API Bot',
            'compositional_2': '📊 Data Analyst',
            'compositional_3': '🛒 Inventory Checker',
            'end2end_1': '📚 Research Assistant',
            'end2end_2': '🔧 Tech Support Bot',
            'end2end_3': '✈️ Travel Planner'
        }
        
        # Return predefined name if available, otherwise generate from task info
        if task_id in task_descriptions:
            return task_descriptions[task_id]
        
        # Handle adaptive task IDs
        if 'adaptive' in task_id:
            base_id = task_id.replace('adaptive_', '').split('_')[0] + '_' + task_id.replace('adaptive_', '').split('_')[1]
            if base_id in task_descriptions:
                return f"{task_descriptions[base_id]} (Adaptive)"
        
        # Fallback: generate name from task description or prompt
        description = task_info.get('description', '')
        prompt = task_info.get('prompt', '')
        
        if 'arithmetic' in description.lower() or 'compute' in prompt.lower():
            return '🧮 Calculator Agent'
        elif 'json' in description.lower() or 'json' in prompt.lower():
            return '📄 JSON Agent'
        elif 'temperature' in prompt.lower() or 'convert' in description.lower():
            return '🌡️ Converter Agent'
        elif 'weather' in prompt.lower():
            return '🌤️ Weather Agent'
        elif 'csv' in prompt.lower() or 'data' in description.lower():
            return '📊 Data Agent'
        elif 'shopping' in prompt.lower() or 'inventory' in prompt.lower():
            return '🛒 Shopping Agent'
        elif 'paper' in prompt.lower() or 'research' in description.lower():
            return '📚 Research Agent'
        elif 'router' in prompt.lower() or 'support' in description.lower():
            return '🔧 Support Agent'
        elif 'itinerary' in prompt.lower() or 'plan' in description.lower():
            return '✈️ Planning Agent'
        else:
            # Generic fallback based on tier
            tier = task_info.get('tier', 'unknown')
            if tier == 'atomic':
                return f'⚛️ {task_id.replace("_", "-").title()}'
            elif tier == 'compositional':
                return f'🔗 {task_id.replace("_", "-").title()}'
            elif tier == 'end-to-end':
                return f'🎯 {task_id.replace("_", "-").title()}'
            else:
                return f'🤖 {task_id.replace("_", "-").title()}'
    
    # Transform data: treat each task as an agent
    agent_performance = {}
    
    for judge, judge_scores in failure_scores.items():
        if isinstance(judge_scores, dict):
            for task_id, task_scores in judge_scores.items():
                if isinstance(task_scores, dict):
                    # Clean task_id for lookup
                    clean_task_id = task_id.replace('adaptive_', '') if 'adaptive_' in task_id else task_id
                    
                    if task_id not in agent_performance:
                        task_info = task_lookup.get(clean_task_id, task_lookup.get(task_id, {}))
                        adaptive_info = adaptive_lookup.get(task_id, [])  # Direct lookup first
                        if not adaptive_info:
                            adaptive_info = adaptive_lookup.get(clean_task_id, [])  # Then clean lookup
                        
                        # Debug print for each agent
                        print(f"DEBUG: Agent {task_id} -> adaptive_info length: {len(adaptive_info)}")
                        
                        agent_performance[task_id] = {
                            'agent_name': generate_agent_name(task_id, task_info),
                            'task_type': task_info.get('tier', 'unknown'),  # Use tier as task_type
                            'task_tier': task_info.get('tier', 'unknown'),
                            'task_description': task_info.get('description', task_info.get('prompt', 'No description')[:100] + "..."),
                            'task_prompt': task_info.get('prompt', 'No prompt available')[:150] + "..." if len(task_info.get('prompt', '')) > 150 else task_info.get('prompt', 'No prompt available'),
                            'judge_scores': {},
                            'metrics': {},
                            'adaptive_info': adaptive_info  # Add adaptive information
                        }
                    
                    agent_performance[task_id]['judge_scores'][judge] = task_scores
                    
                    # Calculate average metrics for this agent
                    for metric, score in task_scores.items():
                        if metric not in agent_performance[task_id]['metrics']:
                            agent_performance[task_id]['metrics'][metric] = []
                        agent_performance[task_id]['metrics'][metric].append(score)
    
    # Calculate final averages for each agent
    for agent_id, agent_data in agent_performance.items():
        for metric, scores in agent_data['metrics'].items():
            agent_data['metrics'][metric] = np.mean(scores)
    
    return agent_performance

def create_agent_overview(agent_data):
    """Create interactive overview of all agents with modal details"""
    st.markdown('<h2 style="text-align: center;">🤖 Multi-Agent Performance Overview</h2>', unsafe_allow_html=True)
    
    if not agent_data:
        st.warning("No agent performance data available.")
        return
    
    # Initialize session state for modal
    if 'show_agent_modal' not in st.session_state:
        st.session_state.show_agent_modal = False
    if 'selected_agent' not in st.session_state:
        st.session_state.selected_agent = None
    
    # Create interactive agent cards
    num_agents = len(agent_data)
    
    # Sort agents by performance for better display
    sorted_agents = sorted(agent_data.items(), 
                          key=lambda x: np.mean(list(x[1]['metrics'].values())), 
                          reverse=True)
    
    # Display agents in rows of 3
    for i in range(0, len(sorted_agents), 3):
        cols = st.columns(3)
        
        for j, (agent_id, agent_info) in enumerate(sorted_agents[i:i+3]):
            with cols[j]:
                avg_performance = np.mean(list(agent_info['metrics'].values()))
                
                # Color based on performance
                if avg_performance >= 0.8:
                    card_class = "success-card"
                    performance_emoji = "🟢"
                elif avg_performance >= 0.6:
                    card_class = "info-card"
                    performance_emoji = "🟡"
                else:
                    card_class = "warning-card"
                    performance_emoji = "🔴"
                
                # Determine rank badge
                all_performances = [np.mean(list(info['metrics'].values())) for info in agent_data.values()]
                agent_rank = sorted(all_performances, reverse=True).index(avg_performance) + 1
                
                if agent_rank == 1:
                    rank_text = "🥇 #1"
                elif agent_rank == 2:
                    rank_text = "🥈 #2"
                elif agent_rank == 3:
                    rank_text = "🥉 #3"
                else:
                    rank_text = f"#{agent_rank}"
                
                # Create clickable card
                st.markdown(f"""
                <div class="{card_class}" style="margin-bottom: 1rem;">
                    <h4>🤖 {agent_info['agent_name']}</h4>
                    <h3>{avg_performance:.3f} {performance_emoji}</h3>
                    <p><strong>Rank:</strong> {rank_text}</p>
                    <p><strong>Type:</strong> {agent_info['task_type'].title()}</p>
                    <p><strong>Tier:</strong> {agent_info['task_tier'].title()}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Button to open modal
                if st.button(f"📊 View Details", key=f"view_agent_{agent_id}", use_container_width=True):
                    st.session_state.show_agent_modal = True
                    st.session_state.selected_agent = agent_id
                    st.rerun()
    
    # Modal dialog
    if st.session_state.show_agent_modal and st.session_state.selected_agent:
        # Get data from create_agent_overview if available
        if 'dashboard_data' in st.session_state:
            show_agent_modal(agent_data, st.session_state.selected_agent, st.session_state.dashboard_data)
        else:
            show_agent_modal(agent_data, st.session_state.selected_agent)

@st.dialog("Agent Details")
def show_agent_modal(agent_data, agent_id, data=None):
    """Show agent details in a modal dialog"""
    if agent_id not in agent_data:
        st.error("Agent not found!")
        return
    
    agent_info = agent_data[agent_id]
    avg_performance = np.mean(list(agent_info['metrics'].values()))
    
    # Modal header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"### 🤖 {agent_info['agent_name']}")
    with col2:
        st.metric("Performance", f"{avg_performance:.3f}")
    with col3:
        if st.button("❌ Close", key="close_modal"):
            st.session_state.show_agent_modal = False
            st.session_state.selected_agent = None
            st.rerun()
    
    st.markdown("---")
    
    # Check if adaptive information is available
    adaptive_info = agent_info.get('adaptive_info', [])
    has_adaptive = len(adaptive_info) > 0
    
    # Debug: Show adaptive status in modal
    st.markdown(f"**🔍 Debug Info:** Adaptive info length: {len(adaptive_info)}, Has adaptive: {has_adaptive}")
    
    # Tabs for different sections - add evolved prompts tab if adaptive data exists
    if has_adaptive:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Metrics", "📋 Task Details", "⚖️ Judge Scores", "🤖 Response", "🎯 Evolved Prompts"])
        st.info(f"✅ 5th tab should be visible! Adaptive info: {len(adaptive_info)} items")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics", "📋 Task Details", "⚖️ Judge Scores", "🤖 Response"])
        st.warning("❌ No adaptive data found for this agent")
    
    with tab1:
        st.subheader("📊 Performance Metrics")
        
        # Performance metrics in columns
        metric_cols = st.columns(3)
        metrics_list = list(agent_info['metrics'].items())
        
        for idx, (metric, score) in enumerate(metrics_list):
            with metric_cols[idx % 3]:
                # Color code the metrics
                if score >= 0.8:
                    st.metric(metric, f"{score:.3f}", delta="Excellent ✅")
                elif score >= 0.6:
                    st.metric(metric, f"{score:.3f}", delta="Good ⚠️")
                else:
                    st.metric(metric, f"{score:.3f}", delta="Needs Improvement ❌")
        
        # Performance radar chart for this agent
        if agent_info['metrics']:
            metrics = list(agent_info['metrics'].keys())
            values = list(agent_info['metrics'].values())
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=agent_info['agent_name'],
                line_color='#667eea'
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                title=f"{agent_info['agent_name']} Performance Profile",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"modal_radar_{agent_id}")
    
    with tab2:
        st.subheader("📋 Task Details")
        
        # Basic info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Task ID:** {agent_id}")
            st.info(f"**Task Type:** {agent_info['task_type'].title()}")
        with col2:
            st.info(f"**Task Tier:** {agent_info['task_tier'].title()}")
            
            # Calculate rank
            all_performances = [np.mean(list(info['metrics'].values())) for info in agent_data.values()]
            agent_rank = sorted(all_performances, reverse=True).index(avg_performance) + 1
            st.info(f"**Rank:** {agent_rank}/{len(agent_data)}")
        
        # Show adaptive enhancement status
        if has_adaptive:
            st.markdown("""
            <div class="adaptive-card">
                <h4>🎯 Adaptive Evaluation Enhanced</h4>
                <p>This agent was evaluated using adaptive difficulty calibration with evolved prompts!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Task description
        st.markdown("**📝 Task Description:**")
        st.write(agent_info['task_description'])
        
        # Task prompt
        st.markdown("**💬 Task Prompt:**")
        st.code(agent_info['task_prompt'], language="text")
    
    with tab3:
        st.subheader("⚖️ Judge Evaluation Breakdown")
        
        if agent_info.get('judge_scores'):
            # Judge comparison chart
            judge_names = list(agent_info['judge_scores'].keys())
            
            # Create comparison chart
            fig = go.Figure()
            
            for judge_name, judge_scores in agent_info['judge_scores'].items():
                metrics = list(judge_scores.keys())
                values = list(judge_scores.values())
                
                fig.add_trace(go.Bar(
                    name=judge_name,
                    x=metrics,
                    y=values,
                    text=[f"{v:.3f}" for v in values],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Judge Score Comparison",
                xaxis_title="Metrics",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"modal_judge_comparison_{agent_id}")
            
            # Detailed judge scores
            st.markdown("**Detailed Judge Scores:**")
            
            judge_data = []
            for judge, scores in agent_info['judge_scores'].items():
                for metric, score in scores.items():
                    judge_data.append({
                        'Judge': judge,
                        'Metric': metric,
                        'Score': f"{score:.3f}",
                        'Rating': "Excellent ✅" if score >= 0.8 else "Good ⚠️" if score >= 0.6 else "Poor ❌"
                    })
            
            if judge_data:
                judge_df = pd.DataFrame(judge_data)
                st.dataframe(judge_df, use_container_width=True, hide_index=True)
        else:
            st.info("No judge scores available for this agent.")
    
    with tab4:
        st.subheader("🤖 Agent Response")
        
        response_data = load_agent_response(agent_id)
        
        if response_data:
            # Response preview
            response_text = response_data.get('response', 'No response available')
            
            st.markdown("**🎯 Agent's Response:**")
            st.text_area("Full Response:", response_text, height=300, disabled=True, key=f"modal_response_{agent_id}")
            
            # Response metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                if response_data.get('timestamp'):
                    st.info(f"**Generated:** {response_data['timestamp']}")
            with col2:
                if response_data.get('model'):
                    st.info(f"**Model:** {response_data['model']}")
            with col3:
                st.info(f"**Length:** {len(response_text)} chars")
            
            # Enhanced features applied
            if response_data.get('enhanced_features_applied'):
                st.markdown("**🚀 Enhanced Features Applied:**")
                features = response_data['enhanced_features_applied']
                
                feature_cols = st.columns(4)
                feature_items = list(features.items())
                for i, (feature_name, enabled) in enumerate(feature_items):
                    with feature_cols[i % 4]:
                        icon = "✅" if enabled else "❌"
                        feature_display = feature_name.replace('_', ' ').title()
                        st.write(f"{icon} {feature_display}")
        else:
            st.info("No detailed response data available for this agent.")
    
    # Evolved Prompts tab (only if adaptive data exists)
    if has_adaptive:
        with tab5:
            st.subheader("🎯 Evolved Prompts & Adaptive Difficulty")
            
            if adaptive_info:
                st.markdown("""
                <div class="evolved-prompt-container">
                    <h4>🧬 Adaptive Evaluation Process</h4>
                    <p>This agent was evaluated using our advanced adaptive difficulty system with evolved prompts!</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show adaptive responses for this agent
                for i, response in enumerate(adaptive_info):
                    task_id = response.get('task_id', 'Unknown')
                    difficulty = response.get('difficulty', 0)
                    performance = response.get('performance', 0)
                    reasoning_steps = response.get('reasoning_steps', 0)
                    time_taken = response.get('time_taken', 0)
                    
                    with st.expander(f"🎯 Adaptive Response #{i+1}: {task_id}"):
                        # Adaptive metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Difficulty", f"{difficulty:.3f}")
                        with col2:
                            st.metric("Performance", f"{performance:.3f}")
                        with col3:
                            st.metric("Reasoning Steps", reasoning_steps)
                        with col4:
                            st.metric("Time Taken", f"{time_taken:.2f}s")
                        
                        st.markdown("---")
                        
                        # Show the base prompt and how it was evolved
                        base_task_id = None
                        
                        # Extract base task ID
                        if 'adaptive_' in task_id:
                            parts = task_id.replace('adaptive_', '').split('_')
                            if len(parts) >= 2:
                                base_task_id = f"{parts[0]}_{parts[1]}"
                        
                        # Load base task prompt if data is available
                        if base_task_id and data:
                            # Try to load the base task from adaptive_base_tasks
                            adaptive_base_tasks = data.get('adaptive_base_tasks', [])
                            base_prompt = None
                            for task in adaptive_base_tasks:
                                if task.get('id') == base_task_id:
                                    base_prompt = task.get('prompt', '')
                                    break
                            
                            actual_evolved_prompt = response.get('evolved_prompt', '')
                            
                            if base_prompt and actual_evolved_prompt:
                                st.markdown("**🔄 Prompt Evolution Analysis:**")
                                
                                # Create side-by-side comparison
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**📝 Original Base Prompt:**")
                                    st.text_area(
                                        f"Base Task ({base_task_id}):",
                                        base_prompt,
                                        height=120,
                                        key=f"base_prompt_{agent_id}_{base_task_id}_{response.get('task_id', 'unknown')}_{i}"
                                    )
                                
                                with col2:
                                    st.markdown("**🎯 Evolved Prompt (Used in Evaluation):**")
                                    st.text_area(
                                        f"Evolved Task ({task_id}):",
                                        actual_evolved_prompt,
                                        height=120,
                                        key=f"evolved_prompt_{agent_id}_{task_id}_{response.get('task_id', 'unknown')}_{i}"
                                    )
                                
                                # Highlight the changes
                                st.markdown("**🔍 What Changed:**")
                                
                                # Find the differences
                                if actual_evolved_prompt != base_prompt:
                                    # Check if it's a prefix addition
                                    if actual_evolved_prompt.endswith(base_prompt):
                                        prefix = actual_evolved_prompt.replace(base_prompt, '').strip()
                                        st.markdown(f"""
                                        <div class="evolved-characteristic">
                                            <strong>✨ Evolution Type:</strong> Prefix Addition<br>
                                            <strong>🎯 Added Prefix:</strong> "{prefix}"<br>
                                            <strong>📊 Impact:</strong> Increased complexity and analytical focus
                                        </div>
                                        """, unsafe_allow_html=True)
                                    elif base_prompt in actual_evolved_prompt:
                                        # It's embedded with modifications
                                        st.markdown(f"""
                                        <div class="evolved-characteristic">
                                            <strong>✨ Evolution Type:</strong> Enhanced Framework<br>
                                            <strong>🎯 Enhancement:</strong> Added analytical structure and optimization cues<br>
                                            <strong>📊 Impact:</strong> Improved reasoning guidance and evaluation depth
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        # Completely different
                                        st.markdown(f"""
                                        <div class="evolved-characteristic">
                                            <strong>✨ Evolution Type:</strong> Complete Restructure<br>
                                            <strong>🎯 Change:</strong> Fundamental prompt modification<br>
                                            <strong>📊 Impact:</strong> New approach to the same underlying task
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.warning("⚠️ No changes detected between base and evolved prompts")
                                
                                # Show evolution metrics
                                st.markdown("**📈 Evolution Metrics:**")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("🎚️ Difficulty Level", f"{difficulty:.1%}")
                                with col2:
                                    st.metric("🧠 Reasoning Steps", reasoning_steps)
                                with col3:
                                    st.metric("📏 Response Length", f"{response.get('response_length', 0):,} chars")
                                with col4:
                                    st.metric("⏱️ Time Taken", f"{time_taken:.1f}s")
                                
                                # Character count comparison
                                base_length = len(base_prompt)
                                evolved_length = len(actual_evolved_prompt)
                                length_change = evolved_length - base_length
                                
                                st.markdown(f"""
                                <div class="adaptive-prompt-text">
                                    <strong>📊 Prompt Statistics:</strong><br>
                                    • Base prompt: {base_length} characters<br>
                                    • Evolved prompt: {evolved_length} characters<br>
                                    • Change: {'+' if length_change > 0 else ''}{length_change} characters ({'+' if length_change > 0 else ''}{(length_change/base_length)*100:.1f}%)
                                </div>
                                """, unsafe_allow_html=True)
                                
                            else:
                                if not base_prompt:
                                    st.warning(f"Base prompt for {base_task_id} not found in adaptive base tasks.")
                                if not actual_evolved_prompt:
                                    st.warning("Evolved prompt data not available for this response.")
                        else:
                            # Fallback - show generic evolved prompt characteristics
                            st.markdown("**🎯 Evolved Prompt Characteristics:**")
                            st.markdown(f"""
                            <div class="adaptive-prompt-text">
                                <div class="evolved-characteristic">
                                    <strong>Adaptive Difficulty Level:</strong> {difficulty:.3f}
                                </div>
                                <div class="evolved-characteristic">
                                    <strong>IRT-Based Calibration:</strong> Dynamic scaling based on agent ability
                                </div>
                                <div class="evolved-characteristic">
                                    <strong>Reasoning Complexity:</strong> {reasoning_steps} cognitive steps required
                                </div>
                                <div class="evolved-characteristic">
                                    <strong>Response Optimization:</strong> Tailored for optimal discrimination
                                </div>
                                <div class="evolved-characteristic">
                                    <strong>Dynamic Generation:</strong> Real-time prompt evolution based on performance
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Summary of adaptive process
                st.markdown("---")
                st.markdown("**📈 Adaptive Process Summary:**")
                
                avg_difficulty = np.mean([r.get('difficulty', 0) for r in adaptive_info])
                avg_performance = np.mean([r.get('performance', 0) for r in adaptive_info])
                total_reasoning = sum([r.get('reasoning_steps', 0) for r in adaptive_info])
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                with summary_col1:
                    st.metric("Avg Difficulty", f"{avg_difficulty:.3f}")
                with summary_col2:
                    st.metric("Avg Performance", f"{avg_performance:.3f}")
                with summary_col3:
                    st.metric("Total Reasoning Steps", total_reasoning)
                
                st.markdown("""
                <div class="trajectory-insight">
                    <h4>🎯 Adaptive Insights</h4>
                    <p>This agent underwent dynamic difficulty calibration with:</p>
                    <ul>
                        <li><strong>Real-time difficulty adjustment</strong> based on performance</li>
                        <li><strong>IRT-based item response modeling</strong> for precise ability estimation</li>
                        <li><strong>Evolved prompt generation</strong> with complexity scaling</li>
                        <li><strong>Optimized evaluation efficiency</strong> through adaptive stopping</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No evolved prompt data available for this agent.")

def load_agent_response(agent_id):
    """Load detailed agent response data for a specific agent/task"""
    try:
        # Try to load from enhanced evaluation results
        if os.path.exists('data/enhanced_evaluation_results.json'):
            with open('data/enhanced_evaluation_results.json', 'r') as f:
                enhanced_data = json.load(f)
            
            # Look for agent outputs in the enhanced results
            evaluation_results = enhanced_data.get('evaluation_results', {})
            if 'agent_outputs' in evaluation_results:
                return evaluation_results['agent_outputs'].get(agent_id)
        
        # Fallback to agent outputs file
        if os.path.exists('data/agent_outputs.json'):
            with open('data/agent_outputs.json', 'r') as f:
                agent_outputs = json.load(f)
            return agent_outputs.get(agent_id)
        
        # Fallback to enhanced agent outputs
        if os.path.exists('data/enhanced_agent_outputs.json'):
            with open('data/enhanced_agent_outputs.json', 'r') as f:
                enhanced_outputs = json.load(f)
            return enhanced_outputs.get(agent_id)
        
    except Exception as e:
        st.error(f"Error loading agent response data: {e}")
    
    return None

def create_agent_comparison_radar(agent_data):
    """Create radar chart comparing all agents"""
    if not agent_data:
        return None
    
    fig = go.Figure()
    
    # Get all metrics
    all_metrics = set()
    for agent_info in agent_data.values():
        all_metrics.update(agent_info['metrics'].keys())
    all_metrics = sorted(list(all_metrics))
    
    # Add trace for each agent
    colors = ['#667eea', '#764ba2', '#4CAF50', '#ff9800', '#f44336', '#9C27B0', '#607D8B', '#795548', '#FF5722']
    
    for i, (agent_id, agent_info) in enumerate(agent_data.items()):
        values = [agent_info['metrics'].get(metric, 0) for metric in all_metrics]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=all_metrics,
            fill='toself',
            name=agent_info['agent_name'],
            line_color=colors[i % len(colors)],
            fillcolor=f"rgba{tuple(list(int(colors[i % len(colors)][j:j+2], 16) for j in (1, 3, 5)) + [0.1])}"
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Multi-Agent Performance Comparison",
        title_x=0.5,
        height=600
    )
    
    return fig

def create_agent_performance_matrix(agent_data):
    """Create performance matrix heatmap for all agents"""
    if not agent_data:
        return None
    
    # Prepare data for heatmap
    agents = []
    metrics = []
    scores = []
    
    all_metrics = set()
    for agent_info in agent_data.values():
        all_metrics.update(agent_info['metrics'].keys())
    all_metrics = sorted(list(all_metrics))
    
    matrix_data = []
    agent_names = []
    
    for agent_id, agent_info in agent_data.items():
        agent_names.append(agent_info['agent_name'])
        row = [agent_info['metrics'].get(metric, 0) for metric in all_metrics]
        matrix_data.append(row)
    
    fig = px.imshow(
        matrix_data,
        x=all_metrics,
        y=agent_names,
        color_continuous_scale='RdYlGn',
        aspect='auto',
        title="Agent Performance Matrix",
        labels=dict(x="Metrics", y="Agents", color="Score")
    )
    
    fig.update_layout(height=400)
    
    return fig

def create_agent_ranking(agent_data):
    """Create agent ranking visualization"""
    if not agent_data:
        return None
    
    # Calculate overall scores for ranking
    ranking_data = []
    for agent_id, agent_info in agent_data.items():
        overall_score = np.mean(list(agent_info['metrics'].values()))
        ranking_data.append({
            'Agent': agent_info['agent_name'],
            'Overall_Score': overall_score,
            'Task_Type': agent_info['task_type'],
            'Task_Tier': agent_info['task_tier']
        })
    
    # Sort by overall score
    ranking_data.sort(key=lambda x: x['Overall_Score'], reverse=True)
    df_ranking = pd.DataFrame(ranking_data)
    
    fig = px.bar(
        df_ranking,
        x='Agent',
        y='Overall_Score',
        color='Task_Type',
        title="Agent Performance Ranking",
        labels={'Overall_Score': 'Average Performance Score'}
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig

def create_judge_agent_analysis(data, agent_data):
    """Create analysis of how different judges rate different agents"""
    failure_scores = data.get('failure_scores', {})
    if not failure_scores or not agent_data:
        return None, None
    
    # Prepare data for judge-agent analysis
    judge_agent_data = []
    
    for judge, judge_scores in failure_scores.items():
        for task_id, task_scores in judge_scores.items():
            if task_id in agent_data:
                agent_name = agent_data[task_id]['agent_name']
                avg_score = np.mean(list(task_scores.values()))
                judge_agent_data.append({
                    'Judge': judge,
                    'Agent': agent_name,
                    'Average_Score': avg_score,
                    'Task_Type': agent_data[task_id]['task_type']
                })
    
    if not judge_agent_data:
        return None, None
    
    df_judge_agent = pd.DataFrame(judge_agent_data)
    
    # Judge-Agent heatmap
    pivot_df = df_judge_agent.pivot_table(values='Average_Score', index='Agent', columns='Judge', aggfunc='mean')
    
    fig_heatmap = px.imshow(
        pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale='RdYlGn',
        title="Judge-Agent Agreement Matrix"
    )
    
    # Judge bias analysis
    fig_bias = px.box(
        df_judge_agent,
        x='Judge',
        y='Average_Score',
        color='Judge',
        title="Judge Scoring Patterns Across Agents"
    )
    
    return fig_heatmap, fig_bias

def create_agent_specialization_analysis(agent_data):
    """Analyze agent specialization by task type and tier"""
    if not agent_data:
        return None, None
    
    # Prepare specialization data
    spec_data = []
    for agent_id, agent_info in agent_data.items():
        overall_score = np.mean(list(agent_info['metrics'].values()))
        spec_data.append({
            'Agent': agent_info['agent_name'],
            'Task_Type': agent_info['task_type'],
            'Task_Tier': agent_info['task_tier'],
            'Performance': overall_score
        })
    
    df_spec = pd.DataFrame(spec_data)
    
    # Task type performance
    fig_type = px.box(
        df_spec,
        x='Task_Type',
        y='Performance',
        color='Task_Type',
        title="Performance by Task Type",
        points="all"
    )
    
    # Task tier performance
    fig_tier = px.scatter(
        df_spec,
        x='Task_Tier',
        y='Performance',
        color='Task_Type',
        size='Performance',
        title="Performance by Task Tier",
        hover_data=['Agent']
    )
    
    return fig_type, fig_tier

def main():
    """Main dashboard function with comprehensive adaptive evaluation features"""
    # Header
    st.markdown('<h1 class="main-header">🎯 AgEval Adaptive Evaluation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Evolution, Prompts, IRT Analysis & Results</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading evaluation data..."):
        data = load_evaluation_data()
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Evaluation mode selection
    st.sidebar.subheader("📊 Evaluation Mode")
    available_modes = []
    if data.get('adaptive_results'):
        available_modes.append("🎯 Adaptive Evaluation")
    if data.get('static_results') or data.get('enhanced_results'):
        available_modes.append("📊 Static Evaluation")
    if len(available_modes) > 1:
        available_modes.append("⚖️ Comparison Mode")
    
    if not available_modes:
        st.error("No evaluation data found. Please run an evaluation first using:")
        st.code("python run_enhanced_evaluation.py")
        return
    
    selected_mode = st.sidebar.selectbox(
        "Select evaluation mode to view:",
        available_modes,
        index=0
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Data status
    st.sidebar.subheader("📊 Data Status")
    data_status = {
        "Adaptive Results": bool(data.get('adaptive_results')),
        "Static Results": bool(data.get('static_results') or data.get('enhanced_results')),
        "Trajectory Plot": os.path.exists("reports/adaptive_evaluation_trajectory.png"),
        "IRT Analysis": bool(data.get('detailed_adaptive', {}).get('irt_response_history')),
        "Evolved Prompts": bool(data.get('detailed_adaptive', {}).get('detailed_responses'))
    }
    
    for status, available in data_status.items():
        if available:
            st.sidebar.success(f"✅ {status}")
        else:
            st.sidebar.error(f"❌ {status}")
    
    # Debug: Show adaptive data info
    adaptive_data = data.get('adaptive_results', {}) or data.get('detailed_adaptive', {})
    detailed_responses = adaptive_data.get('detailed_responses', [])
    if detailed_responses:
        st.sidebar.subheader("🎯 Adaptive Debug Info")
        st.sidebar.info(f"📊 {len(detailed_responses)} adaptive responses found")
        
        # Show sample task IDs for debugging
        sample_task_ids = [r.get('task_id', '') for r in detailed_responses[:3]]
        st.sidebar.write("Sample Task IDs:")
        for task_id in sample_task_ids:
            st.sidebar.code(task_id, language="text")
        
        # Debug: Show the adaptive lookup mapping
        st.sidebar.write("Mapped Base Task IDs:")
        adaptive_lookup_debug = {}
        for response in detailed_responses:
            task_id = response.get('task_id', '')
            if 'adaptive_' in task_id:
                parts = task_id.replace('adaptive_', '').split('_')
                if len(parts) >= 2:
                    base_task_id = f"{parts[0]}_{parts[1]}"
                else:
                    base_task_id = task_id.replace('adaptive_', '')
            else:
                base_task_id = task_id
            
            if base_task_id not in adaptive_lookup_debug:
                adaptive_lookup_debug[base_task_id] = 0
            adaptive_lookup_debug[base_task_id] += 1
        
        for base_id, count in adaptive_lookup_debug.items():
            st.sidebar.write(f"• {base_id}: {count} responses")
    
    # Main content based on selected mode
    if selected_mode == "🎯 Adaptive Evaluation":
        display_adaptive_evaluation_mode(data)
    elif selected_mode == "📊 Static Evaluation":
        display_static_evaluation_mode(data)
    elif selected_mode == "⚖️ Comparison Mode":
        display_comparison_mode(data)

def display_adaptive_evaluation_mode(data):
    """Display comprehensive adaptive evaluation results"""
    adaptive_data = data.get('adaptive_results', {})
    detailed_adaptive = data.get('detailed_adaptive', {})
    
    if not adaptive_data and not detailed_adaptive:
        st.error("No adaptive evaluation data available. Run adaptive evaluation first:")
        st.code("python run_enhanced_evaluation.py")
        return
    
    # Use detailed data if available, fallback to basic adaptive data
    primary_data = detailed_adaptive if detailed_adaptive else adaptive_data
    
    # 1. Adaptive Evaluation Overview
    create_adaptive_evaluation_overview(primary_data)
    
    st.markdown("---")
    
    # 2. Evolved Prompts & Task Generation
    create_evolved_prompts_section(primary_data)
    
    st.markdown("---")
    
    # 3. Interactive Trajectory Analysis
    st.markdown('<h2 class="adaptive-header">📈 Interactive Trajectory Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Interactive trajectory plot
        trajectory_fig = create_adaptive_trajectory_plot(primary_data)
        if trajectory_fig:
            st.plotly_chart(trajectory_fig, use_container_width=True, key="adaptive_trajectory")
        else:
            st.info("No trajectory data available for interactive plot.")
    
    with col2:
        # Display static trajectory image if available
        display_trajectory_plot()
    
    st.markdown("---")
    
    # 4. Item Response Theory Analysis
    create_irt_analysis_section(primary_data)
    
    st.markdown("---")
    
    # 5. Enhanced Agent Analysis - Use original agent data with adaptive enhancements
    with st.spinner("Analyzing agent performance..."):
        agent_data = get_agent_performance_data(data)  # This now includes adaptive info
        # Store data in session state for modal access
        st.session_state.dashboard_data = data
    
    if agent_data:
        st.markdown('<h2 class="adaptive-header">🤖 Agent Performance with Adaptive Enhancement</h2>', unsafe_allow_html=True)
        
        # Show how many agents have adaptive data
        adaptive_agents = [agent_id for agent_id, agent_info in agent_data.items() if agent_info.get('adaptive_info')]
        
        if adaptive_agents:
            st.markdown(f"""
            <div class="adaptive-card">
                <h4>🎯 Adaptive Enhancement Status</h4>
                <p><strong>{len(adaptive_agents)}</strong> out of <strong>{len(agent_data)}</strong> agents were enhanced with adaptive evaluation!</p>
                <p>Click "📊 View Details" on any agent to see evolved prompts and adaptive difficulty calibration.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-card">
                <h4>⚠️ No Adaptive Data Found</h4>
                <p>No agents found with adaptive evaluation data. This could be a data mapping issue.</p>
                <p><strong>Total agents:</strong> {len(agent_data)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Debug: Show which agent IDs we have
            st.subheader("🔍 Debug: Agent IDs")
            agent_ids = list(agent_data.keys())
            st.write(f"Agent IDs found: {agent_ids}")
            
            # Show sample agent adaptive_info
            if agent_data:
                sample_agent_id = list(agent_data.keys())[0]
                sample_adaptive_info = agent_data[sample_agent_id].get('adaptive_info', [])
                st.write(f"Sample agent '{sample_agent_id}' adaptive_info length: {len(sample_adaptive_info)}")
        
        # Agent overview
        create_agent_overview(agent_data)
        
        st.markdown("---")
        
        # Agent comparison visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Agent Performance Radar")
            radar_fig = create_agent_comparison_radar(agent_data)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True, key="adaptive_radar")
        
        with col2:
            st.subheader("📊 Agent Performance Matrix")
            matrix_fig = create_agent_performance_matrix(agent_data)
            if matrix_fig:
                st.plotly_chart(matrix_fig, use_container_width=True, key="adaptive_matrix")
    
    # 6. Detailed Data Tables
    st.markdown("---")
    st.subheader("📊 Detailed Adaptive Data")
    
    tab1, tab2, tab3 = st.tabs(["📈 Trajectory Data", "🎯 Task Evolution", "📊 IRT Parameters"])
    
    with tab1:
        # Display trajectory data
        detailed_responses = primary_data.get('detailed_responses', [])
        if detailed_responses:
            df_responses = pd.DataFrame(detailed_responses)
            st.dataframe(df_responses, use_container_width=True)
        else:
            st.info("No trajectory data available.")
    
    with tab2:
        # Display evolved task characteristics
        if detailed_responses:
            # Group by task type for analysis
            task_evolution_data = []
            for response in detailed_responses:
                task_id = response.get('task_id', '')
                task_type = 'atomic' if 'atomic' in task_id else 'compositional' if 'compositional' in task_id else 'end2end'
                task_evolution_data.append({
                    'Task_ID': task_id,
                    'Task_Type': task_type,
                    'Difficulty': response.get('difficulty', 0),
                    'Performance': response.get('performance', 0),
                    'Reasoning_Steps': response.get('reasoning_steps', 0),
                    'Response_Length': response.get('response_length', 0),
                    'Time_Taken': response.get('time_taken', 0)
                })
            
            df_evolution = pd.DataFrame(task_evolution_data)
            st.dataframe(df_evolution, use_container_width=True)
        else:
            st.info("No task evolution data available.")
    
    with tab3:
        # Display IRT parameters
        irt_history = primary_data.get('irt_response_history', [])
        if irt_history:
            df_irt = pd.DataFrame(irt_history)
            st.dataframe(df_irt, use_container_width=True)
        else:
            st.info("No IRT parameter data available.")

def display_static_evaluation_mode(data):
    """Display traditional static evaluation results"""
    # Transform to agent-based view
    with st.spinner("Analyzing agent performance..."):
        agent_data = get_agent_performance_data(data)
    
    if not agent_data:
        st.error("No static evaluation data could be extracted from the evaluation results.")
        return
    
    # Agent selection
    st.sidebar.subheader("🤖 Agent Filter")
    selected_agents = st.sidebar.multiselect(
        "Select agents to analyze:",
        options=list(agent_data.keys()),
        default=list(agent_data.keys()),
        format_func=lambda x: agent_data[x]['agent_name']
    )
    
    # Filter agent data
    filtered_agent_data = {k: v for k, v in agent_data.items() if k in selected_agents}
    
    # Agent overview
    create_agent_overview(filtered_agent_data)
    
    st.markdown("---")
    
    # Multi-agent comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Multi-Agent Performance Radar")
        radar_fig = create_agent_comparison_radar(filtered_agent_data)
        if radar_fig:
            st.plotly_chart(radar_fig, use_container_width=True, key="static_radar_chart")
        else:
            st.info("No performance data available for radar chart.")
    
    with col2:
        st.subheader("📊 Agent Performance Matrix")
        matrix_fig = create_agent_performance_matrix(filtered_agent_data)
        if matrix_fig:
            st.plotly_chart(matrix_fig, use_container_width=True, key="static_matrix_chart")
        else:
            st.info("No performance matrix data available.")
    
    st.markdown("---")
    
    # Agent ranking and specialization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Agent Performance Ranking")
        ranking_fig = create_agent_ranking(filtered_agent_data)
        if ranking_fig:
            st.plotly_chart(ranking_fig, use_container_width=True, key="static_ranking_chart")
        else:
            st.info("No ranking data available.")
    
    with col2:
        st.subheader("🎯 Agent Specialization")
        type_fig, tier_fig = create_agent_specialization_analysis(filtered_agent_data)
        if type_fig:
            st.plotly_chart(type_fig, use_container_width=True, key="static_specialization_chart")
        else:
            st.info("No specialization data available.")
    
    st.markdown("---")
    
    # Judge-Agent analysis
    st.subheader("⚖️ Judge-Agent Analysis")
    heatmap_fig, bias_fig = create_judge_agent_analysis(data, filtered_agent_data)
    
    if heatmap_fig and bias_fig:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(heatmap_fig, use_container_width=True, key="static_judge_heatmap_chart")
        with col2:
            st.plotly_chart(bias_fig, use_container_width=True, key="static_judge_bias_chart")
    else:
        st.info("No judge-agent analysis data available.")
    
    st.markdown("---")
    
    # Self-evaluation analysis per agent
    st.subheader("🔄 Agent Self-Evaluation Analysis")
    enhanced_results = data.get('enhanced_results', {})
    self_eval_data = enhanced_results.get('self_evaluation_analysis', {})
    
    if self_eval_data:
        # Create self-evaluation summary for each agent
        self_eval_summary = []
        for task_id, eval_data in self_eval_data.items():
            if task_id in filtered_agent_data:
                agent_name = filtered_agent_data[task_id]['agent_name']
                self_eval_summary.append({
                    'Agent': agent_name,
                    'Iterations': eval_data.get('iterations_used', 0),
                    'Converged': eval_data.get('converged', False),
                    'Confidence': eval_data.get('final_evaluation', {}).get('overall_confidence', 0),
                    'Task_Type': filtered_agent_data[task_id]['task_type']
                })
        
        if self_eval_summary:
            df_self_eval = pd.DataFrame(self_eval_summary)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_iterations = px.bar(
                    df_self_eval,
                    x='Agent',
                    y='Iterations',
                    color='Converged',
                    title="Self-Evaluation Iterations by Agent",
                    color_discrete_map={True: 'green', False: 'red'}
                )
                fig_iterations.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_iterations, use_container_width=True, key="static_self_eval_iterations_chart")
            
            with col2:
                fig_confidence = px.bar(
                    df_self_eval,
                    x='Agent',
                    y='Confidence',
                    color='Task_Type',
                    title="Self-Evaluation Confidence by Agent"
                )
                fig_confidence.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_confidence, use_container_width=True, key="static_self_eval_confidence_chart")
    else:
        st.info("No self-evaluation data available.")
    
    st.markdown("---")
    
    # Detailed agent data tables
    st.subheader("📊 Detailed Agent Data")
    
    tab1, tab2, tab3 = st.tabs(["Agent Performance", "Judge Scores", "Agent Details"])
    
    with tab1:
        if filtered_agent_data:
            # Create comprehensive agent performance table
            perf_data = []
            for agent_id, agent_info in filtered_agent_data.items():
                row = {
                    'Agent': agent_info['agent_name'],
                    'Task_Type': agent_info['task_type'],
                    'Task_Tier': agent_info['task_tier'],
                    'Overall_Score': np.mean(list(agent_info['metrics'].values()))
                }
                row.update(agent_info['metrics'])
                perf_data.append(row)
            
            df_performance = pd.DataFrame(perf_data)
            st.dataframe(df_performance, use_container_width=True)
        else:
            st.info("No agent performance data available.")
    
    with tab2:
        if filtered_agent_data:
            # Create detailed judge scores table
            judge_data = []
            for agent_id, agent_info in filtered_agent_data.items():
                for judge, scores in agent_info['judge_scores'].items():
                    for metric, score in scores.items():
                        judge_data.append({
                            'Agent': agent_info['agent_name'],
                            'Judge': judge,
                            'Metric': metric,
                            'Score': score,
                            'Task_Type': agent_info['task_type']
                        })
            
            if judge_data:
                df_judges = pd.DataFrame(judge_data)
                st.dataframe(df_judges, use_container_width=True)
        else:
            st.info("No judge scores available.")
    
    with tab3:
        if filtered_agent_data:
            # Show agent details
            for agent_id, agent_info in filtered_agent_data.items():
                with st.expander(f"{agent_info['agent_name']} Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Agent Type:** {agent_info['agent_name']}")
                        st.write(f"**Task Tier:** {agent_info['task_tier'].title()}")
                        st.write(f"**Overall Score:** {np.mean(list(agent_info['metrics'].values())):.3f}")
                        st.write(f"**Task ID:** {agent_id}")
                    with col2:
                        st.write("**Performance Metrics:**")
                        for metric, score in agent_info['metrics'].items():
                            # Color code the metrics
                            if score >= 0.8:
                                st.write(f"- ✅ {metric}: {score:.3f}")
                            elif score >= 0.6:
                                st.write(f"- ⚠️ {metric}: {score:.3f}")
                            else:
                                st.write(f"- ❌ {metric}: {score:.3f}")
                    
                    st.write("**📋 Task Description:**")
                    st.write(agent_info['task_description'])
                    
                    st.write("**💬 Task Prompt:**")
                    st.code(agent_info['task_prompt'], language="text")

def display_comparison_mode(data):
    """Display comparison between adaptive and static evaluation"""
    st.markdown('<h2 class="adaptive-header">⚖️ Adaptive vs Static Evaluation Comparison</h2>', unsafe_allow_html=True)
    
    # Show comparison analysis
    create_comparison_analysis(data)
    
    # Additional comparison metrics if both are available
    adaptive_data = data.get('adaptive_results', {})
    static_data = data.get('static_results', {}) or data.get('enhanced_results', {})
    
    if adaptive_data and static_data:
        st.markdown("---")
        
        # Detailed comparison table
        st.subheader("📊 Detailed Comparison Metrics")
        
        # Extract metrics from both evaluations
        adaptive_metrics = adaptive_data.get('adaptive_evaluation_results', {})
        if isinstance(adaptive_metrics, dict) and 'adaptive_evaluation_results' in adaptive_metrics:
            adaptive_metrics = adaptive_metrics['adaptive_evaluation_results']
        
        static_results = static_data.get('evaluation_results', {})
        
        comparison_table = {
            "Metric": [
                "Evaluation Items",
                "Convergence",
                "Efficiency",
                "Precision",
                "Time Saved",
                "Adaptability"
            ],
            "Adaptive Evaluation": [
                f"{adaptive_metrics.get('total_items_administered', 0)} items",
                "✅ Yes" if adaptive_metrics.get('convergence_achieved', False) else "❌ No",
                f"{1.0 - adaptive_metrics.get('total_items_administered', 15)/15:.1%}",
                f"SE: {adaptive_metrics.get('ability_standard_error', 0):.3f}",
                f"{15 - adaptive_metrics.get('total_items_administered', 15)} items saved",
                "🎯 Dynamic difficulty"
            ],
            "Static Evaluation": [
                "15 items (fixed)",
                "N/A (fixed set)",
                "0% (baseline)",
                "Standard",
                "0 items saved",
                "❌ No adaptation"
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_table)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)

def get_adaptive_agent_performance_data(adaptive_data):
    """Extract agent performance data from adaptive evaluation results"""
    # This function is now deprecated - we use the enhanced get_agent_performance_data instead
    return {}

if __name__ == "__main__":
    main() 