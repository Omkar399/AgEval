# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AgEval is a comprehensive three-judge AI evaluation system that uses multiple LLM judges (GPT-4, Claude, Gemini) to assess AI agent performance. The system implements a robust 9-phase pipeline with bias calibration, inter-judge agreement analysis, and confidence-based scoring.

## Common Commands

### Setup and Installation
```bash
# Quick setup (creates venv, installs deps, creates config)
./scripts/quick_start.sh

# Manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p logs reports data/cache
```

### Running Evaluations
```bash
# Full evaluation pipeline
python run_evaluation.py

# Run specific phase (1-8)
python run_evaluation.py --phase 3

# Custom configuration
python run_evaluation.py --config custom_config.yaml --tasks custom_tasks.json

# Debug mode
python run_evaluation.py --log-level DEBUG
```

### Full-Stack Dashboard (React + FastAPI)
```bash
# Start complete application (recommended)
./start_fullstack.sh

# Or start components separately:
./start_backend.sh    # FastAPI backend on :8000
./start_frontend.sh   # React frontend on :3000

# Production build
./build_production.sh
```

### Legacy Streamlit Dashboard
```bash
# Generate reports
python generate_report.py

# Launch Streamlit dashboard (legacy)
python run_dashboard.py
# or
streamlit run dashboard.py
```

### Testing and Quality
```bash
# Run tests
python -m pytest tests/ -v

# Code formatting and linting
black src/ tests/
flake8 src/ tests/
mypy src/
```

## Architecture Overview

### Core Pipeline (9 Phases)
1. **Task Suite Definition** - Load diverse evaluation tasks
2. **Judge Configuration** - Initialize three LLM judges
3. **Metric Proposal** - Each judge proposes 5 task-agnostic metrics
4. **Metric Consolidation** - Merge proposals into canonical set
5. **Agent Output Generation** - Generate responses for all tasks
6. **Scoring Phase** - All judges score all outputs on all metrics
7. **Calibration & Reliability** - Bias correction using anchor sets
8. **Aggregation & Reporting** - Compute final performance scores
9. **Iteration & Maintenance** - Framework for continuous improvement

### Key Components

**Pipeline (`src/pipeline.py`)**: Main orchestrator that manages the entire evaluation workflow. Each phase is implemented as a separate method (phase_1_task_suite, phase_2_configure_judges, etc.).

**Judge System (`src/judge.py`)**: 
- `Judge` class handles individual LLM judges (OpenAI, Anthropic, Google)
- `JudgeManager` coordinates multiple judges
- Implements confidence-based scoring (0.0-1.0) rather than binary evaluation

**Agent (`src/agent.py`)**: Represents the AI being evaluated. Supports multiple providers and includes response caching.

**Metrics (`src/metrics.py`)**: 
- `MetricProposer` collects proposals from judges
- `MetricConsolidator` merges proposals into canonical metrics
- Focus on task-agnostic metrics that work across diverse tasks

**Calibration (`src/calibration.py`)**: Implements bias correction using anchor sets with gold standard scores and inter-judge agreement analysis.

**Aggregation (`src/aggregation.py`)**: Combines scores across judges and metrics, computes final performance, and generates comprehensive reports.

### Configuration

The system uses YAML configuration (`config/judges_config.yaml`) to define:
- Judge configurations (model, provider, API keys, temperature)
- Agent configuration (model being evaluated)
- Evaluation parameters (thresholds, optimization settings)

### Data Flow

1. Tasks loaded from `data/tasks.json` (9 diverse tasks across atomic, compositional, and end-to-end tiers)
2. Anchor tasks from `data/anchors.json` (5 tasks with gold standard scores for calibration)
3. All intermediate results saved to `data/` directory with JSON format
4. Response caching in `data/cache/` for efficiency
5. Final reports generated in `reports/` directory

### Key Design Principles

- **Confidence-based scoring**: All metrics use 0.0-1.0 confidence scales rather than binary pass/fail
- **Provider agnostic**: Supports OpenAI, Anthropic, and Google APIs with unified interface
- **Caching**: Extensive response caching to reduce API costs and improve reproducibility
- **Modularity**: Each phase can be run independently for debugging and development
- **Extensibility**: Easy to add new judges, metrics, or tasks

### Task Tiers
- **Atomic Tasks**: Arithmetic, JSON Parsing, Sentiment Analysis
- **Compositional Tasks**: Data Analysis, Creative Writing, Code Generation  
- **End-to-End Tasks**: Research Summary, Problem Solving, Technical Documentation

## Important Files

**Core System:**
- `src/pipeline.py`: Main evaluation orchestrator
- `src/judge.py`: LLM judge implementations 
- `src/agent.py`: Agent being evaluated
- `config/judges_config.yaml`: System configuration
- `data/tasks.json`: Evaluation task suite
- `data/anchors.json`: Calibration anchor tasks
- `run_evaluation.py`: Main execution script
- `generate_report.py`: Report generation

**Full-Stack Dashboard:**
- `backend/main.py`: FastAPI backend server
- `frontend/ageval-dashboard/`: React TypeScript frontend
- `start_fullstack.sh`: Complete application startup
- `build_production.sh`: Production build script

**Legacy Dashboard:**
- `dashboard.py`: Interactive Streamlit dashboard (legacy)

## Development Notes

**Core System:**
- The system is designed for reliability research and bias detection in LLM evaluation
- All API calls include retry logic with exponential backoff
- Response validation ensures JSON outputs are properly formatted
- Comprehensive logging throughout for debugging and monitoring
- Results are deterministic when using temperature=0 for reproducible experiments

**Full-Stack Dashboard:**
- React frontend built with TypeScript, Material-UI, and Recharts for visualizations
- FastAPI backend provides REST API with automatic OpenAPI documentation
- CORS enabled for development; production serves React build from FastAPI
- API endpoints follow RESTful patterns with proper error handling
- WebSocket connections can be added for real-time evaluation updates
- Frontend uses Axios for API calls with automatic error handling and loading states