# 🤖 AgEval - Adaptive AI Agent Evaluation

A modern, real-time dashboard for evaluating AI agents using adaptive Item Response Theory (IRT) algorithms with comprehensive three-judge evaluation framework.

## ✨ Features

- **🧠 Real-time Evaluation**: Watch agents think and respond in real-time
- **📊 Adaptive Testing**: IRT-based adaptive difficulty selection
- **⚖️ Three-Judge Panel**: GPT-4, Claude, and Gemini for robust evaluation
- **🎯 Performance Tracking**: Comprehensive agent performance analysis
- **📈 Live Visualizations**: Interactive charts and trajectory plots
- **⚡ WebSocket Updates**: Instant progress streaming
- **🎨 Modern UI**: Clean, minimalistic Anthropic-style interface

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Copy and edit the configuration file:
```bash
cp config/judges_config.yaml.example config/judges_config.yaml
```

Add your API keys in `config/judges_config.yaml`:
```yaml
judges:
  - name: JudgeA
    api_key: YOUR_OPENAI_API_KEY_HERE  # For GPT-4
  - name: JudgeB  
    api_key: YOUR_ANTHROPIC_API_KEY_HERE  # For Claude
  - name: JudgeC
    api_key: YOUR_GOOGLE_API_KEY_HERE  # For Gemini

agent:
  api_key: YOUR_AGENT_API_KEY_HERE  # For the model being evaluated
```

### 3. Start the Dashboard
```bash
python start_server.py
```

### 4. Open Browser
Navigate to: **http://localhost:8001**

## 📖 Usage

### Real-time Evaluation Dashboard
1. Go to the **Evaluation** tab
2. Select evaluation type (Adaptive or Static)
3. Set number of agents
4. Click **Run Evaluation**
5. Watch real-time progress with:
   - Agent thinking process
   - IRT ability updates
   - Live trajectory plots
   - Detailed logs

### Viewing Results
- **Overview**: High-level metrics and comparison charts
- **Evaluation**: Real-time evaluation monitoring
- **Agents**: Detailed agent performance cards and analysis

### Command-line Evaluation
```bash
# Run traditional evaluation
python run_evaluation.py

# Run adaptive evaluation
python run_enhanced_evaluation.py

# Generate comprehensive report
python generate_report.py
```

## 🏗️ Architecture

```
AgEval/
├── fastapi_app.py          # Modern web dashboard
├── start_server.py         # Server launcher
├── src/                    # Core evaluation modules
│   ├── adaptive_evaluation.py  # IRT-based adaptive testing
│   ├── agent.py                # Agent interface
│   ├── judge.py               # Three-judge evaluation
│   ├── pipeline.py            # Traditional evaluation pipeline
│   └── ...
├── templates/              # Modern frontend UI
│   └── index.html
├── data/                   # Evaluation data and cache
└── config/                 # Configuration files
```

## 📊 Evaluation Methods

### 1. Adaptive Evaluation (Recommended)
- **IRT-based**: Uses Item Response Theory for optimal task selection
- **Efficient**: Up to 70% fewer tasks needed
- **Real-time**: Live progress monitoring
- **Precise**: Confidence interval estimation

### 2. Traditional Three-Judge Evaluation
- **Robust**: Three diverse LLM judges (GPT-4, Claude, Gemini)
- **Bias-corrected**: Calibration using anchor tasks
- **Comprehensive**: 9-phase evaluation pipeline
- **Reliable**: Inter-judge agreement analysis

## 🎯 Task Suite

### Atomic Tasks (3)
- **Arithmetic**: Basic mathematical calculations
- **JSON Parsing**: Structured data extraction  
- **Sentiment Analysis**: Text classification

### Compositional Tasks (3)  
- **Data Analysis**: Multi-step analytical reasoning
- **Creative Writing**: Open-ended text generation
- **Code Generation**: Programming task completion

### End-to-End Tasks (3)
- **Research Summary**: Complex information synthesis
- **Problem Solving**: Multi-faceted reasoning
- **Technical Documentation**: Comprehensive explanation

## 📈 Key Benefits

- **⚡ Efficiency**: Adaptive evaluation reduces testing time by 70%
- **📊 Precision**: IRT-based ability estimation with confidence intervals  
- **🔍 Transparency**: Complete visibility into evaluation process
- **🚀 Real-time**: Live progress tracking and visualization
- **⚖️ Robustness**: Three-judge validation with bias correction

## 🛠️ Development

### Run with Auto-reload
```bash
uvicorn fastapi_app:app --reload --port 8001
```

### Run Tests
```bash
python test_adaptive_evaluation.py
python tests/test_basic.py
```

## 📊 Data Files

- `data/tasks.json` - Evaluation tasks
- `data/anchors.json` - IRT anchor points
- `data/adaptive_evaluation_results.json` - Latest adaptive results
- `data/comprehensive_analysis.json` - Traditional evaluation results
- `data/cache/` - Cached responses for efficiency

---

**Ready to evaluate your AI agents?** Run `python start_server.py` and start exploring! 🚀

For traditional command-line evaluation, use `python run_evaluation.py` or `python run_enhanced_evaluation.py` for adaptive testing.