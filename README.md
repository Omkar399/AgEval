# AgEval - Three-Judge AI Evaluation System

A comprehensive evaluation framework that uses three diverse LLM judges to assess AI agent performance with bias calibration and reliability analysis.

## 🎯 Overview

AgEval implements a robust 9-phase pipeline where three different LLM judges independently propose task-agnostic metrics, which are then consolidated into a canonical rubric. The system includes bias calibration using anchor sets, inter-judge agreement analysis, and comprehensive reporting.

### Three-Judge Panel
- **Judge A**: OpenAI GPT-4 Turbo - Strong reasoning and consistency
- **Judge B**: Anthropic Claude 3 Sonnet - Balanced performance  
- **Judge C**: Google Gemini 1.5 Flash - Fast and cost-effective

This diverse panel ensures robust evaluation by leveraging different model architectures, training approaches, and reasoning styles.

## 🚀 Quick Start

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd AgEval
   chmod +x scripts/quick_start.sh
   ./scripts/quick_start.sh
   ```

2. **Configure API Keys**
   Edit `config/judges_config.yaml` and add your API keys:
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

3. **Run Evaluation**
   ```bash
   python run_evaluation.py
   ```

4. **Generate Reports**
   ```bash
   python generate_report.py
   ```

## 📋 System Architecture

### 9-Phase Pipeline

1. **Task Suite Definition** - Load and validate diverse evaluation tasks
2. **Judge Configuration** - Initialize three diverse LLM judges  
3. **Metric Proposal** - Each judge independently proposes 5 metrics
4. **Metric Consolidation** - Merge proposals into canonical set of 5 metrics
5. **Agent Output Generation** - Generate responses for all tasks
6. **Scoring Phase** - All judges score all outputs on all metrics
7. **Calibration & Reliability** - Bias correction using anchor sets
8. **Aggregation & Reporting** - Compute final performance scores
9. **Iteration & Maintenance** - Framework for continuous improvement

### Key Features

- ✅ **Task-Agnostic Metrics** that work across different task types
- ✅ **Bias Calibration** using anchor sets to correct systematic errors
- ✅ **Inter-Judge Agreement** analysis with Cohen's κ and correlations  
- ✅ **Cost Optimization** through response caching and batching
- ✅ **Extensible Design** for adding new tasks, metrics, or judges
- ✅ **Comprehensive Reporting** with performance breakdowns and visualizations

## 📊 Task Suite

The evaluation includes 9 diverse tasks across three tiers:

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

## 🔧 Configuration

### Judge Configuration
```yaml
judges:
  - name: JudgeA
    model: gpt-4-turbo-2024-04-09
    provider: openai
    api_key: YOUR_OPENAI_API_KEY_HERE
    
  - name: JudgeB  
    model: claude-3-sonnet-20240229
    provider: anthropic
    api_key: YOUR_ANTHROPIC_API_KEY_HERE
    
  - name: JudgeC
    model: gemini-1.5-flash
    provider: google
    api_key: YOUR_GOOGLE_API_KEY_HERE
```

### Agent Configuration
```yaml
agent:
  name: TestAgent
  model: gpt-4-turbo-2024-04-09  # Change this to your model
  provider: openai  # Change this to your provider
  api_key: YOUR_AGENT_API_KEY_HERE
```

## 📈 Usage Examples

### Basic Evaluation
```bash
# Run complete evaluation
python run_evaluation.py

# Run specific phase
python run_evaluation.py --phase 3

# Custom configuration
python run_evaluation.py --config custom_config.yaml
```

### Advanced Usage
```bash
# Custom data paths
python run_evaluation.py --tasks custom_tasks.json --anchors custom_anchors.json

# Debug mode
python run_evaluation.py --log-level DEBUG

# Generate reports with custom output
python generate_report.py --output-dir custom_reports/
```

## 📊 Output and Reports

The system generates:

- **Performance Scores**: Overall and per-metric performance
- **Calibration Analysis**: Judge bias detection and correction
- **Reliability Metrics**: Inter-judge agreement analysis
- **Visual Reports**: Charts and graphs showing performance breakdowns
- **HTML Dashboard**: Comprehensive evaluation report

## 🔬 Methodology

### Bias Calibration
- Uses 5 anchor tasks with gold standard scores
- Detects systematic judge biases (e.g., consistently scoring 0.1 higher)
- Applies bias corrections to improve accuracy

### Reliability Analysis  
- Computes Cohen's κ for inter-judge agreement
- Identifies subjective metrics with low agreement
- Provides confidence intervals for scores

### Statistical Aggregation
- Bias-corrected averaging across judges
- Weighted scoring based on judge reliability
- Confidence intervals and uncertainty quantification

## 🛠️ Development

### Installation for Development
```bash
pip install -e .[dev]
```

### Running Tests
```bash
python -m pytest tests/ -v
```

### Code Quality
```bash
black src/ tests/
flake8 src/ tests/  
mypy src/
```

## 📁 Project Structure

```
AgEval/
├── src/                    # Core implementation
│   ├── pipeline.py         # Main evaluation pipeline
│   ├── judge.py           # LLM judge implementation
│   ├── agent.py           # Agent interface
│   ├── metrics.py         # Metric consolidation
│   ├── calibration.py     # Bias correction
│   ├── aggregation.py     # Score aggregation
│   └── utils.py           # Utilities and helpers
├── config/                 # Configuration files
├── data/                   # Tasks, anchors, and results
├── tests/                  # Test suite
├── scripts/                # Utility scripts
├── reports/                # Generated reports
├── run_evaluation.py       # Main execution script
├── generate_report.py      # Report generator
└── README.md              # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- Anthropic for Claude API  
- Google for Gemini API
- The research community for evaluation methodologies