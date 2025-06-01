"""
AgEval - Three-Judge AI Evaluation System

A comprehensive evaluation framework that uses three separate LLMs as judges 
to score AI agent outputs across diverse tasks.
"""

__version__ = "1.0.0"
__author__ = "AgEval Team"

from .judge import Judge, JudgeManager
from .agent import Agent
from .metrics import MetricProposer, MetricConsolidator
from .calibration import Calibrator
from .aggregation import Aggregator
from .pipeline import EvaluationPipeline

__all__ = [
    "Judge",
    "JudgeManager", 
    "Agent",
    "MetricProposer",
    "MetricConsolidator",
    "Calibrator",
    "Aggregator",
    "EvaluationPipeline"
] 