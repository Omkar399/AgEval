#!/usr/bin/env python3
"""
Basic tests for AgEval system components.
"""

import unittest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils import load_json, save_json, normalize_score, generate_cache_key
from src.metrics import MetricConsolidator
from src.calibration import Calibrator
from src.aggregation import Aggregator

class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_normalize_score(self):
        """Test score normalization."""
        # Binary scores
        self.assertEqual(normalize_score(1, "binary"), 1.0)
        self.assertEqual(normalize_score(0, "binary"), 0.0)
        self.assertEqual(normalize_score(0.5, "binary"), 0.0)  # Should round down
        
        # Numeric scores
        self.assertEqual(normalize_score(0.5, "numeric"), 0.5)
        self.assertEqual(normalize_score(1.5, "numeric"), 1.0)  # Should clip
        self.assertEqual(normalize_score(-0.5, "numeric"), 0.0)  # Should clip
        
        # Categorical scores
        self.assertEqual(normalize_score("low", "categorical"), 0.0)
        self.assertEqual(normalize_score("medium", "categorical"), 0.5)
        self.assertEqual(normalize_score("high", "categorical"), 1.0)
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        key1 = generate_cache_key("test", "data")
        key2 = generate_cache_key("test", "data")
        key3 = generate_cache_key("different", "data")
        
        self.assertEqual(key1, key2)  # Same inputs should give same key
        self.assertNotEqual(key1, key3)  # Different inputs should give different keys
        self.assertEqual(len(key1), 32)  # MD5 hash should be 32 characters

class TestMetricConsolidator(unittest.TestCase):
    """Test metric consolidation logic."""
    
    def setUp(self):
        self.consolidator = MetricConsolidator()
    
    def test_are_similar_metrics(self):
        """Test metric similarity detection."""
        metric1 = {"name": "Arithmetic Correctness", "definition": "Test", "scale": "Binary"}
        metric2 = {"name": "Math Correctness", "definition": "Test", "scale": "Binary"}
        metric3 = {"name": "JSON Parsing", "definition": "Test", "scale": "Binary"}
        
        # Should detect arithmetic/math similarity
        self.assertTrue(self.consolidator._are_similar_metrics(metric1, metric2))
        
        # Should not detect similarity between arithmetic and JSON
        self.assertFalse(self.consolidator._are_similar_metrics(metric1, metric3))
    
    def test_consolidate_metrics(self):
        """Test metric consolidation."""
        proposals = {
            "JudgeA": [
                {"name": "Arithmetic Correctness", "definition": "Test", "scale": "Binary"},
                {"name": "JSON Parsing", "definition": "Test", "scale": "Binary"},
                {"name": "Clarity", "definition": "Test", "scale": "Numeric"},
                {"name": "Completeness", "definition": "Test", "scale": "Binary"},
                {"name": "Relevance", "definition": "Test", "scale": "Numeric"}
            ],
            "JudgeB": [
                {"name": "Math Correctness", "definition": "Test", "scale": "Binary"},  # Similar to Arithmetic
                {"name": "Format Compliance", "definition": "Test", "scale": "Binary"},
                {"name": "Response Quality", "definition": "Test", "scale": "Numeric"},
                {"name": "Task Completion", "definition": "Test", "scale": "Binary"},
                {"name": "Accuracy", "definition": "Test", "scale": "Numeric"}
            ]
        }
        
        consolidated = self.consolidator.consolidate_metrics(proposals, target_count=5)
        
        self.assertEqual(len(consolidated), 5)
        self.assertTrue(all(isinstance(metric, dict) for metric in consolidated))
        self.assertTrue(all("name" in metric for metric in consolidated))

class TestCalibrator(unittest.TestCase):
    """Test calibration functionality."""
    
    def setUp(self):
        self.calibrator = Calibrator()
    
    def test_calibrate_judges(self):
        """Test bias calibration."""
        # Mock anchor scores and gold standards
        anchor_scores = {
            "JudgeA": {
                "task1": {"Metric1": 0.8, "Metric2": 0.6},
                "task2": {"Metric1": 0.9, "Metric2": 0.7}
            }
        }
        
        anchor_gold = {
            "task1": {"Metric1": 0.7, "Metric2": 0.5},
            "task2": {"Metric1": 0.8, "Metric2": 0.6}
        }
        
        metrics = [
            {"name": "Metric1", "scale": "Numeric"},
            {"name": "Metric2", "scale": "Numeric"}
        ]
        
        bias_offsets = self.calibrator.calibrate_judges(anchor_scores, anchor_gold, metrics)
        
        self.assertIn("JudgeA", bias_offsets)
        self.assertIn("Metric1", bias_offsets["JudgeA"])
        self.assertIn("Metric2", bias_offsets["JudgeA"])
        
        # JudgeA consistently scores 0.1 higher than gold standard
        self.assertAlmostEqual(bias_offsets["JudgeA"]["Metric1"], 0.1, places=2)
        self.assertAlmostEqual(bias_offsets["JudgeA"]["Metric2"], 0.1, places=2)

class TestAggregator(unittest.TestCase):
    """Test score aggregation."""
    
    def setUp(self):
        self.aggregator = Aggregator()
    
    def test_aggregate_scores(self):
        """Test score aggregation."""
        calibrated_scores = {
            "JudgeA": {
                "task1": {"Metric1": 0.8, "Metric2": 0.6},
                "task2": {"Metric1": 0.7, "Metric2": 0.5}
            },
            "JudgeB": {
                "task1": {"Metric1": 0.9, "Metric2": 0.7},
                "task2": {"Metric1": 0.8, "Metric2": 0.6}
            }
        }
        
        metrics = [
            {"name": "Metric1", "scale": "Numeric"},
            {"name": "Metric2", "scale": "Numeric"}
        ]
        
        aggregated = self.aggregator.aggregate_scores(calibrated_scores, metrics)
        
        self.assertIn("task1", aggregated)
        self.assertIn("task2", aggregated)
        self.assertIn("Metric1", aggregated["task1"])
        self.assertIn("Metric2", aggregated["task1"])
        
        # Check that scores are properly averaged
        self.assertAlmostEqual(aggregated["task1"]["Metric1"], 0.85, places=2)  # (0.8 + 0.9) / 2
        self.assertAlmostEqual(aggregated["task1"]["Metric2"], 0.65, places=2)  # (0.6 + 0.7) / 2
    
    def test_compute_overall_performance(self):
        """Test overall performance computation."""
        aggregated_scores = {
            "task1": {"Metric1": 0.8, "Metric2": 0.6},
            "task2": {"Metric1": 0.7, "Metric2": 0.5}
        }
        
        metrics = [
            {"name": "Metric1", "scale": "Numeric"},
            {"name": "Metric2", "scale": "Numeric"}
        ]
        
        overall = self.aggregator.compute_overall_performance(aggregated_scores, metrics)
        
        self.assertIn("Metric1", overall)
        self.assertIn("Metric2", overall)
        
        # Check that overall scores are properly averaged across tasks
        self.assertAlmostEqual(overall["Metric1"], 0.75, places=2)  # (0.8 + 0.7) / 2
        self.assertAlmostEqual(overall["Metric2"], 0.55, places=2)  # (0.6 + 0.5) / 2

class TestDataIntegrity(unittest.TestCase):
    """Test data file integrity."""
    
    def test_tasks_file(self):
        """Test that tasks.json is valid."""
        tasks = load_json("data/tasks.json")
        
        self.assertIsInstance(tasks, list)
        self.assertGreater(len(tasks), 0)
        
        for task in tasks:
            self.assertIn("id", task)
            self.assertIn("prompt", task)
            self.assertIn("tier", task)
            self.assertIn(task["tier"], ["atomic", "compositional", "end-to-end"])
    
    def test_anchors_file(self):
        """Test that anchors.json is valid."""
        anchors = load_json("data/anchors.json")
        
        self.assertIsInstance(anchors, list)
        self.assertGreater(len(anchors), 0)
        
        for anchor in anchors:
            self.assertIn("id", anchor)
            self.assertIn("prompt", anchor)
            self.assertIn("gold_answer", anchor)
            self.assertIn("gold_metrics", anchor)
            
            # Check that gold_metrics contains valid scores
            for metric_name, score in anchor["gold_metrics"].items():
                self.assertIsInstance(score, (int, float))
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 1)

if __name__ == "__main__":
    # Change to project root directory
    os.chdir(Path(__file__).parent.parent)
    
    # Run tests
    unittest.main(verbosity=2) 