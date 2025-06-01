"""
Metric proposal and consolidation for the three-judge evaluation system.
"""

import logging
from typing import Dict, Any, List, Set
from collections import Counter
import json
from .utils import validate_json_response

logger = logging.getLogger(__name__)

class MetricProposer:
    """Handles metric proposal from judges."""
    
    def __init__(self):
        pass
    
    def collect_proposals(self, judge_manager, tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Collect metric proposals from all judges."""
        logger.info("Collecting metric proposals from judges...")
        proposals = judge_manager.propose_metrics(tasks)
        
        # Log proposals for review
        for judge_name, metrics in proposals.items():
            logger.info(f"Judge {judge_name} proposed:")
            for i, metric in enumerate(metrics, 1):
                logger.info(f"  {i}. {metric.get('name', 'Unknown')} - {metric.get('scale', 'Unknown')}")
        
        return proposals

class MetricConsolidator:
    """Consolidates metric proposals into a canonical set."""
    
    def __init__(self):
        pass
    
    def consolidate_metrics(self, proposals: Dict[str, List[Dict[str, Any]]], target_count: int = 5) -> List[Dict[str, Any]]:
        """Consolidate metric proposals into a canonical set."""
        logger.info(f"Consolidating metrics from {len(proposals)} judges...")
        
        # Flatten all proposals
        all_metrics = []
        for judge_name, metrics in proposals.items():
            for metric in metrics:
                metric['proposed_by'] = judge_name
                all_metrics.append(metric)
        
        logger.info(f"Total proposed metrics: {len(all_metrics)}")
        
        # Group similar metrics by name similarity
        grouped_metrics = self._group_similar_metrics(all_metrics)
        
        # Select best representative from each group
        consolidated = self._select_best_metrics(grouped_metrics, target_count)
        
        logger.info(f"Consolidated to {len(consolidated)} canonical metrics:")
        for i, metric in enumerate(consolidated, 1):
            logger.info(f"  {i}. {metric['name']} ({metric['scale']})")
        
        return consolidated
    
    def _group_similar_metrics(self, metrics: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group metrics with similar names or definitions."""
        groups = []
        used_indices = set()
        
        for i, metric in enumerate(metrics):
            if i in used_indices:
                continue
            
            # Start a new group with this metric
            group = [metric]
            used_indices.add(i)
            
            # Find similar metrics
            for j, other_metric in enumerate(metrics):
                if j in used_indices or i == j:
                    continue
                
                if self._are_similar_metrics(metric, other_metric):
                    group.append(other_metric)
                    used_indices.add(j)
            
            groups.append(group)
        
        logger.info(f"Grouped {len(metrics)} metrics into {len(groups)} groups")
        return groups
    
    def _are_similar_metrics(self, metric1: Dict[str, Any], metric2: Dict[str, Any]) -> bool:
        """Check if two metrics are similar enough to be grouped."""
        name1 = metric1.get('name', '').lower()
        name2 = metric2.get('name', '').lower()
        
        # Check for exact name matches
        if name1 == name2:
            return True
        
        # Check for key word overlaps
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        # If they share significant words, consider them similar
        if len(words1.intersection(words2)) >= 1 and len(words1.union(words2)) <= 4:
            return True
        
        # Check for semantic similarity in common metric types
        similar_groups = [
            {'arithmetic', 'math', 'calculation', 'numeric', 'correctness'},
            {'json', 'format', 'structure', 'parsing', 'field'},
            {'reasoning', 'logic', 'thought', 'chain', 'step'},
            {'plan', 'coherence', 'sequence', 'order', 'flow'},
            {'message', 'communication', 'appropriate', 'tone', 'clarity'}
        ]
        
        for group in similar_groups:
            if any(word in name1 for word in group) and any(word in name2 for word in group):
                return True
        
        return False
    
    def _select_best_metrics(self, grouped_metrics: List[List[Dict[str, Any]]], target_count: int) -> List[Dict[str, Any]]:
        """Select the best representative metrics from groups."""
        # Sort groups by size (larger groups = more consensus)
        grouped_metrics.sort(key=len, reverse=True)
        
        selected = []
        
        # Take the best metric from each group until we have enough
        for group in grouped_metrics:
            if len(selected) >= target_count:
                break
            
            # Select the best metric from this group
            best_metric = self._select_best_from_group(group)
            selected.append(best_metric)
        
        # If we don't have enough metrics, add more from remaining groups
        while len(selected) < target_count and len(grouped_metrics) > len(selected):
            remaining_groups = grouped_metrics[len(selected):]
            if remaining_groups:
                best_metric = self._select_best_from_group(remaining_groups[0])
                selected.append(best_metric)
            else:
                break
        
        # If we still don't have enough, create default metrics
        if len(selected) < target_count:
            selected.extend(self._create_default_metrics(target_count - len(selected)))
        
        return selected[:target_count]
    
    def _select_best_from_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best metric from a group."""
        if len(group) == 1:
            return group[0]
        
        # Prefer metrics with clearer definitions
        scored_metrics = []
        for metric in group:
            score = 0
            definition = metric.get('definition', '')
            
            # Prefer longer, more detailed definitions
            score += len(definition.split()) * 0.1
            
            # Prefer confidence-based scales, then numeric, then binary, then categorical
            scale = metric.get('scale', '').lower()
            if 'confidence' in scale or '[0.0-1.0]' in scale:
                score += 3  # Highest preference for confidence scoring
            elif 'numeric' in scale:
                score += 2
            elif 'binary' in scale:
                score += 1.5
            elif 'categorical' in scale:
                score += 1
            
            # Prefer metrics with specific computation instructions
            if any(word in definition.lower() for word in ['if', 'compute', 'count', 'measure', 'check']):
                score += 1
            
            scored_metrics.append((score, metric))
        
        # Return the highest scoring metric
        scored_metrics.sort(key=lambda x: x[0], reverse=True)
        best_metric = scored_metrics[0][1]
        
        # Merge information from other metrics in the group if helpful
        return self._merge_metric_info(best_metric, group)
    
    def _merge_metric_info(self, best_metric: Dict[str, Any], group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge information from grouped metrics to improve the best one."""
        merged = best_metric.copy()
        
        # Collect all judges who proposed similar metrics
        proposers = [m.get('proposed_by', 'Unknown') for m in group]
        merged['proposed_by'] = proposers
        merged['consensus_count'] = len(group)
        
        return merged
    
    def _create_default_metrics(self, count: int) -> List[Dict[str, Any]]:
        """Create default metrics if not enough were proposed."""
        defaults = [
            {
                "name": "Task Completion",
                "definition": "Confidence in how completely the agent addressed all parts of the prompt and task requirements. Higher scores for comprehensive responses that cover all aspects.",
                "scale": "Confidence [0.0-1.0]",
                "proposed_by": ["system"],
                "is_default": True
            },
            {
                "name": "Response Relevance", 
                "definition": "Confidence in how relevant and on-topic the agent's response is to the given prompt. Higher scores for responses that directly address the question or task.",
                "scale": "Confidence [0.0-1.0]",
                "proposed_by": ["system"],
                "is_default": True
            },
            {
                "name": "Factual Accuracy",
                "definition": "Confidence in the factual correctness and technical accuracy of the information provided. Higher scores for responses with verified correct information.",
                "scale": "Confidence [0.0-1.0]", 
                "proposed_by": ["system"],
                "is_default": True
            },
            {
                "name": "Clarity and Coherence",
                "definition": "Confidence in how clear, well-structured, and easy to understand the response is. Higher scores for responses with logical flow and good organization.",
                "scale": "Confidence [0.0-1.0]",
                "proposed_by": ["system"],
                "is_default": True
            },
            {
                "name": "Response Quality",
                "definition": "Overall confidence in the quality and usefulness of the response considering all factors like depth, insight, and practical value.",
                "scale": "Confidence [0.0-1.0]",
                "proposed_by": ["system"],
                "is_default": True
            }
        ]
        
        logger.warning(f"Adding {count} default confidence-based metrics to reach target count")
        return defaults[:count] 