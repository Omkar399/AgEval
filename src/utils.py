"""
Utility functions for the AgEval system.
"""

import json
import yaml
import hashlib
import time
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime and pandas timestamp objects."""
    
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):  # Handle other datetime-like objects
            return obj.isoformat()
        elif hasattr(obj, 'to_pydatetime'):  # Handle pandas datetime objects
            return obj.to_pydatetime().isoformat()
        return super().default(obj)

def load_config(config_path: str = "config/judges_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise

def load_json(file_path: str) -> Any:
    """Load data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        raise

def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """Save data to JSON file with custom encoder for datetime objects."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, cls=CustomJSONEncoder)

def generate_cache_key(*args) -> str:
    """Generate a cache key from arguments."""
    content = json.dumps(args, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()

class ResponseCache:
    """Simple file-based cache for API responses."""
    
    def __init__(self, cache_dir: str = "data/cache", duration: int = 86400):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.duration = duration
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached response if valid."""
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            
            # Check if cache is still valid
            cached_time = datetime.fromisoformat(cached['timestamp'])
            if datetime.now() - cached_time > timedelta(seconds=self.duration):
                cache_file.unlink()  # Remove expired cache
                return None
            
            return cached['data']
        except (json.JSONDecodeError, KeyError, ValueError):
            # Invalid cache file, remove it
            cache_file.unlink()
            return None
    
    def set(self, key: str, data: Any) -> None:
        """Cache response data."""
        cache_file = self.cache_dir / f"{key}.json"
        cached = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        with open(cache_file, 'w') as f:
            json.dump(cached, f, indent=2)

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def validate_json_response(response: str) -> Dict[str, Any]:
    """Validate and parse JSON response from LLM."""
    try:
        # Try to extract JSON from response if it's wrapped in markdown
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        
        # Remove any leading/trailing whitespace and parse
        return json.loads(response.strip())
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.error(f"Response content: {response}")
        raise ValueError(f"Invalid JSON response: {e}")

def normalize_score(score: Any, scale: str) -> float:
    """Normalize score to 0-1 range based on scale type."""
    # Handle confidence-based scoring (new format)
    if "confidence" in scale.lower() or "[0.0-1.0]" in scale.lower():
        return max(0.0, min(1.0, float(score)))
    elif scale.lower() == "binary":
        return float(score) if score in [0, 1] else 0.0
    elif scale.lower() == "numeric":
        return max(0.0, min(1.0, float(score)))
    elif scale.lower() == "categorical":
        # Map categorical to numeric
        if isinstance(score, str):
            mapping = {"low": 0.0, "medium": 0.5, "high": 1.0}
            return mapping.get(score.lower(), 0.0)
        elif isinstance(score, (int, float)):
            # Assume 0, 1, 2 mapping
            return min(1.0, max(0.0, float(score) / 2.0))
    
    return 0.0

def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split list into batches of specified size."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """Retry function with exponential backoff."""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
    return wrapper 