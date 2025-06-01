"""
Agent implementation for the AI being evaluated.
"""

import openai
import anthropic
import google.generativeai as genai
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from .utils import ResponseCache, generate_cache_key, retry_with_backoff

logger = logging.getLogger(__name__)

class Agent:
    """The AI agent being evaluated by the three-judge system."""
    
    def __init__(self, config: Dict[str, Any], cache: Optional[ResponseCache] = None):
        self.name = config['name']
        self.config = config
        self.provider = config['provider']
        self.model = config['model']
        self.api_key = config['api_key']
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2000)
        self.cache = cache
        
        # Initialize API client
        if self.provider == 'openai':
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.provider == 'anthropic':
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == 'google':
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        logger.info(f"Initialized agent: {self.name} using {self.model}")
    
    @retry_with_backoff
    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """Make API call to the agent's LLM."""
        if self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        
        elif self.provider == 'anthropic':
            # Convert messages format for Anthropic
            system_message = ""
            user_messages = []
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    user_messages.append(msg)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_message,
                messages=user_messages
            )
            return response.content[0].text
        
        elif self.provider == 'google':
            # Convert messages to single prompt for Gemini
            prompt = ""
            for msg in messages:
                if msg['role'] == 'system':
                    prompt += f"System: {msg['content']}\n\n"
                elif msg['role'] == 'user':
                    prompt += f"User: {msg['content']}\n\n"
                elif msg['role'] == 'assistant':
                    prompt += f"Assistant: {msg['content']}\n\n"
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
            
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
    
    def generate_response(self, prompt: str, task_id: str, use_cache: bool = True) -> Dict[str, Any]:
        """Generate response for a given prompt."""
        # Generate cache key
        cache_key = generate_cache_key(self.name, self.model, prompt, self.temperature)
        
        # Try cache first
        if use_cache and self.cache:
            cached_response = self.cache.get(cache_key)
            if cached_response:
                logger.debug(f"Using cached response for task {task_id}")
                return cached_response
        
        # Make API call
        messages = [{"role": "user", "content": prompt}]
        response_text = self._call_api(messages)
        
        # Prepare response data
        response_data = {
            "task_id": task_id,
            "prompt": prompt,
            "response": response_text,
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "temperature": self.temperature
        }
        
        # Cache response
        if use_cache and self.cache:
            self.cache.set(cache_key, response_data)
        
        logger.info(f"Agent generated response for task {task_id}")
        return response_data
    
    def generate_all_outputs(self, tasks: List[Dict[str, Any]], use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
        """Generate outputs for all tasks in the suite."""
        outputs = {}
        failed_tasks = []
        
        for task in tasks:
            try:
                output = self.generate_response(task['prompt'], task['id'], use_cache)
                outputs[task['id']] = output
            except Exception as e:
                logger.error(f"Failed to generate output for task {task['id']}: {e}")
                failed_tasks.append(task['id'])
                # Create a placeholder output for failed tasks
                outputs[task['id']] = {
                    "task_id": task['id'],
                    "prompt": task['prompt'],
                    "response": f"ERROR: Failed to generate response - {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model,
                    "temperature": self.temperature,
                    "error": True
                }
        
        if failed_tasks:
            logger.warning(f"Failed to generate outputs for {len(failed_tasks)} tasks: {failed_tasks}")
        
        logger.info(f"Agent completed {len(outputs)} tasks ({len(outputs) - len(failed_tasks)} successful)")
        return outputs
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": self.name,
            "model": self.model,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        } 