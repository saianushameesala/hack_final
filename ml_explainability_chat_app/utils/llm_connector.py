"""
LLM Connector Module for ML Explainability Chat App.
Handles connections to OpenAI for natural language explanations.
"""
import os
import logging
import time
from typing import Dict, Any, Optional, List, Union
import openai
import os
# Configure logging
logger = logging.getLogger("llm_connector")

class LLMConnector:
    """Base class for LLM connections"""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """Initialize the LLM connector"""
        self.model_name = model_name
        self.api_key = api_key
        self.is_initialized = False
        
    def generate_text(self, prompt: str, system_prompt: str = None, 
                     max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate text from the LLM"""
        raise NotImplementedError("Child classes must implement generate_text")
    
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.is_initialized

class OpenAIConnector(LLMConnector):
    """OpenAI API connector"""
    
    def __init__(self, model_name: str = "gpt-4.1-nano", api_key: str = None):
        """Initialize OpenAI connector"""
        super().__init__(model_name, api_key)
        
        # Get API key
        if self.api_key is None:
            if 'OPENAI_API_KEY' in os.environ:
                self.api_key = os.environ['OPENAI_API_KEY']
            
        # Setup client
        if self.api_key:
            try:
                openai.api_key = self.api_key
                # Use correct OpenAI API (>=1.0.0)
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": "Hello"}],
                    max_tokens=5
                )
                self.is_initialized = True
                logger.info(f"OpenAI connector initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Error initializing OpenAI: {str(e)}")
        else:
            logger.error("No OpenAI API key provided")
            
    def generate_text(self, prompt: str, system_prompt: str = None, 
                     max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate text using OpenAI's API"""
        if not self.is_initialized:
            raise RuntimeError("OpenAI connector not initialized")
            
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": "You are an expert in explaining machine learning concepts."})
            
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = openai.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    return response.choices[0].message.content.strip()
                except openai.RateLimitError:
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise RuntimeError(f"OpenAI API error: {str(e)}")

def get_available_llm() -> LLMConnector:
    """Get an available LLM connector"""
    
    try:
        connector = OpenAIConnector()
        if connector.is_initialized:
            logger.info("Using OpenAI connector")
            return connector
    except Exception as e:
        logger.error(f"OpenAI connector failed: {str(e)}")
    
    logger.error("No LLM connector available")
    raise RuntimeError("No working LLM connector available")
