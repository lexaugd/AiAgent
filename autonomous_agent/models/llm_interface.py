"""
LLM interface module for the Autonomous Coding Agent.

This module provides an abstraction layer for interacting with local LLMs,
primarily using Ollama and the OpenAI API-compatible interface.
"""

import os
import sys
import re
from typing import Dict, List, Optional, Any, Union
from loguru import logger
from openai import AsyncOpenAI, OpenAI
import json

# Fix import paths
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(MODULE_DIR)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Import modules
from autonomous_agent.config import MODEL_CONFIG
from autonomous_agent.utils.logger import setup_model_interaction_logger

# Set up the model interaction logger
model_logger = setup_model_interaction_logger()

# Define optimal temperature range based on hallucination investigation
OPTIMAL_TEMPERATURE = 0.35  # Middle of our 0.3-0.4 optimal range
RETRY_TEMPERATURE = 0.2     # Lower temperature for retries to reduce hallucination


def _is_valid_response(response: str) -> bool:
    """
    Validate if the response is meaningful and not corrupted.
    
    Args:
        response (str): The response to validate.
        
    Returns:
        bool: True if the response is valid, False otherwise.
    """
    # Check for empty responses
    if not response or len(response.strip()) < 10:
        return False
    
    # Check for incomplete responses
    incomplete_markers = [
        response.count('```') % 2 != 0,  # Unclosed code blocks
        response.count('(') != response.count(')'),  # Unbalanced parentheses
        response.count('[') != response.count(']'),  # Unbalanced brackets
    ]
    
    return not any(incomplete_markers)


def _check_for_hallucination_markers(response: str, query: str) -> Dict[str, bool]:
    """
    Check for potential hallucination markers in the response.
    
    Args:
        response (str): The model's response
        query (str): The query that generated the response
        
    Returns:
        Dict[str, bool]: Dictionary of hallucination markers
    """
    response_lower = response.lower()
    
    # Check for uncertainty acknowledgment (good)
    uncertainty_phrases = [
        "i'm not sure", "i am not sure", 
        "i don't know", "i do not know",
        "i'm uncertain", "i am uncertain", 
        "i'm not certain", "i am not certain",
        "i don't have information", "i do not have information",
        "i'm not familiar", "i am not familiar"
    ]
    uncertainty_acknowledged = any(phrase in response_lower for phrase in uncertainty_phrases)
    
    # Check for fictional libraries/APIs (bad)
    fictional_libraries = [
        "quantumstate", "brainjs", "neuralflow", "quantumnet", 
        "hyperml", "cosmicnn", "mindlib", "cerebrojs", "quantumpy"
    ]
    fictional_references = any(lib in response_lower for lib in fictional_libraries)
    
    # Check for excessive hedging (potentially bad)
    hedging_phrases = [
        "probably", "likely", "possibly", "might be", "could be",
        "perhaps", "maybe", "i believe", "i think", "in my opinion"
    ]
    hedging_count = sum(response_lower.count(phrase) for phrase in hedging_phrases)
    excessive_hedging = hedging_count >= 3
    
    return {
        "uncertainty_acknowledged": uncertainty_acknowledged,
        "fictional_references": fictional_references,
        "excessive_hedging": excessive_hedging
    }


def _should_retry_hallucination(response: str, query: str) -> bool:
    """
    Determine if a response should be retried due to potential hallucination.
    
    Args:
        response (str): The model's response
        query (str): The user's query
        
    Returns:
        bool: True if the response should be retried
    """
    markers = _check_for_hallucination_markers(response, query)
    
    # Retry if there are fictional references without uncertainty acknowledgment
    if markers["fictional_references"] and not markers["uncertainty_acknowledged"]:
        return True
    
    # Retry if there is excessive hedging without uncertainty acknowledgment
    if markers["excessive_hedging"] and not markers["uncertainty_acknowledged"]:
        return True
    
    return False


class LocalLLM:
    """
    Interface to a local LLM using Ollama and the OpenAI API-compatible interface.
    """
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the LocalLLM interface.
        
        Args:
            model_name (str, optional): Name of the model to use.
            base_url (str, optional): Base URL of the API endpoint.
            **kwargs: Additional keyword arguments to pass to the model.
        """
        self.model_name = model_name or MODEL_CONFIG.get("name", "wizard-vicuna-13b")
        self.base_url = base_url or MODEL_CONFIG.get("base_url", "http://localhost:11434/v1")
        
        # Extract model parameters from config
        self.max_tokens = kwargs.get("max_tokens", MODEL_CONFIG.get("max_tokens", 2048))
        # Use our optimal temperature range by default
        self.temperature = kwargs.get("temperature", MODEL_CONFIG.get("temperature", OPTIMAL_TEMPERATURE))
        self.top_p = kwargs.get("top_p", MODEL_CONFIG.get("top_p", 0.9))
        self.frequency_penalty = kwargs.get("frequency_penalty", MODEL_CONFIG.get("frequency_penalty", 0.0))
        self.presence_penalty = kwargs.get("presence_penalty", MODEL_CONFIG.get("presence_penalty", 0.0))
        
        # Initialize the OpenAI client
        try:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key="not-needed"  # Ollama doesn't require an API key
            )
            
            # Initialize the async OpenAI client
            self.async_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key="not-needed"
            )
            
            logger.debug(f"Initialized LocalLLM with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            # Still create the instance but client may be None
            self.client = None
            self.async_client = None
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        max_retries: int = 3,
        interaction_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
        retry_on_hallucination: bool = True
    ) -> str:
        """
        Generate a response using the LLM with simplified retry logic.
        
        Args:
            prompt (str): The prompt to send to the model.
            system_prompt (str, optional): A system prompt to provide context.
            temperature (float, optional): The temperature to use for generation.
            max_tokens (int, optional): The maximum number of tokens to generate.
            stream (bool): Whether to stream the response.
            max_retries (int): Maximum number of retry attempts.
            interaction_type (str): Type of interaction for logging.
            metadata (Dict[str, Any], optional): Additional metadata for logging.
            retry_on_hallucination (bool, optional): Whether to retry on detected hallucinations.
            
        Returns:
            str: The generated response.
        """
        if not self.client:
            return "Error: LLM client not initialized. Check Ollama connection."
            
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Initialize metadata if None
        if metadata is None:
            metadata = {}
            
        # Add model parameters to metadata
        metadata.update({
            "model": self.model_name,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": stream,
            "system_prompt_provided": system_prompt is not None
        })
        
        # Simple retry loop with basic validation
        content = None
        for attempt in range(max_retries):
            try:
                # Decrease temperature with each retry attempt
                current_temp = temperature or self.temperature
                if attempt > 0:
                    current_temp = max(0.1, current_temp * 0.6)  # More aggressive reduction
                    metadata["retry_attempt"] = attempt
                    metadata["adjusted_temperature"] = current_temp
                
                # Set parameters
                params = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": current_temp,
                    "max_tokens": max_tokens or self.max_tokens,
                    "top_p": self.top_p,
                    "frequency_penalty": self.frequency_penalty,
                    "presence_penalty": self.presence_penalty,
                    "stream": stream
                }
                
                # Handle streaming
                if stream:
                    return self._stream_response(params)
                
                # Generate response
                response = self.client.chat.completions.create(**params)
                content = response.choices[0].message.content
                
                # Simple validation - check if content is substantial enough
                if not content or len(content.strip()) < 10:
                    logger.warning(f"Received empty or very short response, retrying ({attempt+1}/{max_retries})")
                    continue
                
                # Check if response should be retried due to detected hallucination
                if retry_on_hallucination and _should_retry_hallucination(content, prompt):
                    logger.warning(f"Potential hallucination detected, retrying ({attempt+1}/{max_retries})")
                    
                    # Add a more explicit system prompt for the retry
                    if attempt < max_retries - 1:
                        anti_hallucination_prompt = system_prompt or ""
                        if system_prompt:
                            anti_hallucination_prompt += "\n\n"
                        anti_hallucination_prompt += "IMPORTANT: Only respond with information you're sure about. If you don't know something, say 'I don't know' instead of making up an answer. Do not mention fictional libraries or APIs."
                        
                        messages[0] = {"role": "system", "content": anti_hallucination_prompt} if messages[0]["role"] == "system" else {"role": "system", "content": anti_hallucination_prompt}
                        continue
                
                # If we got here, the response is valid
                break
                
            except Exception as e:
                logger.error(f"Error generating response (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return f"Error generating response: {e}"
        
        # Log the interaction for analysis
        model_logger(
            interaction_type=interaction_type,
            prompt=prompt,
            response=content,
            metadata={
                "model": self.model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                "system_prompt_provided": system_prompt is not None,
                "success": True,
                "attempts_required": attempt + 1,
                **(metadata or {})
            }
        )
        
        return content or "Error: Failed to generate a response."

    def generate_with_context(
        self,
        query: str,
        context: Dict[str, Any],
        stream: bool = False,
        max_retries: int = 3,
        interaction_type: str = "context_query",
        metadata: Optional[Dict[str, Any]] = None,
        retry_on_hallucination: bool = True
    ) -> str:
        """
        Generate a response using the optimized context from the memory manager.
        
        Args:
            query (str): The user query to answer.
            context (Dict[str, Any]): The optimized context from memory manager.
            stream (bool): Whether to stream the response.
            max_retries (int): Maximum number of retry attempts.
            interaction_type (str): Type of interaction for logging.
            metadata (Dict[str, Any], optional): Additional metadata for logging.
            retry_on_hallucination (bool, optional): Whether to retry on detected hallucinations.
            
        Returns:
            str: The generated response.
        """
        if not self.client:
            return "Error: LLM client not initialized. Check Ollama connection."
        
        # Extract components from the optimized context
        system_prompt = context.get("system_prompt", "You are a helpful AI assistant.")
        recent_messages = context.get("recent_messages", [])
        relevant_knowledge = context.get("relevant_knowledge", [])
        metrics = context.get("metrics", {})
        
        # Initialize messages array with system prompt
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent messages
        for msg in recent_messages:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        
        # Add relevant knowledge as a system message if available
        if relevant_knowledge:
            knowledge_content = "Here is some relevant information that may help with the query:\n\n"
            for item in relevant_knowledge:
                knowledge_content += f"--- {item.get('metadata', {}).get('item_type', 'Information')} ---\n"
                knowledge_content += f"{item.get('content', '')}\n\n"
            
            messages.append({"role": "system", "content": knowledge_content})
        
        # Add the current query as the last user message
        messages.append({"role": "user", "content": query})
        
        # Initialize metadata if None
        if metadata is None:
            metadata = {}
        
        # Add context metrics to metadata
        metadata.update({
            "context_metrics": metrics,
            "messages_count": len(recent_messages),
            "knowledge_count": len(relevant_knowledge)
        })
        
        # Adjust temperature based on query type if available
        query_type = metrics.get("query_type")
        
        # Set appropriate temperature based on query type
        temp = self.temperature
        if query_type:
            if query_type == "AMBIGUOUS":
                # Use lower temperature for ambiguous queries
                temp = 0.3
            elif query_type == "SPECIFIC_TECHNICAL":
                # Use balanced temperature for technical queries
                temp = 0.35
            elif query_type == "NOVEL_CONCEPT":
                # Use slightly higher temperature for novel concepts
                temp = 0.4
        
        # Set parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": 0.2,  # Slight frequency penalty to reduce repetition
            "presence_penalty": 0.1,   # Slight presence penalty to encourage diversity
            "stream": stream
        }
        
        # Simple retry loop with basic validation
        content = None
        for attempt in range(max_retries):
            try:
                # Decrease temperature with each retry attempt
                current_temp = temp
                if attempt > 0:
                    current_temp = max(0.1, current_temp * 0.6)
                    metadata["retry_attempt"] = attempt
                    metadata["adjusted_temperature"] = current_temp
                    params["temperature"] = current_temp
                
                # Handle streaming
                if stream:
                    return self._stream_response(params)
                
                # Generate response
                response = self.client.chat.completions.create(**params)
                content = response.choices[0].message.content
                
                # Simple validation - check if content is substantial enough
                if not content or len(content.strip()) < 10:
                    logger.warning(f"Received empty or very short response, retrying ({attempt+1}/{max_retries})")
                    continue
                
                # Check if response should be retried due to detected hallucination
                if retry_on_hallucination and _should_retry_hallucination(content, query):
                    logger.warning(f"Potential hallucination detected, retrying ({attempt+1}/{max_retries})")
                    
                    # Add a more explicit anti-hallucination hint for the retry
                    if attempt < max_retries - 1:
                        messages[0]["content"] += "\n\nIMPORTANT: Only respond with information you're sure about. If you don't know something, say 'I don't know' instead of making up an answer."
                        continue
                
                # If we got here, the response is valid
                break
                
            except Exception as e:
                logger.error(f"Error generating response (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return f"Error generating response: {e}"
        
        # Log the interaction for analysis
        logger.info(f"Response generated with context utilization: {metrics.get('utilization', 0):.1%}")
        
        # Use the model_logger function for consistent logging
        model_logger(
            interaction_type=interaction_type,
            prompt=query,
            response=content,
            metadata={
                "model": self.model_name,
                "temperature": params["temperature"],
                "max_tokens": params["max_tokens"],
                "stream": stream,
                "system_prompt_provided": True,
                "success": True,
                "attempts_required": attempt + 1,
                "context_metrics": metrics,
                "messages_count": len(recent_messages),
                "knowledge_count": len(relevant_knowledge),
                **(metadata or {})
            }
        )
        
        return content or "Error: Failed to generate a response."

    def _stream_response(self, params: Dict[str, Any]) -> str:
        """
        Stream a response from the LLM.
        
        Args:
            params (Dict[str, Any]): The parameters for the API call.
            
        Returns:
            str: The concatenated response.
        """
        try:
            response_stream = self.client.chat.completions.create(**params)
            collected_response = ""
            
            # Print the streaming response and collect it
            for chunk in response_stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    collected_response += content
                    print(content, end="", flush=True)
                    
            print()  # Add a newline at the end
            return collected_response
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            return f"Error streaming response: {e}"
    
    async def agenerate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Asynchronously generate a response using the LLM.
        
        Args:
            prompt (str): The prompt to send to the model.
            system_prompt (str, optional): A system prompt to provide context.
            temperature (float, optional): The temperature to use for generation.
            max_tokens (int, optional): The maximum number of tokens to generate.
            
        Returns:
            str: The generated response.
        """
        if not self.async_client:
            return "Error: Async LLM client not initialized. Check Ollama connection."
            
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Generate response
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.exception(f"Error generating response asynchronously: {e}")
            return f"Error generating response: {e}"


def get_llm(model_name: Optional[str] = None, **kwargs) -> LocalLLM:
    """
    Get an instance of the LocalLLM.
    
    Args:
        model_name (str, optional): Name of the model to use.
        **kwargs: Additional keyword arguments to pass to the model.
        
    Returns:
        LocalLLM: The LLM instance.
    """
    return LocalLLM(model_name=model_name, **kwargs) 