"""
Model manager for the Autonomous Coding Agent.

This module provides a unified interface for managing multiple language models
and coordinating their use based on task requirements.
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from loguru import logger

from .llm_interface import LocalLLM, get_llm
from autonomous_agent.config import MODEL_CONFIG


class ModelType(Enum):
    """Enum representing different types of models."""
    REASONING = "reasoning"
    CODING = "coding"
    GENERAL = "general"


class TaskType(Enum):
    """Enum representing different types of tasks."""
    PLANNING = "planning"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    PROBLEM_SOLVING = "problem_solving"
    EXPLANATION = "explanation"
    GENERAL = "general"


class ModelManager:
    """
    Manages multiple language models and coordinates their use based on task requirements.
    """
    
    def __init__(
        self,
        agent_id: str = "default",
        reasoning_model_config: Optional[Dict[str, Any]] = None,
        coding_model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the model manager.
        
        Args:
            agent_id (str): The ID of the agent using this model manager.
            reasoning_model_config (Dict[str, Any], optional): Configuration for the reasoning model.
            coding_model_config (Dict[str, Any], optional): Configuration for the coding model.
        """
        self.agent_id = agent_id
        
        # Initialize model configurations
        self.coding_model_config = coding_model_config or MODEL_CONFIG
        self.reasoning_model_config = reasoning_model_config or {
            "name": "phi3:mini",  # Default reasoning model
            "base_url": "http://localhost:11434/v1",  # Ollama API endpoint
            "max_tokens": 4096,
            "temperature": 0.7,  # Higher temperature for more creative reasoning
            "top_p": 0.95,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.2,
        }
        
        # Initialize models
        self.coding_model = get_llm(**self.coding_model_config)
        self.reasoning_model = get_llm(**self.reasoning_model_config)
        
        # Task routing configuration
        self.task_model_mapping = {
            TaskType.PLANNING: ModelType.REASONING,
            TaskType.CODE_GENERATION: ModelType.CODING,
            TaskType.CODE_REVIEW: ModelType.CODING,
            TaskType.PROBLEM_SOLVING: ModelType.REASONING,
            TaskType.EXPLANATION: ModelType.REASONING,
            TaskType.GENERAL: ModelType.REASONING,
        }
        
        # Performance tracking
        self.model_stats = {
            ModelType.REASONING: {"calls": 0, "tokens_in": 0, "tokens_out": 0, "time": 0},
            ModelType.CODING: {"calls": 0, "tokens_in": 0, "tokens_out": 0, "time": 0},
        }
        
        logger.info(f"Initialized ModelManager for agent: {agent_id}")
    
    def classify_task(self, query: str) -> TaskType:
        """
        Classify the task type based on the query.
        
        Args:
            query (str): The user query or task description.
            
        Returns:
            TaskType: The classified task type.
        """
        # Simple keyword-based classification
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["plan", "design", "architecture", "approach", "strategy"]):
            return TaskType.PLANNING
        
        if any(term in query_lower for term in ["optimize", "improve", "refactor", "better", "enhance", "review"]):
            return TaskType.CODE_REVIEW
            
        if any(term in query_lower for term in ["fix", "debug", "problem", "error", "issue", "solve", "incorrect"]):
            return TaskType.PROBLEM_SOLVING
            
        if any(term in query_lower for term in ["generate", "create", "write", "implement", "code", "function", "class"]):
            return TaskType.CODE_GENERATION
            
        if any(term in query_lower for term in ["explain", "describe", "clarify", "understand", "how does"]):
            return TaskType.EXPLANATION
            
        return TaskType.GENERAL
    
    def get_model_for_task(self, task_type: TaskType) -> LocalLLM:
        """
        Get the appropriate model for the given task type.
        
        Args:
            task_type (TaskType): The type of task.
            
        Returns:
            LocalLLM: The appropriate model interface.
        """
        model_type = self.task_model_mapping.get(task_type, ModelType.REASONING)
        
        if model_type == ModelType.CODING:
            return self.coding_model
        else:
            return self.reasoning_model
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]],
        task_type: Optional[TaskType] = None,
        model_type: Optional[ModelType] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using the appropriate model.
        
        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.
            task_type (TaskType, optional): The type of task. If not provided, it will be classified.
            model_type (ModelType, optional): Force a specific model type. Overrides task_type.
            **kwargs: Additional parameters to pass to the model.
            
        Returns:
            str: The generated response.
        """
        # Determine which model to use
        if model_type:
            selected_model_type = model_type
        else:
            if not task_type:
                # Classify the task based on the last user message
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        task_type = self.classify_task(msg.get("content", ""))
                        break
                
                if not task_type:
                    task_type = TaskType.GENERAL
            
            selected_model_type = self.task_model_mapping.get(task_type, ModelType.REASONING)
        
        # Get the appropriate model
        model = self.get_model_for_task(task_type) if task_type else (
            self.coding_model if selected_model_type == ModelType.CODING else self.reasoning_model
        )
        
        # Format messages for the model
        prompt = messages[-1]["content"] if messages else ""
        system_prompt = None
        
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content")
                break
        
        # Add model-specific system prompt enhancements if not already present
        if system_prompt:
            if selected_model_type == ModelType.REASONING and "reasoning assistant" not in system_prompt.lower():
                system_prompt = f"You are a reasoning assistant that provides thoughtful, accurate responses. {system_prompt}"
            elif selected_model_type == ModelType.CODING and "coding assistant" not in system_prompt.lower():
                system_prompt = f"You are a coding assistant that provides clean, efficient, and correct code. {system_prompt}"
        else:
            if selected_model_type == ModelType.REASONING:
                system_prompt = "You are a reasoning assistant that provides thoughtful, accurate responses."
            else:
                system_prompt = "You are a coding assistant that provides clean, efficient, and correct code."
        
        # Track performance
        start_time = time.time()
        
        # Generate response
        metadata = {
            "agent_id": self.agent_id,
            "task_type": task_type.value if task_type else "unknown",
            "model_type": selected_model_type.value,
        }
        
        interaction_type = f"{selected_model_type.value}_{task_type.value if task_type else 'general'}"
        
        response = model.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            interaction_type=interaction_type,
            metadata=metadata,
            **kwargs
        )
        
        elapsed_time = time.time() - start_time
        
        # Update statistics
        self.model_stats[selected_model_type]["calls"] += 1
        self.model_stats[selected_model_type]["time"] += elapsed_time
        # TODO: Add token counting for more detailed stats
        
        logger.debug(f"Generated response using {selected_model_type.value} model for {task_type.value if task_type else 'general'} task")
        
        return response
    
    async def generate_response_async(
        self, 
        messages: List[Dict[str, str]],
        task_type: Optional[TaskType] = None,
        model_type: Optional[ModelType] = None,
        **kwargs
    ) -> str:
        """
        Generate a response asynchronously using the appropriate model.
        
        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.
            task_type (TaskType, optional): The type of task. If not provided, it will be classified.
            model_type (ModelType, optional): Force a specific model type. Overrides task_type.
            **kwargs: Additional parameters to pass to the model.
            
        Returns:
            str: The generated response.
        """
        # Determine which model to use (same logic as synchronous version)
        if model_type:
            selected_model_type = model_type
        else:
            if not task_type:
                # Classify the task based on the last user message
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        task_type = self.classify_task(msg.get("content", ""))
                        break
                
                if not task_type:
                    task_type = TaskType.GENERAL
            
            selected_model_type = self.task_model_mapping.get(task_type, ModelType.REASONING)
        
        # Get the appropriate model
        model = self.get_model_for_task(task_type) if task_type else (
            self.coding_model if selected_model_type == ModelType.CODING else self.reasoning_model
        )
        
        # Format messages for the model
        prompt = messages[-1]["content"] if messages else ""
        system_prompt = None
        
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content")
                break
        
        # Add model-specific system prompt enhancements if not already present
        if system_prompt:
            if selected_model_type == ModelType.REASONING and "reasoning assistant" not in system_prompt.lower():
                system_prompt = f"You are a reasoning assistant that provides thoughtful, accurate responses. {system_prompt}"
            elif selected_model_type == ModelType.CODING and "coding assistant" not in system_prompt.lower():
                system_prompt = f"You are a coding assistant that provides clean, efficient, and correct code. {system_prompt}"
        else:
            if selected_model_type == ModelType.REASONING:
                system_prompt = "You are a reasoning assistant that provides thoughtful, accurate responses."
            else:
                system_prompt = "You are a coding assistant that provides clean, efficient, and correct code."
        
        # Track performance
        start_time = time.time()
        
        # Generate response asynchronously
        response = await model.agenerate(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs
        )
        
        elapsed_time = time.time() - start_time
        
        # Update statistics
        self.model_stats[selected_model_type]["calls"] += 1
        self.model_stats[selected_model_type]["time"] += elapsed_time
        
        logger.debug(f"Generated async response using {selected_model_type.value} model for {task_type.value if task_type else 'general'} task")
        
        return response
    
    def generate_with_planning(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Tuple[str, str]:
        """
        Generate a response using a two-step process:
        1. Use the reasoning model to create a plan
        2. Use the coding model to implement the plan
        
        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.
            **kwargs: Additional parameters to pass to the models.
            
        Returns:
            Tuple[str, str]: The planning response and the implementation response.
        """
        # Get the original query
        original_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                original_query = msg.get("content", "")
                break
        
        if not original_query:
            return "No query provided", "No query provided"
        
        # Step 1: Generate a plan using the reasoning model
        planning_query = f"Create a detailed plan for implementing this: {original_query}. Break down the solution into clear, numbered steps that can be followed to write the code. Focus only on planning, not implementation."
        
        plan = self.generate_response(
            messages=[{"role": "user", "content": planning_query}],
            model_type=ModelType.REASONING,
            **kwargs
        )
        
        # Step 2: Implement the plan using the coding model
        implementation_query = f"Implement the following plan in code:\n\n{plan}\n\nThe original request was: {original_query}"
        
        implementation = self.generate_response(
            messages=[{"role": "user", "content": implementation_query}],
            model_type=ModelType.CODING,
            **kwargs
        )
        
        return plan, implementation
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the models.
        
        Returns:
            Dict[str, Any]: Performance statistics.
        """
        return self.model_stats
    
    def update_model_config(self, model_type: ModelType, config_updates: Dict[str, Any]) -> None:
        """
        Update the configuration for a specific model.
        
        Args:
            model_type (ModelType): The type of model to update.
            config_updates (Dict[str, Any]): The configuration updates.
        """
        if model_type == ModelType.CODING:
            self.coding_model_config.update(config_updates)
            self.coding_model = get_llm(**self.coding_model_config)
        else:
            self.reasoning_model_config.update(config_updates)
            self.reasoning_model = get_llm(**self.reasoning_model_config)
        
        logger.info(f"Updated configuration for {model_type.value} model")


def get_model_manager(agent_id: str = "default") -> ModelManager:
    """
    Factory function to get a ModelManager instance.
    
    Args:
        agent_id (str): The ID of the agent.
        
    Returns:
        ModelManager: A ModelManager instance.
    """
    return ModelManager(agent_id=agent_id) 