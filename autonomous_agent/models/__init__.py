"""
Model-related modules for the Autonomous Coding Agent.
"""

from .llm_interface import LocalLLM, get_llm
from .model_manager import ModelManager, ModelType, TaskType, get_model_manager
from .prompt_templates import get_system_prompt, create_prompt_with_context 