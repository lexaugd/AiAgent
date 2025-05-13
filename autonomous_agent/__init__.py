"""
Autonomous Coding Agent

A locally-hosted autonomous coding agent with advanced memory systems,
self-improvement capabilities, and multi-agent collaboration.
Features a dual-model architecture with specialized models for reasoning and coding.
"""

__version__ = "0.1.0"

# Main components
from .agents import BaseAgent, CodingAgent, get_agent
from .models import get_llm, get_model_manager, ModelType, TaskType
from .memory import get_memory_manager, get_short_term_memory, get_long_term_memory 