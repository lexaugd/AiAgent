"""
Configuration settings for the Autonomous Coding Agent.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
CONVERSATION_HISTORY_DIR = DATA_DIR / "conversation_history"

# Ensure directories exist
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(CONVERSATION_HISTORY_DIR, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "name": "deepseek-coder:6.7b-instruct",
    "base_url": "http://localhost:11434/v1",  # Ollama API endpoint
    "max_tokens": 4096,
    "temperature": 0.35,  # Set to optimal value for code generation
    "top_p": 0.95,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1,
}

# Reasoning model configuration
REASONING_MODEL_CONFIG = {
    "name": "phi3:mini",
    "base_url": "http://localhost:11434/v1",
    "max_tokens": 4096,
    "temperature": 0.7,  # Higher temperature for more creative reasoning
    "top_p": 0.95,
    "frequency_penalty": 0.2,
    "presence_penalty": 0.2,
}

# Combined models configuration
DUAL_MODEL_CONFIG = {
    "coding": MODEL_CONFIG,
    "reasoning": REASONING_MODEL_CONFIG,
}

# Memory configuration
MEMORY_CONFIG = {
    "short_term": {
        "max_token_limit": 4000,
        "memory_key": "chat_history",
        "return_messages": True,
    },
    "long_term": {
        "collection_name": "code_knowledge",
        "embedding_dimension": 384,  # For all-MiniLM-L6-v2
    },
}

# Learning system configuration
LEARNING_CONFIG = {
    "experience_storage_dir": DATA_DIR / "learning" / "experiences",
    "feedback_storage_dir": DATA_DIR / "learning" / "feedback",
    "reflection_storage_dir": DATA_DIR / "learning" / "reflections",
    "experience_cache_size": 100,
    "feedback_cache_size": 50,
    "knowledge_extraction_threshold": 0.6,
    "embedding_batch_size": 5,
    "reflection_period": 20,  # Number of experiences before auto-reflection
    "auto_reflection_enabled": True,
}

# Learning directories
for dir_path in [
    LEARNING_CONFIG["experience_storage_dir"],
    LEARNING_CONFIG["feedback_storage_dir"],
    LEARNING_CONFIG["reflection_storage_dir"]
]:
    os.makedirs(dir_path, exist_ok=True)

# Agent configuration
AGENT_CONFIG = {
    "verbose": True,
    "max_iterations": 10,
    "max_execution_time": 60,  # seconds
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    "file": BASE_DIR / "logs" / "agent.log",
}

# Web search configuration (optional)
WEB_SEARCH_ENABLED = False
WEB_SEARCH_CONFIG = {
    "search_engine": "duckduckgo",
    "max_results": 5,
}

# Development mode flag
DEV_MODE = True  # Set to False in production 