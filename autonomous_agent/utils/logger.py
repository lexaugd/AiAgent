"""
Logger utilities for the Autonomous Coding Agent.
"""

import os
import sys
import json
import datetime
from pathlib import Path
from loguru import logger
from typing import Callable, Optional, Dict, Any

# Fix import paths
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(MODULE_DIR)
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Import modules
from autonomous_agent.config import LOGGING_CONFIG

def setup_logger(debug=False):
    """
    Configure the logger for the application.
    
    Args:
        debug (bool): Whether to enable debug mode logging
    """
    # Clear default handlers
    logger.remove()
    
    # Set the log level based on debug mode
    log_level = "DEBUG" if debug else LOGGING_CONFIG.get("level", "INFO")
    
    # Ensure log directory exists
    log_file = LOGGING_CONFIG.get("file")
    if log_file:
        log_dir = Path(log_file).parent
        os.makedirs(log_dir, exist_ok=True)
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=LOGGING_CONFIG.get("format"),
        level=log_level,
        colorize=True,
    )
    
    # Add file handler if configured
    if log_file:
        logger.add(
            log_file,
            rotation="10 MB",  # Rotate when the file reaches 10MB
            retention="1 week",  # Keep logs for 1 week
            compression="zip",  # Compress rotated logs
            format=LOGGING_CONFIG.get("format"),
            level=log_level,
        )
    
    logger.debug(f"Logger initialized with level: {log_level}")
    return logger

def setup_model_interaction_logger() -> Callable:
    """
    Set up a logger for model interactions that writes to a JSONL file.
    
    Returns:
        Callable: A function that logs model interactions.
    """
    # Ensure logs directory exists
    os.makedirs("logs/model_interactions", exist_ok=True)
    
    # Create a JSONL file for all interactions
    log_file = "logs/model_interactions.jsonl"
    
    def log_model_interaction(
        interaction_type: str,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a model interaction to the JSONL file.
        
        Args:
            interaction_type (str): The type of interaction (e.g., 'general', 'context_query').
            prompt (str): The prompt sent to the model.
            response (str): The response from the model.
            metadata (Dict[str, Any], optional): Additional metadata about the interaction.
        """
        # Create a log entry
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "interaction_type": interaction_type,
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {}
        }
        
        # Write to the main JSONL file
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to model interaction log: {e}")
            
        # Also write to a dated, typed log file for easier analysis
        try:
            date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
            typed_log_file = f"logs/model_interactions/{interaction_type}_{date_str}.json"
            with open(typed_log_file, "w") as f:
                f.write(json.dumps(log_entry, indent=2))
        except Exception as e:
            logger.error(f"Failed to write to typed model interaction log: {e}")
    
    return log_model_interaction 