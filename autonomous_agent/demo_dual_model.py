"""
Demo script for the dual-model architecture.

This script demonstrates the ModelManager's dual-model architecture in action,
showing how it uses different models for different types of tasks.
"""

import os
import sys
import time
from typing import Dict, List, Any
from loguru import logger

# Add the parent directory to the path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the ModelManager
from autonomous_agent.models.model_manager import (
    ModelManager, ModelType, TaskType, get_model_manager
)

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")


def run_demo():
    """Run the demo showcasing the dual-model architecture."""
    logger.info("=== Dual-Model Architecture Demo ===\n")
    logger.info("This demo shows how different models are used for different tasks.")
    
    # Initialize the model manager
    model_manager = get_model_manager("demo")
    
    # Define some example tasks
    tasks = [
        {
            "type": "explanation",
            "query": "Explain how a hash table works and its time complexity",
            "description": "Reasoning task - explaining a concept"
        },
        {
            "type": "code_generation",
            "query": "Write a Python function to check if a string is a palindrome",
            "description": "Coding task - generating code"
        },
        {
            "type": "problem_solving",
            "query": "How would you debug a memory leak in a Python application?",
            "description": "Reasoning task - problem solving"
        },
        {
            "type": "planning",
            "query": "Design an API for a bookstore management system",
            "description": "Reasoning task - architectural planning"
        },
        {
            "type": "code_review",
            "query": "Review this code for efficiency issues: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "description": "Coding task - code review"
        }
    ]
    
    # Process each task
    for i, task in enumerate(tasks):
        logger.info(f"\n=== Task {i+1}: {task['description']} ===")
        logger.info(f"Query: {task['query']}")
        
        # Classify the task
        task_type = model_manager.classify_task(task['query'])
        logger.info(f"Classified as: {task_type.value}")
        
        # Determine which model will be used
        model_type = model_manager.task_model_mapping.get(task_type, ModelType.REASONING)
        logger.info(f"Using model: {model_type.value.upper()}")
        
        # Generate response
        start_time = time.time()
        response = model_manager.generate_response(
            messages=[{"role": "user", "content": task['query']}],
            task_type=task_type
        )
        elapsed_time = time.time() - start_time
        
        # Show stats
        logger.info(f"Response time: {elapsed_time:.2f}s, Length: {len(response)} chars")
        
        # Show response preview
        preview = response[:200] + "..." if len(response) > 200 else response
        logger.info(f"Response preview:\n{preview}\n")
        
        # Small delay to make the output easier to follow
        time.sleep(1)
    
    # Now demonstrate the combined planning and implementation
    logger.info("\n=== Combined Planning and Implementation ===")
    complex_task = "Create a Python class for managing a library with books and members"
    logger.info(f"Complex task: {complex_task}")
    
    # Use the planning and implementation approach
    start_time = time.time()
    plan, implementation = model_manager.generate_with_planning(
        messages=[{"role": "user", "content": complex_task}]
    )
    elapsed_time = time.time() - start_time
    
    # Show plan preview
    plan_preview = plan[:200] + "..." if len(plan) > 200 else plan
    logger.info(f"\nPlan preview (from REASONING model):\n{plan_preview}")
    
    # Show implementation preview
    impl_preview = implementation[:200] + "..." if len(implementation) > 200 else implementation
    logger.info(f"\nImplementation preview (from CODING model):\n{impl_preview}")
    
    # Show total time
    logger.info(f"\nTotal time for planning and implementation: {elapsed_time:.2f}s")
    
    # Show overall statistics
    stats = model_manager.get_stats()
    logger.info("\n=== Model Usage Statistics ===")
    for model_type, model_stats in stats.items():
        calls = model_stats["calls"]
        avg_time = model_stats["time"] / calls if calls > 0 else 0
        logger.info(f"{model_type.value.capitalize()} model: {calls} calls, avg time: {avg_time:.2f}s")


if __name__ == "__main__":
    run_demo() 