"""
Test script for the ModelManager implementation.

This script tests the core functionality of the ModelManager:
1. Task classification
2. Response generation with both models
3. Two-step planning and implementation
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
logger.add(sys.stdout, level="INFO")


def test_task_classification() -> bool:
    """Test the task classification functionality."""
    logger.info("Testing task classification...")
    
    model_manager = get_model_manager("test")
    
    test_cases = [
        ("Can you explain how a binary tree works?", TaskType.EXPLANATION),
        ("Write a Python function to sort a list of integers.", TaskType.CODE_GENERATION),
        ("Optimize this code for better performance.", TaskType.CODE_REVIEW),
        ("Fix this bug in my code.", TaskType.PROBLEM_SOLVING),
        ("Design an architecture for a social media application.", TaskType.PLANNING),
        ("What's the capital of France?", TaskType.GENERAL),
    ]
    
    success = True
    
    for query, expected_type in test_cases:
        result = model_manager.classify_task(query)
        match = result == expected_type
        success = success and match
        status = "✅" if match else "❌"
        logger.info(f"{status} Query: '{query}' -> Classified as: {result.value} (Expected: {expected_type.value})")
    
    return success


def test_model_selection() -> bool:
    """Test the model selection based on task type."""
    logger.info("\nTesting model selection...")
    
    model_manager = get_model_manager("test")
    
    test_cases = [
        (TaskType.EXPLANATION, ModelType.REASONING),
        (TaskType.CODE_GENERATION, ModelType.CODING),
        (TaskType.CODE_REVIEW, ModelType.CODING),
        (TaskType.PROBLEM_SOLVING, ModelType.REASONING),
        (TaskType.PLANNING, ModelType.REASONING),
        (TaskType.GENERAL, ModelType.REASONING),
    ]
    
    success = True
    
    for task_type, expected_model_type in test_cases:
        model = model_manager.get_model_for_task(task_type)
        model_type = ModelType.CODING if model == model_manager.coding_model else ModelType.REASONING
        match = model_type == expected_model_type
        success = success and match
        status = "✅" if match else "❌"
        logger.info(f"{status} Task type: {task_type.value} -> Selected model: {model_type.value} (Expected: {expected_model_type.value})")
    
    return success


def test_response_generation() -> bool:
    """Test generating responses with different models."""
    logger.info("\nTesting response generation...")
    
    model_manager = get_model_manager("test")
    
    test_cases = [
        ("Explain how a binary search algorithm works", TaskType.EXPLANATION),
        ("Write a Python function to implement a binary search algorithm", TaskType.CODE_GENERATION),
    ]
    
    success = True
    
    for query, task_type in test_cases:
        messages = [{"role": "user", "content": query}]
        start_time = time.time()
        response = model_manager.generate_response(messages, task_type=task_type)
        elapsed_time = time.time() - start_time
        
        # Basic validation - response is not empty and reasonably sized
        valid = len(response) > 50
        success = success and valid
        status = "✅" if valid else "❌"
        
        # Determine which model was used
        model_type = model_manager.task_model_mapping.get(task_type, ModelType.REASONING)
        
        logger.info(f"{status} Generated response using {model_type.value} model for: '{query}'")
        logger.info(f"Time taken: {elapsed_time:.2f}s, Response length: {len(response)} chars")
        logger.info(f"First 150 chars of response: {response[:150]}...\n")
    
    return success


def test_planning_implementation() -> bool:
    """Test the two-step planning and implementation process."""
    logger.info("\nTesting planning and implementation...")
    
    model_manager = get_model_manager("test")
    
    query = "Create a Python class for a simple blog with posts and comments"
    messages = [{"role": "user", "content": query}]
    
    start_time = time.time()
    plan, implementation = model_manager.generate_with_planning(messages)
    elapsed_time = time.time() - start_time
    
    # Basic validation - responses are not empty and reasonably sized
    valid_plan = len(plan) > 50
    valid_implementation = len(implementation) > 50 and "class" in implementation.lower()
    success = valid_plan and valid_implementation
    status = "✅" if success else "❌"
    
    logger.info(f"{status} Generated plan and implementation for: '{query}'")
    logger.info(f"Time taken: {elapsed_time:.2f}s")
    logger.info(f"Plan length: {len(plan)} chars, Implementation length: {len(implementation)} chars")
    logger.info(f"First 150 chars of plan: {plan[:150]}...")
    logger.info(f"First 150 chars of implementation: {implementation[:150]}...")
    
    return success


def main() -> None:
    """Run all tests."""
    logger.info("=== Model Manager Test Suite ===\n")
    
    results = [
        ("Task Classification", test_task_classification()),
        ("Model Selection", test_model_selection()),
        ("Response Generation", test_response_generation()),
        ("Planning & Implementation", test_planning_implementation()),
    ]
    
    # Print summary
    logger.info("\n=== Test Results Summary ===")
    
    all_passed = True
    for name, result in results:
        status = "PASS" if result else "FAIL"
        all_passed = all_passed and result
        logger.info(f"{name}: {status}")
    
    # Print stats
    model_manager = get_model_manager("test")
    stats = model_manager.get_stats()
    
    logger.info("\n=== Model Stats ===")
    for model_type, model_stats in stats.items():
        if model_stats["calls"] > 0:
            avg_time = model_stats["time"] / model_stats["calls"]
            logger.info(f"{model_type.value.capitalize()} model: {model_stats['calls']} calls, avg time: {avg_time:.2f}s")
    
    # Final result
    logger.info(f"\nOverall result: {'PASS' if all_passed else 'FAIL'}")


if __name__ == "__main__":
    main() 