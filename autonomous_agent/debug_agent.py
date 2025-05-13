#!/usr/bin/env python3
"""
Debug script for testing the agent in a simplified manner.
"""

import sys
import time
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

def test_basic_completion():
    """Test basic completion with the LLM."""
    from models.llm_interface import get_llm
    
    print("\n=== Testing Basic Completion ===")
    
    # Initialize the LLM
    llm = get_llm()
    
    # Test completion
    start_time = time.time()
    response = llm.generate(
        prompt="Write a simple Python function to calculate the factorial of a number.",
        max_tokens=200
    )
    end_time = time.time()
    
    print(f"Response time: {end_time - start_time:.2f} seconds")
    print(f"\nResponse:\n{response}")

def test_with_coding_agent():
    """Test completion with the coding agent."""
    from agents.coding_agent import get_coding_agent
    
    print("\n=== Testing Coding Agent ===")
    
    # Initialize the coding agent
    agent = get_coding_agent()
    
    # Test code generation
    print("\nTesting code generation...")
    start_time = time.time()
    response = agent.generate_code(
        "Write a Python function to check if a number is prime.",
        "python"
    )
    end_time = time.time()
    
    print(f"Response time: {end_time - start_time:.2f} seconds")
    print(f"\nResponse:\n{response}")
    
    # Test code explanation
    print("\nTesting code explanation...")
    code_to_explain = """
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
"""
    
    start_time = time.time()
    response = agent.explain_code(code_to_explain)
    end_time = time.time()
    
    print(f"Response time: {end_time - start_time:.2f} seconds")
    print(f"\nResponse:\n{response}")

def main():
    """Main debug function."""
    print("\n===== Agent Debug Tool =====")
    print("Testing the agent with the fixed LLM interface.")
    
    try:
        # Test basic completion
        test_basic_completion()
        
        # Test with coding agent
        test_with_coding_agent()
        
        print("\nAll tests completed successfully.")
        
    except Exception as e:
        logger.exception(f"Error in debug tests: {e}")
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main() 