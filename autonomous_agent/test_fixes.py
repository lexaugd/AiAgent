#!/usr/bin/env python3
"""
Test script to verify fixes for the logger issues and other system errors.
"""
import os
import sys
from pathlib import Path

# Ensure the current directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary modules
from autonomous_agent.utils.logger import setup_model_interaction_logger
from autonomous_agent.models.llm_interface import get_llm
from autonomous_agent.agents.coding_agent import CodingAgent

def main():
    """Test the fixes we've made to the system."""
    print("Testing fixes for the autonomous agent system...")
    
    # Test model_logger
    print("\n1. Testing model_logger...")
    model_logger = setup_model_interaction_logger()
    model_logger(
        interaction_type="test",
        prompt="This is a test prompt",
        response="This is a test response",
        metadata={"test": True, "source": "test_fixes.py"}
    )
    print("✓ Model logger test completed - check logs/model_interactions.jsonl")
    
    # Test LLM interface
    print("\n2. Testing LLM interface...")
    try:
        llm = get_llm(model_name="wizard-vicuna-uncensored:latest")
        print(f"✓ LLM interface initialized with model: {llm.model_name}")
    except Exception as e:
        print(f"✗ Error initializing LLM interface: {e}")
    
    # Test CodingAgent
    print("\n3. Testing CodingAgent initialization...")
    try:
        agent = CodingAgent(name="TestAgent", llm=llm)
        print(f"✓ CodingAgent initialized with name: {agent.name} and ID: {agent.id}")
    except Exception as e:
        print(f"✗ Error initializing CodingAgent: {e}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 