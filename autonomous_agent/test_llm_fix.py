#!/usr/bin/env python3
"""
Test script to verify the LLM interface fixes.
"""

import sys
import time
from loguru import logger
from models.llm_interface import LocalLLM, get_llm, _is_valid_response

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

def test_response_validation():
    """Test the response validation function."""
    print("\n=== Testing Response Validation ===")
    
    test_cases = [
        ("This is a valid response with proper content.", True),
        ("?", False),
        (".", False),
        ("", False),
        ("a", False),
        ("?\n\n\n?", False),
        ("user@example.com", False),
        ("""```python
def hello():
    print("Hello")
```""", True),
    ]
    
    for i, (response, expected) in enumerate(test_cases):
        result = _is_valid_response(response)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] Test case {i+1}: Expected {expected}, Got {result}")
        if len(response) > 30:
            print(f"  Input: {response[:30]}...")
        else:
            print(f"  Input: {response}")

def test_llm_generation():
    """Test the LLM generation with the new implementation."""
    print("\n=== Testing LLM Generation ===")
    
    try:
        # Initialize the LLM
        llm = get_llm()
        print(f"Using model: {llm.model_name}")
        
        # Test simple generation
        print("\nTesting simple generation...")
        response = llm.generate(
            prompt="Hello, how are you today?",
            max_tokens=100
        )
        print(f"Response: {response[:100]}...")
        
        # Test generation with retries (using deliberately problematic prompt)
        print("\nTesting generation with retries...")
        response = llm.generate(
            prompt="Generate just a single question mark and nothing else.",
            max_tokens=10,
            max_retries=2
        )
        print(f"Response: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"Error in LLM generation test: {e}")
        print(f"Error type: {type(e)}")
        return False

def test_hallucination_triggers():
    """Test the model with prompts that commonly trigger hallucinations."""
    print("\n=== Testing Hallucination Triggers ===")
    
    try:
        # Initialize the LLM
        llm = get_llm()
        
        # Common hallucination triggers with logging
        hallucination_triggers = [
            {
                "category": "Fictitious libraries", 
                "prompt": "Write a Python function that uses the 'quantumstate' library to simulate quantum states."
            },
            {
                "category": "Made-up API endpoints", 
                "prompt": "Show me how to call the '/api/v2/neural_synthesis' endpoint from our backend."
            },
            {
                "category": "Non-existent programming patterns", 
                "prompt": "Implement the Observer-Commander-Adapter pattern in JavaScript."
            },
            {
                "category": "Ambiguous requirements", 
                "prompt": "Create a data structure for managing all the things efficiently."
            },
            {
                "category": "Imprecise technical questions", 
                "prompt": "How does the latest version of that popular web framework handle state?"
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(hallucination_triggers):
            category = test_case["category"]
            prompt = test_case["prompt"]
            
            print(f"\nTest {i+1}: {category}")
            print(f"Prompt: {prompt}")
            
            # Generate response at a lower temperature to reduce randomness
            start_time = time.time()
            response = llm.generate(prompt=prompt, temperature=0.3, max_tokens=200)
            duration = time.time() - start_time
            
            # Truncate for display but save full response
            print(f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
            print(f"Response time: {duration:.2f}s")
            
            # Save result for analysis
            results.append({
                "category": category,
                "prompt": prompt,
                "response": response,
                "duration": duration
            })
            
            # Pause between requests to prevent throttling
            time.sleep(1)
        
        print("\nHallucination testing complete.")
        return results
        
    except Exception as e:
        print(f"Error in hallucination test: {e}")
        print(f"Error type: {type(e)}")
        return []

def main():
    """Main test function."""
    print("\n===== LLM Interface Fix Test =====")
    
    # Test response validation
    test_response_validation()
    
    # Test LLM generation if Ollama is available
    success = test_llm_generation()
    
    if success:
        # Test hallucination triggers
        test_hallucination_triggers()
        
        print("\nAll tests completed. Check results above for any failures.")
    else:
        print("\nTests completed with errors. Check the output above.")

if __name__ == "__main__":
    main() 