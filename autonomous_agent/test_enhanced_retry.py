#!/usr/bin/env python3
"""
Test script to validate the enhanced retry mechanism for hallucination detection.
"""

import sys
import json
from typing import Dict, List, Any
from loguru import logger

from models.llm_interface import get_llm, _check_for_hallucination_markers, _should_retry_hallucination

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")


def test_hallucination_detection():
    """Test the hallucination detection functions."""
    print("\n=== Testing Hallucination Detection ===")
    
    test_cases = [
        {
            "name": "Acknowledges uncertainty",
            "response": "I'm not sure about the QuantumState library. It doesn't appear to be a widely-used quantum computing library in Python.",
            "query": "How do I use the QuantumState library?",
            "expected": {
                "uncertainty_acknowledged": True,
                "fictional_references": True,
                "excessive_hedging": False
            },
            "should_retry": False
        },
        {
            "name": "Fictional library without uncertainty",
            "response": "The QuantumState library is a powerful tool for quantum computing in Python. First, install it using pip: `pip install quantumstate`",
            "query": "How do I use the QuantumState library?",
            "expected": {
                "uncertainty_acknowledged": False,
                "fictional_references": True,
                "excessive_hedging": False
            },
            "should_retry": True
        },
        {
            "name": "Excessive hedging",
            "response": "I think you might possibly want to consider using the approach that could potentially be more efficient. It's likely that this method might work better in most cases, but I believe it depends on your specific requirements.",
            "query": "What's the best approach for this problem?",
            "expected": {
                "uncertainty_acknowledged": False,
                "fictional_references": False,
                "excessive_hedging": True
            },
            "should_retry": True
        },
        {
            "name": "Normal valid response",
            "response": "To merge two sorted arrays in Python, you can use the following approach:\n\n```python\ndef merge_sorted_arrays(arr1, arr2):\n    result = []\n    i = j = 0\n    while i < len(arr1) and j < len(arr2):\n        if arr1[i] < arr2[j]:\n            result.append(arr1[i])\n            i += 1\n        else:\n            result.append(arr2[j])\n            j += 1\n    result.extend(arr1[i:])\n    result.extend(arr2[j:])\n    return result\n```",
            "query": "Write a function to merge two sorted arrays in Python.",
            "expected": {
                "uncertainty_acknowledged": False,
                "fictional_references": False,
                "excessive_hedging": False
            },
            "should_retry": False
        }
    ]
    
    # Test each case
    for i, case in enumerate(test_cases):
        print(f"\nTest case {i+1}: {case['name']}")
        
        # Test marker detection
        markers = _check_for_hallucination_markers(case["response"], case["query"])
        print("  Detected markers:")
        for key, value in markers.items():
            print(f"    {key}: {value} (Expected: {case['expected'][key]})")
            assert markers[key] == case["expected"][key], f"Expected {case['expected'][key]} for {key}, got {markers[key]}"
        
        # Test retry decision
        should_retry = _should_retry_hallucination(case["response"], case["query"])
        print(f"  Should retry: {should_retry} (Expected: {case['should_retry']})")
        assert should_retry == case["should_retry"], f"Expected should_retry={case['should_retry']}, got {should_retry}"
    
    print("\nAll hallucination detection tests passed!")


def test_retry_mechanism():
    """Test the enhanced retry mechanism with common hallucination triggers."""
    print("\n=== Testing Enhanced Retry Mechanism ===")
    
    # Initialize the LLM
    llm = get_llm()
    print(f"Using model: {llm.model_name}")
    
    # Test cases that might trigger hallucination
    test_queries = [
        "Explain how to use the QuantumState library for quantum computing in Python.",
        "How do I call the /api/v2/neural_synthesis endpoint in my application?",
        "Explain the difference between multiprocessing and multithreading in Python."  # Control case (real concept)
    ]
    
    results = []
    
    # Test with and without retry mechanism
    for query in test_queries:
        print(f"\nTesting query: {query[:50]}...")
        
        # Test without enhanced retry
        print("  Without enhanced retry:")
        response_without_retry = llm.generate(
            prompt=query,
            temperature=0.7,
            max_tokens=300,
            retry_on_hallucination=False
        )
        markers_without_retry = _check_for_hallucination_markers(response_without_retry, query)
        print(f"    Response length: {len(response_without_retry)} chars")
        print(f"    Uncertainty acknowledged: {markers_without_retry['uncertainty_acknowledged']}")
        print(f"    Fictional references: {markers_without_retry['fictional_references']}")
        print(f"    Excessive hedging: {markers_without_retry['excessive_hedging']}")
        
        # Test with enhanced retry
        print("  With enhanced retry:")
        response_with_retry = llm.generate(
            prompt=query,
            temperature=0.7,
            max_tokens=300,
            retry_on_hallucination=True
        )
        markers_with_retry = _check_for_hallucination_markers(response_with_retry, query)
        print(f"    Response length: {len(response_with_retry)} chars")
        print(f"    Uncertainty acknowledged: {markers_with_retry['uncertainty_acknowledged']}")
        print(f"    Fictional references: {markers_with_retry['fictional_references']}")
        print(f"    Excessive hedging: {markers_with_retry['excessive_hedging']}")
        
        # Store results
        results.append({
            "query": query,
            "without_retry": {
                "response": response_without_retry[:300] + "..." if len(response_without_retry) > 300 else response_without_retry,
                "markers": markers_without_retry
            },
            "with_retry": {
                "response": response_with_retry[:300] + "..." if len(response_with_retry) > 300 else response_with_retry,
                "markers": markers_with_retry
            },
            "improvement": {
                "uncertainty_acknowledged": markers_with_retry["uncertainty_acknowledged"] > markers_without_retry["uncertainty_acknowledged"],
                "fictional_references": markers_with_retry["fictional_references"] < markers_without_retry["fictional_references"],
                "excessive_hedging": markers_with_retry["excessive_hedging"] < markers_without_retry["excessive_hedging"]
            }
        })
    
    # Print summary
    print("\nRetry Mechanism Effectiveness Summary:")
    improvements_count = 0
    for result in results:
        improvements = sum(1 for value in result["improvement"].values() if value)
        if improvements > 0:
            improvements_count += 1
            print(f"  Query: {result['query'][:50]}... - {improvements} improvements")
            for key, improved in result["improvement"].items():
                if improved:
                    print(f"    ✓ Improved {key}")
        else:
            print(f"  Query: {result['query'][:50]}... - No improvements")
    
    print(f"\nOverall: Improved {improvements_count}/{len(results)} test cases ({improvements_count/len(results)*100:.0f}%)")
    
    # Save detailed results
    with open("reports/retry_mechanism_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    test_hallucination_detection()
    test_retry_mechanism() 