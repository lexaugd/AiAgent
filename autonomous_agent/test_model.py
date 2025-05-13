"""
Test script to verify the functionality of the DeepSeek-Coder model.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from autonomous_agent.models.llm_interface import get_llm

def test_basic_response():
    """Test basic query response."""
    llm = get_llm()
    print(f"Using model: {llm.model_name}")
    
    response = llm.generate(
        prompt="Write a Python function to sort a list of integers using the quicksort algorithm.",
        temperature=0.35
    )
    
    print("\n=== Basic Response Test ===")
    print(response)
    print("=" * 50)

def test_code_reasoning():
    """Test reasoning about code with potential bugs."""
    llm = get_llm()
    
    buggy_code = """
def factorial(n):
    if n <= 0:
        return 1
    else:
        return n * factorial(n)  # Bug: missing -1
        
def binary_search(arr, target):
    left = 0
    right = len(arr)  # Bug: should be len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return -1
"""
    
    prompt = f"""
Analyze the following Python code and identify any bugs or issues:

```python
{buggy_code}
```

For each bug, explain why it's a problem and provide a fix.
"""
    
    response = llm.generate(
        prompt=prompt,
        temperature=0.35
    )
    
    print("\n=== Code Reasoning Test ===")
    print(response)
    print("=" * 50)

def test_complex_problem():
    """Test ability to solve a more complex coding problem."""
    llm = get_llm()
    
    prompt = """
Design a class called LRUCache that implements a Least Recently Used (LRU) cache.

Implement the following methods:
1. __init__(self, capacity): Initialize the LRU cache with positive capacity.
2. get(self, key): Return the value of the key if the key exists, otherwise return -1.
3. put(self, key, value): Update the value of the key if it exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity, evict the least recently used key.

Both get and put operations should have O(1) time complexity.

Provide a complete Python implementation with comments explaining your approach.
"""
    
    response = llm.generate(
        prompt=prompt,
        temperature=0.35
    )
    
    print("\n=== Complex Problem Test ===")
    print(response)
    print("=" * 50)

if __name__ == "__main__":
    test_basic_response()
    test_code_reasoning()
    test_complex_problem() 