#!/usr/bin/env python3
"""
Test script for the memory system.

This script tests the key components of the memory system:
1. Short-term memory
2. Long-term memory with ChromaDB
3. Memory manager coordination
4. Advanced retrieval mechanisms
"""

import time
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

# Add parent directory to path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import memory components
from autonomous_agent.memory import (
    ShortTermMemory,
    LongTermMemory,
    MemoryManager,
    MemoryItem,
    MemoryType,
    MemoryPriority,
    ExtendedMemoryItem,
    MemoryMetadata,
    ContextAwareRetrieval,
    get_memory_manager
)

from autonomous_agent.memory.embeddings import EmbeddingGenerator, CodeChunker


def test_short_term_memory():
    """Test the short-term memory functionality."""
    print("\n=== Testing Short-Term Memory ===")
    
    # Create a short-term memory instance with a unique ID
    memory_id = f"test_{int(time.time())}"
    short_term = ShortTermMemory(conversation_id=memory_id)
    
    # Add some test messages
    print("Adding messages to short-term memory...")
    short_term.add_user_message("Hello, I need help with a Python function")
    short_term.add_assistant_message("Sure, I can help with Python. What function are you working on?")
    short_term.add_user_message("I'm trying to write a function that sorts a list of dictionaries by a specific key")
    
    # Get the messages and print them
    messages = short_term.get_messages()
    print(f"Retrieved {len(messages)} messages from memory")
    
    # Test serialization format
    llm_format = short_term.get_messages_for_llm()
    print(f"Messages in LLM format: {len(llm_format)}")
    
    # Test trimming by adding messages until we exceed the token limit
    print("\nTesting token limit management...")
    original_limit = short_term.max_token_limit
    short_term.max_token_limit = 100  # Set a small limit for testing
    
    # Add messages until we exceed the limit
    for i in range(10):
        short_term.add_user_message(f"This is test message {i} to test the token limit feature")
    
    # Verify that some messages were trimmed
    trimmed_messages = short_term.get_messages()
    print(f"After trimming: {len(trimmed_messages)} messages remain")
    
    # Clear the memory
    short_term.clear()
    print(f"After clearing: {len(short_term.get_messages())} messages")
    
    # Restore the original limit
    short_term.max_token_limit = original_limit
    
    return True


def test_long_term_memory():
    """Test the long-term memory functionality."""
    print("\n=== Testing Long-Term Memory ===")
    
    # Create a long-term memory instance with a unique collection
    collection_name = f"test_collection_{int(time.time())}"
    long_term = LongTermMemory(collection_name=collection_name)
    
    print(f"Created long-term memory with collection: {collection_name}")
    
    # Add some test items
    print("\nAdding items to long-term memory...")
    
    # Code example
    python_code = """
def sort_dicts_by_key(dict_list, key):
    \"\"\"
    Sort a list of dictionaries by a specific key.
    
    Args:
        dict_list (list): List of dictionaries to sort
        key (str): The key to sort by
        
    Returns:
        list: Sorted list of dictionaries
    \"\"\"
    return sorted(dict_list, key=lambda x: x.get(key, ''))
"""
    
    code_item_id = long_term.add_item(
        python_code, 
        item_type="code", 
        metadata={
            "language": "python",
            "description": "Function to sort dictionaries by key",
            "tags": ["python", "sorting", "dictionaries"]
        }
    )
    print(f"Added code item with ID: {code_item_id}")
    
    # Concept explanation
    concept_text = """
Sorting in Python can be done using the built-in sorted() function or the .sort() method on lists.
The sorted() function returns a new sorted list while .sort() modifies the list in-place.
Both can take a key function to customize the sorting logic.
"""
    
    concept_item_id = long_term.add_item(
        concept_text, 
        item_type="concept", 
        metadata={
            "topic": "Python sorting",
            "difficulty": "beginner",
            "tags": ["python", "sorting", "concepts"]
        }
    )
    print(f"Added concept item with ID: {concept_item_id}")
    
    # Error example
    error_text = """
TypeError: 'NoneType' object is not subscriptable

This error occurs when you try to use subscript notation [] on a None value.
Common causes include:
1. Accessing a dictionary key that doesn't exist using dict[key] instead of dict.get(key)
2. Using a function that returns None when it fails
3. Forgetting to initialize a variable before using it
"""
    
    error_item_id = long_term.add_item(
        error_text, 
        item_type="error", 
        metadata={
            "error_type": "TypeError",
            "tags": ["python", "error", "NoneType"]
        }
    )
    print(f"Added error item with ID: {error_item_id}")
    
    # Test retrieval
    print("\nTesting retrieval...")
    
    # Query for sorting-related items
    sorting_results = long_term.query("python sorting dictionaries", n_results=3)
    print(f"Retrieved {len(sorting_results)} items related to 'python sorting dictionaries'")
    
    # Query for error-related items
    error_results = long_term.query("TypeError None subscriptable", item_type="error", n_results=3)
    print(f"Retrieved {len(error_results)} items related to 'TypeError None subscriptable'")
    
    # Test direct item retrieval
    item = long_term.get_item(code_item_id)
    print(f"Retrieved item by ID: {item.item_id}. Type: {item.item_type}")
    
    # Test metadata filtering
    python_items = long_term.search_by_metadata({"language": "python"})
    print(f"Retrieved {len(python_items)} items with metadata language=python")
    
    # Get collection statistics
    stats = long_term.get_collection_stats()
    print(f"Collection stats: {stats}")
    
    return True


def test_memory_manager():
    """Test the memory manager functionality."""
    print("\n=== Testing Memory Manager ===")
    
    # Create a memory manager with unique IDs
    agent_id = f"test_agent_{int(time.time())}"
    collection_name = f"test_collection_{int(time.time())}"
    
    memory_manager = get_memory_manager(agent_id=agent_id)
    
    # Override the long-term memory to use our test collection
    memory_manager.long_term = LongTermMemory(collection_name=collection_name)
    
    print(f"Created memory manager for agent: {agent_id}")
    
    # Test adding messages to short-term memory
    print("\nTesting message handling...")
    memory_manager.add_message("user", "I need to sort a list of dictionaries in Python")
    memory_manager.add_message("assistant", "I can help with that. You can use the sorted() function with a key parameter.")
    memory_manager.add_message("user", "Can you show me an example?")
    
    # Get conversation history
    history = memory_manager.get_conversation_history()
    print(f"Retrieved {len(history)} messages from conversation history")
    
    # Test adding items to long-term memory
    print("\nTesting long-term memory integration...")
    
    # Add a code example
    code_text = """
# Sort a list of dictionaries by the 'name' key
people = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}, {'name': 'Charlie', 'age': 35}]
sorted_people = sorted(people, key=lambda x: x['name'])
print(sorted_people)
"""
    
    code_id = memory_manager.add_to_long_term(
        code_text, 
        item_type="code", 
        metadata={"language": "python", "topic": "sorting"}
    )
    print(f"Added code example with ID: {code_id}")
    
    # Add a concept
    memory_manager.add_to_long_term(
        "Lambda functions in Python are small anonymous functions defined with the 'lambda' keyword.",
        item_type="concept",
        metadata={"topic": "python lambda", "difficulty": "intermediate"}
    )
    
    # Test semantic chunking of larger code
    print("\nTesting code chunking and storage...")
    large_code = """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load and prepare data
def load_data(filepath):
    \"\"\"Load data from CSV file.\"\"\"
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    \"\"\"Preprocess the data for training.\"\"\"
    # Handle missing values
    data = data.fillna(data.mean())
    
    # Convert categorical variables
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype('category').cat.codes
    
    return data

# Train model
def train_model(X, y):
    \"\"\"Train a random forest model.\"\"\"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    return model, accuracy, report, (X_test, y_test)

# Main function
def main():
    data = load_data('data.csv')
    data = preprocess_data(data)
    
    # Separate features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    model, accuracy, report, test_data = train_model(X, y)
    print(f"Model accuracy: {accuracy:.4f}")
    print(report)
    
    return model

if __name__ == "__main__":
    main()
"""
    
    chunk_ids = memory_manager.add_code_to_long_term(
        large_code, 
        language="python", 
        metadata={"topic": "machine learning", "framework": "scikit-learn"}
    )
    print(f"Large code split into {len(chunk_ids)} chunks and stored")
    
    # Test working memory
    print("\nTesting working memory...")
    memory_manager.set_working_memory("current_task", "Implementing sorting function")
    memory_manager.set_working_memory("code_language", "python")
    
    task = memory_manager.get_working_memory("current_task")
    print(f"Retrieved task from working memory: {task}")
    
    # Test context refresh
    print("\nTesting context refresh...")
    context = memory_manager.refresh_context("python sort dictionaries")
    print(f"Refreshed context with {len(context['relevant_knowledge'])} relevant knowledge items")
    
    # Test memory statistics
    stats = memory_manager.get_stats()
    print(f"Memory statistics: {stats}")
    
    return True


def test_advanced_retrieval():
    """Test the advanced retrieval mechanisms."""
    print("\n=== Testing Advanced Retrieval ===")
    
    # Create a long-term memory instance with a unique collection
    collection_name = f"test_retrieval_{int(time.time())}"
    long_term = LongTermMemory(collection_name=collection_name)
    
    # Create an embedding generator
    embedding_generator = EmbeddingGenerator()
    
    # Create a context-aware retrieval system
    retrieval = ContextAwareRetrieval(
        long_term_memory=long_term,
        embedding_generator=embedding_generator
    )
    
    # Add some test content
    print("\nAdding test content for retrieval...")
    
    # Python code examples
    python_examples = [
        ("""
def fibonacci(n):
    \"\"\"Generate Fibonacci sequence up to n\"\"\"
    a, b = 0, 1
    result = []
    while a < n:
        result.append(a)
        a, b = b, a + b
    return result
""", "Fibonacci sequence generator"),
        
        ("""
def merge_sort(arr):
    \"\"\"Implement merge sort algorithm\"\"\"
    if len(arr) <= 1:
        return arr
        
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)
    
def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
""", "Merge sort implementation"),
        
        ("""
def binary_search(arr, target):
    \"\"\"Implement binary search algorithm\"\"\"
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Target not found
""", "Binary search implementation")
    ]
    
    # JavaScript code examples
    js_examples = [
        ("""
// Fibonacci sequence in JavaScript
function fibonacci(n) {
    let a = 0, b = 1;
    const result = [];
    
    while (a < n) {
        result.push(a);
        [a, b] = [b, a + b];
    }
    
    return result;
}
""", "Fibonacci sequence in JavaScript"),
        
        ("""
// Fetch API example
async function fetchData(url) {
    try {
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Fetch error:', error);
    }
}
""", "JavaScript Fetch API example")
    ]
    
    # Concepts
    concepts = [
        ("Recursion is a programming technique where a function calls itself to solve a problem. It's commonly used in algorithms like tree traversal, factorial calculation, and the Fibonacci sequence.", "Recursion concept"),
        
        ("Time complexity is a measure of how the runtime of an algorithm increases as the size of the input grows. Big O notation (e.g., O(n), O(log n), O(n²)) is used to express this relationship.", "Time complexity concept"),
        
        ("Memoization is an optimization technique where the results of expensive function calls are cached and reused when the same inputs occur again. It's particularly useful for recursive algorithms like Fibonacci.", "Memoization concept")
    ]
    
    # Add Python examples
    for code, description in python_examples:
        long_term.add_item(
            code, 
            item_type="code", 
            metadata={
                "language": "python",
                "description": description,
                "tags": ["python", "algorithm", description.split()[0].lower()]
            }
        )
    
    # Add JavaScript examples
    for code, description in js_examples:
        long_term.add_item(
            code, 
            item_type="code", 
            metadata={
                "language": "javascript",
                "description": description,
                "tags": ["javascript", "algorithm", description.split()[0].lower()]
            }
        )
    
    # Add concepts
    for text, description in concepts:
        long_term.add_item(
            text, 
            item_type="concept", 
            metadata={
                "topic": description.split()[0].lower(),
                "tags": ["concept", "programming", "computer science"]
            }
        )
    
    print(f"Added {len(python_examples)} Python examples, {len(js_examples)} JavaScript examples, and {len(concepts)} concepts")
    
    # Test basic retrieval
    print("\nTesting basic retrieval...")
    result = retrieval.retrieve("fibonacci sequence algorithm", n_results=3)
    print(f"Retrieved {len(result.items)} items for 'fibonacci sequence algorithm'")
    print(f"Query took {result.retrieval_time:.3f} seconds")
    print(f"Expanded queries: {result.metadata.get('expanded_queries', [])[:2]}")
    
    # Test code-specific retrieval
    print("\nTesting code-specific retrieval...")
    result = retrieval.retrieve_code_examples("sort algorithm in python", language="python", n_results=2)
    print(f"Retrieved {len(result.items)} Python sort examples")
    
    # Test with context
    print("\nTesting context-aware retrieval...")
    context = {
        "language": "javascript",
        "current_task": "Implementing API calls"
    }
    result = retrieval.retrieve("fetch data from API", context=context, n_results=2)
    print(f"Retrieved {len(result.items)} items for 'fetch data from API' with JavaScript context")
    
    # Test retrieval by type
    print("\nTesting retrieval by type...")
    type_results = retrieval.retrieve_by_type(
        query="algorithm complexity",
        memory_types=["concept", "code"],
        n_results=2
    )
    
    for memory_type, result in type_results.items():
        print(f"Retrieved {len(result.items)} {memory_type} items for 'algorithm complexity'")
    
    return True


def main():
    """Run the memory system tests."""
    print("=== Memory System Tests ===")
    
    tests = [
        ("Short-Term Memory", test_short_term_memory),
        ("Long-Term Memory", test_long_term_memory),
        ("Memory Manager", test_memory_manager),
        ("Advanced Retrieval", test_advanced_retrieval)
    ]
    
    results = {}
    
    for name, test_func in tests:
        print(f"\nRunning {name} test...")
        try:
            success = test_func()
            results[name] = "Passed" if success else "Failed"
        except Exception as e:
            print(f"Error in {name} test: {e}")
            results[name] = f"Error: {e}"
    
    # Print summary
    print("\n=== Test Summary ===")
    for name, result in results.items():
        print(f"{name}: {result}")


if __name__ == "__main__":
    main() 