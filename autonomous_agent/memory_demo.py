#!/usr/bin/env python3
"""
Memory System Demo Script

This script demonstrates how the autonomous agent's memory system works in practice,
including storing and retrieving code examples and concepts.
"""

import time
from pathlib import Path
import sys
from typing import List, Dict, Any
from loguru import logger

# Add parent directory to path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import memory components
from autonomous_agent.memory import (
    MemoryManager,
    get_memory_manager,
    MemoryItem,
    MemoryType
)

def main():
    """Run the memory system demo."""
    print("=== Autonomous Agent Memory System Demo ===\n")
    
    # Initialize the memory manager with a unique ID for this demo
    agent_id = f"demo_agent_{int(time.time())}"
    memory_manager = get_memory_manager(agent_id=agent_id)
    
    print(f"Initialized Memory Manager for agent: {agent_id}\n")
    
    # 1. Simulate a conversation with the agent
    print("=== Simulating a Conversation ===")
    memory_manager.add_message("user", "I need help with Python list comprehensions")
    memory_manager.add_message("assistant", "List comprehensions are a concise way to create lists in Python. What specific help do you need?")
    memory_manager.add_message("user", "How can I filter a list using comprehensions?")
    memory_manager.add_message("assistant", "You can add a conditional clause to filter items. For example: [x for x in numbers if x > 5]")
    memory_manager.add_message("user", "Could you show me an example with strings?")
    
    # Get the conversation history
    history = memory_manager.get_conversation_history()
    print(f"Conversation has {len(history)} messages\n")
    
    # 2. Store some knowledge in long-term memory
    print("=== Storing Knowledge in Long-Term Memory ===")
    
    # Store a code example
    list_comp_code = """
# Basic list comprehension
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = [x for x in numbers if x % 2 == 0]
print(even_numbers)  # Output: [2, 4, 6, 8, 10]

# List comprehension with strings
fruits = ['apple', 'banana', 'cherry', 'date', 'elderberry']
long_fruits = [fruit for fruit in fruits if len(fruit) > 5]
print(long_fruits)  # Output: ['banana', 'cherry', 'elderberry']
"""
    
    code_id = memory_manager.add_to_long_term(
        list_comp_code,
        item_type="code",
        metadata={
            "language": "python",
            "topic": "list comprehension",
            "difficulty": "beginner",
            "tags": ["python", "list", "comprehension", "filtering"]
        }
    )
    print(f"Stored code example with ID: {code_id}")
    
    # Store a concept explanation
    list_comp_concept = """
List comprehensions provide a concise way to create lists based on existing lists (or other iterables).
The syntax is: [expression for item in iterable if condition].
- expression: what to do with each item
- item: the variable representing each element in the iterable
- iterable: a list, tuple, set, etc. to iterate over
- condition: optional filter to only include some items

List comprehensions are more readable and often faster than equivalent for loops.
"""
    
    concept_id = memory_manager.add_to_long_term(
        list_comp_concept,
        item_type="concept",
        metadata={
            "topic": "list comprehension",
            "language": "python",
            "difficulty": "beginner",
            "tags": ["python", "list", "comprehension", "syntax"]
        }
    )
    print(f"Stored concept explanation with ID: {concept_id}")
    
    # 3. Retrieve knowledge based on a query
    print("\n=== Retrieving Knowledge from Long-Term Memory ===")
    
    # Set up a current task in working memory
    memory_manager.set_working_memory("current_task", "Explaining list comprehensions")
    memory_manager.set_working_memory("code_language", "python")
    
    # Simulate a query to retrieve relevant knowledge for the current conversation
    query = "How to filter strings using Python list comprehensions"
    
    print(f"Query: '{query}'")
    retrieved_items = memory_manager.retrieve_relevant(query, n_results=3)
    
    print(f"\nRetrieved {len(retrieved_items)} relevant items:")
    for i, item in enumerate(retrieved_items, 1):
        print(f"\n--- Item {i} ({item.item_type}) ---")
        # Truncate content for display
        content = item.content if len(item.content) < 200 else item.content[:200] + "..."
        print(content)
        print(f"Metadata: {item.metadata}")
    
    # 4. Demonstrate context refreshing
    print("\n=== Refreshing Context ===")
    context = memory_manager.refresh_context(query)
    
    print(f"Context contains:")
    print(f"- {len(context['recent_messages'])} recent messages")
    print(f"- {len(context['relevant_knowledge'])} relevant knowledge items")
    print(f"- {len(context['working_memory'])} working memory items")
    
    # 5. Show memory statistics
    print("\n=== Memory Statistics ===")
    stats = memory_manager.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main() 