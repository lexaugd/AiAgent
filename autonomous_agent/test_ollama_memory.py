#!/usr/bin/env python3
"""
Test script to verify Ollama integration with the memory system.

This script demonstrates:
1. Setting up a connection to an Ollama model
2. Creating a memory manager
3. Adding a few messages to memory
4. Retrieving responses from the model using the memory context
5. Testing context awareness with follow-up questions
"""

import sys
import os
from pathlib import Path

# Add paths to make imports work
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

import requests
import json
import time
from memory.manager import get_memory_manager
from config import MEMORY_CONFIG

def call_ollama_model(messages, model="wizard-vicuna-13b"):
    """Call the Ollama API with the given messages."""
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result["message"]["content"]
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return None

def main():
    """Run the test script to verify Ollama integration with the memory system."""
    print("=== Testing Ollama Integration with Memory System ===\n")
    
    # Initialize the memory manager
    agent_id = f"ollama_test_{int(time.time())}"
    memory_manager = get_memory_manager(agent_id=agent_id)
    
    print(f"Initialized Memory Manager for agent: {agent_id}\n")
    
    # 1. Add system message
    system_message = """You are an AI assistant with access to a memory system.
You have both short-term memory (conversation history) and long-term memory
(knowledge stored in a vector database)."""
    memory_manager.add_message("system", system_message)
    
    # 2. Add some user and assistant messages
    memory_manager.add_message("user", "Hello! Can you tell me about Python list comprehensions?")
    memory_manager.add_message("assistant", "List comprehensions are a concise way to create lists in Python. They provide a way to transform and filter data in a single line of code.")
    memory_manager.add_message("user", "Can you provide an example of filtering with list comprehensions?")
    
    # Get messages to send to model
    messages = memory_manager.get_messages_for_llm()
    print(f"Sending {len(messages)} messages to Ollama model\n")
    
    # 3. Call Ollama API for first response
    assistant_response = call_ollama_model(messages)
    if assistant_response:
        memory_manager.add_message("assistant", assistant_response)
        
        print("=== First Model Response ===")
        print(assistant_response)
    else:
        print("Failed to get response from model")
        return
            
    # 4. Ask a follow-up question to test context awareness
    print("\n=== Testing Follow-up Question ===")
    memory_manager.add_message("user", "Can you modify your example to only include fruits with the letter 'a' in them?")
    
    # Get updated messages and call model again
    follow_up_messages = memory_manager.get_messages_for_llm()
    print(f"Sending follow-up with {len(follow_up_messages)} messages in context\n")
    
    # 5. Call Ollama API for follow-up response
    follow_up_response = call_ollama_model(follow_up_messages)
    if follow_up_response:
        memory_manager.add_message("assistant", follow_up_response)
        
        print("=== Follow-up Response ===")
        print(follow_up_response)
    else:
        print("Failed to get follow-up response from model")
        return
    
    # 6. Test memory retrieval
    print("\n=== Testing Memory Retrieval ===")
    
    # First, refresh context with a query to get relevant information
    context = memory_manager.refresh_context("Python list comprehension filter fruits")
    print(f"Context contains {len(context['relevant_knowledge'])} relevant knowledge items")
    
    # Print memory statistics
    print("\n=== Memory Statistics ===")
    stats = memory_manager.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
        
    # 7. Store a conversation summary in long-term memory
    try:
        print("\n=== Storing Conversation Summary in Long-Term Memory ===")
        summary_id = memory_manager.store_conversation_summary()
        print(f"Stored conversation summary with ID: {summary_id}")
    except Exception as e:
        print(f"Error storing conversation summary: {e}")

if __name__ == "__main__":
    main() 