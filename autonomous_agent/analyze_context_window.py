#!/usr/bin/env python3
"""
Script for analyzing context window utilization in the memory system.

This script examines how effectively the context window is being used during
memory retrieval and context assembly, which can help identify potential
causes of hallucination due to context limitations.
"""

import os
import sys
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from memory.manager import MemoryManager, get_memory_manager
from memory.types import MemoryPriority
from memory.embeddings import EmbeddingGenerator
from config import MEMORY_CONFIG, MODEL_CONFIG

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a string.
    This is a rough approximation based on word count.
    
    Args:
        text (str): Text to estimate tokens for
        
    Returns:
        int: Estimated token count
    """
    # A rough approximation: tokens ≈ 1.3 * words
    words = text.split()
    return int(len(words) * 1.3)


def analyze_conversation_context(memory_manager: MemoryManager, 
                               conversation_length: int = 10,
                               query: str = "How can I improve performance?") -> Dict[str, Any]:
    """
    Analyze context window utilization in conversation context.
    
    Args:
        memory_manager (MemoryManager): Memory manager instance
        conversation_length (int): Number of messages to simulate
        query (str): Query for context retrieval
        
    Returns:
        Dict[str, Any]: Analysis results
    """
    # Generate a simulated conversation
    roles = ["user", "assistant"]
    for i in range(conversation_length):
        role = roles[i % 2]
        # Create messages of varying lengths to simulate real conversations
        if role == "user":
            length = random.randint(10, 100)
            content = f"User message {i+1} with {length} simulated words. " + " word" * length
        else:
            length = random.randint(50, 200)
            content = f"Assistant response {i+1} with {length} simulated words. " + " word" * length
        
        memory_manager.add_message(role, content)
    
    # Get the conversation history
    messages = memory_manager.get_conversation_history()
    
    # Estimate token counts for each message
    message_tokens = [estimate_tokens(msg["content"]) for msg in messages]
    total_conversation_tokens = sum(message_tokens)
    
    # Get context with relevant items
    context = memory_manager.refresh_context(query)
    
    # Estimate tokens in the context
    recent_message_tokens = sum([estimate_tokens(msg["content"]) for msg in context["recent_messages"]])
    relevant_knowledge_tokens = sum([estimate_tokens(item["content"]) for item in context["relevant_knowledge"]])
    working_memory_tokens = sum([estimate_tokens(str(v)) for v in context["working_memory"].values()])
    
    total_context_tokens = recent_message_tokens + relevant_knowledge_tokens + working_memory_tokens
    
    # Get model context window size
    model_context_size = MODEL_CONFIG.get("max_tokens", 4096)
    
    # Calculate utilization metrics
    conversation_utilization = total_conversation_tokens / model_context_size
    context_utilization = total_context_tokens / model_context_size
    
    # Calculate balance of components in the context
    context_breakdown = {
        "recent_messages": recent_message_tokens / total_context_tokens if total_context_tokens > 0 else 0,
        "relevant_knowledge": relevant_knowledge_tokens / total_context_tokens if total_context_tokens > 0 else 0,
        "working_memory": working_memory_tokens / total_context_tokens if total_context_tokens > 0 else 0
    }
    
    return {
        "conversation_length": conversation_length,
        "total_conversation_tokens": total_conversation_tokens,
        "total_context_tokens": total_context_tokens,
        "model_context_size": model_context_size,
        "conversation_utilization": conversation_utilization,
        "context_utilization": context_utilization,
        "context_breakdown": context_breakdown,
        "message_token_counts": message_tokens,
        "recent_message_tokens": recent_message_tokens,
        "relevant_knowledge_tokens": relevant_knowledge_tokens,
        "working_memory_tokens": working_memory_tokens
    }


def analyze_variable_conversation_sizes(base_agent_id: str = "context_test") -> Dict[str, Any]:
    """
    Analyze how context window utilization changes with conversation length.
    
    Args:
        base_agent_id (str): Base agent ID prefix
        
    Returns:
        Dict[str, Any]: Analysis results
    """
    # Test with different conversation lengths
    conversation_lengths = [5, 10, 20, 30, 50]
    results = {}
    
    for length in conversation_lengths:
        agent_id = f"{base_agent_id}_{length}"
        memory_manager = get_memory_manager(agent_id)
        
        # Analyze with this conversation length
        result = analyze_conversation_context(memory_manager, conversation_length=length)
        results[length] = result
        
        logger.info(f"Conversation length {length}: {result['context_utilization']:.1%} context utilization")
    
    return results


def analyze_query_effect(base_agent_id: str = "query_test") -> Dict[str, Any]:
    """
    Analyze how different query types affect context window utilization.
    
    Args:
        base_agent_id (str): Base agent ID prefix
        
    Returns:
        Dict[str, Any]: Analysis results
    """
    # Set up a standard conversation
    memory_manager = get_memory_manager(base_agent_id)
    conversation_length = 15
    
    # Generate a simulated conversation once
    roles = ["user", "assistant"]
    for i in range(conversation_length):
        role = roles[i % 2]
        if role == "user":
            length = random.randint(10, 100)
            content = f"User message {i+1} with {length} simulated words. " + " word" * length
        else:
            length = random.randint(50, 200)
            content = f"Assistant response {i+1} with {length} simulated words. " + " word" * length
        
        memory_manager.add_message(role, content)
    
    # Test with different query types
    query_types = {
        "specific_technical": "How do I implement parallel processing in Python?",
        "ambiguous": "How can I make my code better?",
        "novel_concept": "How do I implement quantum algorithms?",
        "context_dependent": "Could you explain that in more detail?"
    }
    
    results = {}
    
    for query_type, query in query_types.items():
        # Analyze with this query
        result = analyze_conversation_context(memory_manager, 
                                            conversation_length=0,  # Don't add more messages
                                            query=query)
        results[query_type] = result
        
        logger.info(f"Query type '{query_type}': {result['context_utilization']:.1%} context utilization")
    
    return results


def visualize_results(variable_sizes_results: Dict[str, Any], 
                     query_results: Dict[str, Any], 
                     output_dir: str = "reports") -> None:
    """
    Visualize the context window utilization analysis results.
    
    Args:
        variable_sizes_results (Dict[str, Any]): Results from variable conversation size analysis
        query_results (Dict[str, Any]): Results from query effect analysis
        output_dir (str): Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot conversation length vs. context utilization
    conversation_lengths = sorted(variable_sizes_results.keys())
    utilization_values = [variable_sizes_results[length]["context_utilization"] for length in conversation_lengths]
    
    plt.figure(figsize=(10, 6))
    plt.plot(conversation_lengths, utilization_values, marker='o', linewidth=2)
    plt.xlabel('Conversation Length (messages)')
    plt.ylabel('Context Window Utilization')
    plt.title('Context Window Utilization vs. Conversation Length')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, length in enumerate(conversation_lengths):
        plt.text(length, utilization_values[i] + 0.01, f"{utilization_values[i]:.1%}", 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'context_vs_conversation_length.png'))
    plt.close()
    
    # 2. Plot context breakdown by conversation length
    lengths = sorted(variable_sizes_results.keys())
    recent_msg_pct = [variable_sizes_results[length]["context_breakdown"]["recent_messages"] for length in lengths]
    relevant_knowledge_pct = [variable_sizes_results[length]["context_breakdown"]["relevant_knowledge"] for length in lengths]
    working_memory_pct = [variable_sizes_results[length]["context_breakdown"]["working_memory"] for length in lengths]
    
    plt.figure(figsize=(12, 6))
    width = 0.8
    
    # Create a stacked bar chart
    bottom_vals = np.zeros(len(lengths))
    
    p1 = plt.bar(lengths, recent_msg_pct, width, label='Recent Messages', color='#3498db')
    bottom_vals = np.add(bottom_vals, recent_msg_pct)
    
    p2 = plt.bar(lengths, relevant_knowledge_pct, width, bottom=bottom_vals, label='Relevant Knowledge', color='#2ecc71')
    bottom_vals = np.add(bottom_vals, relevant_knowledge_pct)
    
    p3 = plt.bar(lengths, working_memory_pct, width, bottom=bottom_vals, label='Working Memory', color='#e74c3c')
    
    plt.xlabel('Conversation Length (messages)')
    plt.ylabel('Proportion of Context')
    plt.title('Context Composition by Conversation Length')
    plt.xticks(lengths)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    # Add percentage labels to the middle of each segment
    for i, length in enumerate(lengths):
        # Recent messages label
        if recent_msg_pct[i] > 0.05:  # Only add label if segment is large enough
            plt.text(i+1, recent_msg_pct[i]/2, f"{recent_msg_pct[i]:.0%}", 
                    ha='center', va='center', color='white')
        
        # Relevant knowledge label
        if relevant_knowledge_pct[i] > 0.05:
            plt.text(i+1, bottom_vals[i] - relevant_knowledge_pct[i]/2, f"{relevant_knowledge_pct[i]:.0%}", 
                    ha='center', va='center', color='white')
        
        # Working memory label
        if working_memory_pct[i] > 0.05:
            plt.text(i+1, bottom_vals[i] + working_memory_pct[i]/2, f"{working_memory_pct[i]:.0%}", 
                    ha='center', va='center', color='white')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the legend
    plt.savefig(os.path.join(output_dir, 'context_composition.png'))
    plt.close()
    
    # 3. Plot query type effect on context utilization
    query_types = list(query_results.keys())
    query_utilization = [query_results[qt]["context_utilization"] for qt in query_types]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(query_types, query_utilization, color='#9b59b6')
    plt.xlabel('Query Type')
    plt.ylabel('Context Window Utilization')
    plt.title('Context Utilization by Query Type')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.1%}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'context_by_query_type.png'))
    plt.close()
    
    # 4. Save the raw data
    combined_results = {
        "variable_sizes_analysis": variable_sizes_results,
        "query_type_analysis": query_results
    }
    
    with open(os.path.join(output_dir, 'context_window_analysis.json'), 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Main function to run the context window utilization analysis."""
    
    # Analyze with variable conversation sizes
    logger.info("Analyzing context utilization with variable conversation sizes")
    variable_sizes_results = analyze_variable_conversation_sizes()
    
    # Analyze query effect
    logger.info("Analyzing query type effect on context utilization")
    query_results = analyze_query_effect()
    
    # Visualize results
    visualize_results(variable_sizes_results, query_results)
    
    # Print summary
    print("\n=== Context Window Utilization Analysis Summary ===")
    
    print("\nContext utilization by conversation length:")
    for length, data in sorted(variable_sizes_results.items()):
        print(f"  {length} messages: {data['context_utilization']:.1%} utilization, " +
              f"{data['total_context_tokens']} tokens / {data['model_context_size']} max tokens")
    
    print("\nContext utilization by query type:")
    for query_type, data in query_results.items():
        print(f"  {query_type}: {data['context_utilization']:.1%} utilization, " +
              f"{data['total_context_tokens']} tokens / {data['model_context_size']} max tokens")
    
    print(f"\nDetailed results saved to reports/context_window_analysis.json")


if __name__ == "__main__":
    main() 