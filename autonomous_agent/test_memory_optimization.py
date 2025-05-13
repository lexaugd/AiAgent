#!/usr/bin/env python3
"""
Test script to verify memory optimization and measure context window utilization.
"""

import time
import json
import os
import sys
import uuid
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
from loguru import logger

from memory.manager import get_memory_manager
from memory.memory_optimizer import MemoryOptimizer, QueryType, QueryComplexity
from models.llm_interface import get_llm
from config import MODEL_CONFIG


def setup_logger():
    """Set up test logger."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    return logger


def test_query_classification():
    """Test the query classification functionality."""
    logger.info("Testing query classification")
    
    # Initialize memory optimizer
    optimizer = MemoryOptimizer()
    
    # Test cases
    test_queries = [
        "How do I implement a binary search tree in Python?",
        "Can you explain this code?",
        "What's the best way to improve my application's performance?",
        "Tell me about quantum computing algorithms",
        "How do I use the parallel processing library?"
    ]
    
    results = []
    
    for query in test_queries:
        # For testing, use an empty conversation history
        query_type, complexity = optimizer.classify_query(query, [])
        
        # Get token allocations
        allocations = optimizer.get_token_allocations(query_type, complexity)
        
        # Print and store the results
        logger.info(f"Query: {query}")
        logger.info(f"  Classification: {query_type.name}, {complexity.name}")
        logger.info(f"  Token allocations: {allocations}")
        logger.info("")
        
        results.append({
            "query": query,
            "query_type": query_type.name,
            "complexity": complexity.name,
            "allocations": allocations
        })
    
    # Save the results to a file
    os.makedirs("reports", exist_ok=True)
    with open("reports/query_classification_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def test_context_utilization():
    """Test the context window utilization with and without optimization."""
    logger.info("Testing context window utilization")
    
    # Create a unique ID for this test session
    session_id = f"test_optimization_{uuid.uuid4().hex[:8]}"
    
    # Initialize memory manager with the session ID
    memory_manager = get_memory_manager(session_id)
    
    # Initialize the memory optimizer
    model_context_size = MODEL_CONFIG.get("max_tokens", 4096)
    optimizer = MemoryOptimizer(model_context_size=model_context_size)
    
    # Test queries of different types
    test_cases = [
        {
            "name": "Specific Technical",
            "query": "How do I implement a binary search tree in Python?",
            "message_count": 10
        },
        {
            "name": "Context Dependent",
            "query": "Can you explain how this code works?",
            "message_count": 15
        },
        {
            "name": "Ambiguous",
            "query": "How can I make my code better?",
            "message_count": 10
        },
        {
            "name": "Novel Concept",
            "query": "How do I implement quantum computing algorithms?",
            "message_count": 10
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        query = test_case["query"]
        message_count = test_case["message_count"]
        logger.info(f"Test case: {test_case['name']}")
        logger.info(f"  Query: {query}")
        
        # Clear the memory manager for this test
        memory_manager.clear_short_term()
        
        # Add some simulated conversation history
        for i in range(message_count):
            role = "user" if i % 2 == 0 else "assistant"
            content_length = 100 if role == "user" else 200
            content = f"{role.capitalize()} message {i+1} with simulated content. " + "word " * content_length
            memory_manager.add_message(role, content)
        
        # Measure without optimization - use the original refresh_context method
        start_time = time.time()
        original_context = memory_manager.refresh_context(query)
        original_time = time.time() - start_time
        
        # Calculate original token usage
        original_messages = original_context["recent_messages"]
        original_knowledge = original_context["relevant_knowledge"]
        original_message_content = "\n".join([msg["content"] for msg in original_messages])
        original_knowledge_content = "\n".join([item["content"] for item in original_knowledge])
        
        original_tokens = optimizer.estimate_tokens(original_message_content) + optimizer.estimate_tokens(original_knowledge_content)
        original_utilization = original_tokens / model_context_size
        
        logger.info(f"  Without optimization:")
        logger.info(f"    Time: {original_time:.3f} seconds")
        logger.info(f"    Context utilization: {original_utilization:.1%}")
        logger.info(f"    Messages: {len(original_messages)}")
        logger.info(f"    Knowledge items: {len(original_knowledge)}")
        
        # Now test with our optimization
        system_prompt = "You are a helpful AI assistant."
        start_time = time.time()
        
        # Get messages and knowledge items
        messages = memory_manager.short_term.get_messages()
        knowledge_items = memory_manager.long_term.query(query, n_results=10)
        
        # Use the memory optimizer to build an optimized context
        optimized_context = optimizer.build_optimized_context(
            query=query,
            system_prompt=system_prompt,
            messages=messages,
            knowledge_items=knowledge_items,
            working_memory={}
        )
        optimization_time = time.time() - start_time
        
        # Calculate optimized token usage
        optimized_message_content = "\n".join([msg.content for msg in optimized_context["messages"]])
        optimized_knowledge_content = "\n".join([item.content for item in optimized_context["knowledge_items"]])
        
        optimized_tokens = (
            optimizer.estimate_tokens(system_prompt) +
            optimizer.estimate_tokens(optimized_message_content) + 
            optimizer.estimate_tokens(optimized_knowledge_content)
        )
        metrics = optimized_context["metrics"]
        
        logger.info(f"  With optimization:")
        logger.info(f"    Time: {optimization_time:.3f} seconds")
        logger.info(f"    Context utilization: {metrics['utilization']:.1%}")
        logger.info(f"    Query classification: {metrics['query_type']}, {metrics['complexity']}")
        logger.info(f"    Messages: {len(optimized_context['messages'])}")
        logger.info(f"    Knowledge items: {len(optimized_context['knowledge_items'])}")
        
        # Calculate improvement
        utilization_improvement = (metrics['utilization'] / original_utilization - 1) * 100
        logger.info(f"  Improvement: {utilization_improvement:.1f}% higher utilization")
        logger.info("")
        
        # Store the results
        results.append({
            "test_case": test_case["name"],
            "query": query,
            "message_count": message_count,
            "original": {
                "time": original_time,
                "tokens": original_tokens,
                "utilization": original_utilization,
                "message_count": len(original_messages),
                "knowledge_count": len(original_knowledge)
            },
            "optimized": {
                "time": optimization_time,
                "tokens": optimized_tokens,
                "utilization": metrics["utilization"],
                "query_type": metrics["query_type"],
                "complexity": metrics["complexity"],
                "message_count": len(optimized_context["messages"]),
                "knowledge_count": len(optimized_context["knowledge_items"])
            },
            "improvement": utilization_improvement
        })
    
    # Save the results to a file
    with open("reports/context_utilization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    visualize_utilization_results(results)
    
    return results


def test_response_generation():
    """Test response generation with the optimized context."""
    logger.info("Testing response generation with optimized context")
    
    # Create a unique ID for this test session
    session_id = f"test_response_{uuid.uuid4().hex[:8]}"
    
    # Initialize memory manager with the session ID
    memory_manager = get_memory_manager(session_id)
    
    # Initialize LLM
    llm = get_llm()
    
    # Test queries
    test_queries = [
        "How do I implement a binary search tree in Python?",
        "What's the best way to handle errors in JavaScript?",
        "Can you explain how asynchronous programming works?"
    ]
    
    results = []
    
    for query in test_queries:
        logger.info(f"Testing query: {query}")
        
        # Get optimized context
        context = memory_manager.refresh_context(query)
        
        # Generate response with optimized context
        start_time = time.time()
        response = llm.generate_with_context(
            query=query,
            context=context
        )
        generation_time = time.time() - start_time
        
        # Log response metrics
        metrics = context.get("metrics", {})
        response_length = len(response)
        
        logger.info(f"  Response generation time: {generation_time:.2f} seconds")
        logger.info(f"  Response length: {response_length} characters")
        logger.info(f"  Context utilization: {metrics.get('utilization', 0):.1%}")
        logger.info(f"  Query classified as: {metrics.get('query_type', 'unknown')}, {metrics.get('complexity', 'unknown')}")
        logger.info(f"  Response excerpt: {response[:100]}...")
        logger.info("")
        
        # Store results
        results.append({
            "query": query,
            "metrics": metrics,
            "generation_time": generation_time,
            "response_length": response_length,
            "response_excerpt": response[:200]
        })
    
    # Save the results to a file
    with open("reports/response_generation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def visualize_utilization_results(results: List[Dict[str, Any]]):
    """Create visualizations of the context utilization improvements."""
    # Create a figure for utilization comparison
    plt.figure(figsize=(12, 6))
    
    test_cases = [r["test_case"] for r in results]
    original_util = [r["original"]["utilization"] for r in results]
    optimized_util = [r["optimized"]["utilization"] for r in results]
    
    x = np.arange(len(test_cases))
    width = 0.35
    
    # Create the bar chart
    bars1 = plt.bar(x - width/2, original_util, width, label='Original', color='#3498db')
    bars2 = plt.bar(x + width/2, optimized_util, width, label='Optimized', color='#2ecc71')
    
    plt.xlabel('Query Type')
    plt.ylabel('Context Window Utilization')
    plt.title('Context Window Utilization Comparison')
    plt.xticks(x, test_cases)
    plt.legend()
    
    # Add value labels to the bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    plt.savefig("reports/context_utilization_comparison.png")
    
    # Create a second figure for improvement percentages
    plt.figure(figsize=(10, 5))
    
    improvements = [r["improvement"] for r in results]
    
    bars = plt.bar(test_cases, improvements, color='#9b59b6')
    
    plt.xlabel('Query Type')
    plt.ylabel('Utilization Improvement (%)')
    plt.title('Context Window Utilization Improvement')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("reports/context_utilization_improvement.png")
    
    logger.info("Visualizations saved to 'reports' directory")


def main():
    """Run the memory optimization tests."""
    # Set up logger
    setup_logger()
    logger.info("Starting memory optimization tests")
    
    # Ensure reports directory exists
    os.makedirs("reports", exist_ok=True)
    
    # Run the tests
    test_query_classification()
    utilization_results = test_context_utilization()
    test_response_generation()
    
    # Print summary
    print("\n=== Memory Optimization Test Summary ===")
    print("\nContext utilization improvements:")
    
    total_improvement = 0
    for result in utilization_results:
        improvement = result["improvement"]
        total_improvement += improvement
        print(f"  {result['test_case']}: {result['original']['utilization']:.1%} → {result['optimized']['utilization']:.1%} ({improvement:+.1f}%)")
    
    avg_improvement = total_improvement / len(utilization_results)
    print(f"\nAverage improvement: {avg_improvement:.1f}%")
    print(f"Detailed results saved to 'reports' directory")


if __name__ == "__main__":
    main() 