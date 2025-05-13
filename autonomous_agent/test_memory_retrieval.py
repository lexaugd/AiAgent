#!/usr/bin/env python3
"""
Test script for evaluating memory retrieval relevance with different query types.

This script tests how the memory system responds to different types of queries,
and evaluates the relevance of retrieved information to help identify potential
causes of hallucination.
"""

import os
import sys
import time
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

from memory.manager import MemoryManager, get_memory_manager
from memory.long_term import LongTermMemory, MemoryItem
from memory.types import MemoryType, MemoryPriority
from memory.embeddings import EmbeddingGenerator
from config import MEMORY_CONFIG

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")


def setup_test_memory(agent_id: str = "retrieval_test") -> MemoryManager:
    """
    Set up a test memory system with sample data.
    
    Args:
        agent_id (str): Identifier for the test memory system
        
    Returns:
        MemoryManager: Configured memory manager with test data
    """
    memory_manager = get_memory_manager(agent_id)
    
    # Clear any existing test data
    memory_manager.clear_short_term()
    memory_manager.clear_working_memory()
    
    # Add test conversation
    memory_manager.add_message("user", "I'm working on a project that needs to process large amounts of data efficiently.")
    memory_manager.add_message("assistant", "There are several approaches for efficient data processing. Would you like to learn about streaming, batch processing, or parallel processing?")
    memory_manager.add_message("user", "I'm particularly interested in parallel processing with Python.")
    memory_manager.add_message("assistant", "Python offers several libraries for parallel processing, including multiprocessing, concurrent.futures, and asyncio. For data processing specifically, libraries like Dask and Ray are quite powerful.")
    
    # Add code examples to long-term memory
    sample_codes = [
        {
            "content": """
def process_data_parallel(data_list, process_func, n_workers=4):
    \"\"\"Process data in parallel using multiprocessing.\"\"\"
    from multiprocessing import Pool
    
    with Pool(processes=n_workers) as pool:
        results = pool.map(process_func, data_list)
    return results
""",
            "language": "python",
            "metadata": {
                "description": "Basic multiprocessing example with Pool",
                "priority": MemoryPriority.HIGH
            }
        },
        {
            "content": """
async def process_data_async(data_list, process_func):
    \"\"\"Process data asynchronously using asyncio.\"\"\"
    import asyncio
    
    tasks = [process_func(item) for item in data_list]
    results = await asyncio.gather(*tasks)
    return results
""",
            "language": "python",
            "metadata": {
                "description": "Asyncio-based data processing",
                "priority": MemoryPriority.MEDIUM
            }
        },
        {
            "content": """
def process_large_dataset(file_path, chunk_size=1000):
    \"\"\"Process a large dataset in chunks to save memory.\"\"\"
    import pandas as pd
    
    # Process file in chunks
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    results = []
    
    for chunk in chunks:
        # Process each chunk
        processed = chunk.apply(lambda x: x * 2)
        results.append(processed)
    
    # Combine results
    return pd.concat(results)
""",
            "language": "python",
            "metadata": {
                "description": "Memory-efficient data processing with pandas",
                "priority": MemoryPriority.MEDIUM
            }
        },
        {
            "content": """
import ray
ray.init()

@ray.remote
def process_chunk(chunk):
    # Process the chunk
    return chunk * 2

def process_with_ray(data):
    # Split data into chunks
    chunks = np.array_split(data, 10)
    
    # Process in parallel
    futures = [process_chunk.remote(chunk) for chunk in chunks]
    results = ray.get(futures)
    
    # Combine results
    return np.concatenate(results)
""",
            "language": "python",
            "metadata": {
                "description": "Distributed computing with Ray",
                "priority": MemoryPriority.HIGH
            }
        },
        {
            "content": """
from concurrent.futures import ThreadPoolExecutor

def process_with_threads(data_list, process_func, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_func, data_list))
    return results
""",
            "language": "python",
            "metadata": {
                "description": "Multithreading with ThreadPoolExecutor",
                "priority": MemoryPriority.MEDIUM
            }
        },
        {
            "content": """
class DataProcessor:
    def __init__(self, n_workers=4):
        self.n_workers = n_workers
    
    def process_batch(self, batch):
        # Process a single batch
        return [item * 2 for item in batch]
    
    def process_all(self, data):
        # Split into batches
        batch_size = len(data) // self.n_workers
        batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        
        # Process each batch
        from multiprocessing import Pool
        with Pool(processes=self.n_workers) as pool:
            results = pool.map(self.process_batch, batches)
        
        # Flatten results
        return [item for sublist in results for item in sublist]
""",
            "language": "python",
            "metadata": {
                "description": "OOP approach to batch processing",
                "priority": MemoryPriority.LOW
            }
        }
    ]
    
    # Add each code example to long-term memory
    for code in sample_codes:
        memory_manager.add_code_to_long_term(
            code["content"],
            code["language"],
            metadata=code["metadata"]
        )
    
    # Add some conceptual knowledge
    concepts = [
        {
            "content": "Parallel processing divides a task into subtasks that are processed simultaneously by multiple processors. This approach can significantly reduce processing time for suitable tasks.",
            "type": "concept",
            "metadata": {
                "description": "Parallel processing definition",
                "priority": MemoryPriority.MEDIUM
            }
        },
        {
            "content": "Python's Global Interpreter Lock (GIL) prevents multiple native threads from executing Python bytecodes at once. This means that threads cannot execute Python code in parallel, though they can run concurrently during I/O operations.",
            "type": "concept",
            "metadata": {
                "description": "Python GIL limitation",
                "priority": MemoryPriority.HIGH
            }
        },
        {
            "content": "For CPU-bound tasks in Python, the multiprocessing module is often more effective than threading due to the GIL. For I/O-bound tasks, threading or asyncio may be more appropriate.",
            "type": "concept",
            "metadata": {
                "description": "Choosing between multiprocessing and threading",
                "priority": MemoryPriority.HIGH
            }
        }
    ]
    
    # Add each concept to long-term memory
    for concept in concepts:
        memory_manager.add_to_long_term(
            concept["content"],
            concept["type"],
            metadata=concept["metadata"]
        )
    
    logger.info(f"Test memory system initialized with {len(sample_codes)} code examples and {len(concepts)} concepts")
    return memory_manager


def test_query_types(memory_manager: MemoryManager) -> Dict[str, Any]:
    """
    Test different types of queries and evaluate retrieval relevance.
    
    Args:
        memory_manager (MemoryManager): Memory manager to test
        
    Returns:
        Dict[str, Any]: Test results
    """
    # Define query types to test
    query_types = {
        "specific_technical": [
            "How do I use multiprocessing in Python?",
            "Can you show me how to process data with asyncio?",
            "What's the best way to handle large datasets in Python?"
        ],
        "ambiguous": [
            "How do I process data efficiently?",
            "What's the best approach for my project?",
            "How can I speed up my code?"
        ],
        "novel_concepts": [
            "How do I implement quantum computing algorithms in Python?",
            "Can you help me with blockchain data processing?",
            "How to use neural networks for data transformation?"
        ],
        "context_dependent": [
            "Which approach would work better for my use case?",
            "Can you explain more about that library you mentioned?",
            "What are the alternatives to this method?"
        ]
    }
    
    results = {}
    
    # Test each query type
    for query_type, queries in query_types.items():
        logger.info(f"Testing {query_type} queries")
        query_results = []
        
        for query in queries:
            logger.info(f"Query: {query}")
            
            # Get relevant items
            start_time = time.time()
            relevant_items = memory_manager.retrieve_relevant(query, n_results=3)
            retrieval_time = time.time() - start_time
            
            # Calculate a simple relevance score based on keyword matching
            # (In a real implementation, this would use more sophisticated methods)
            relevance_scores = []
            for item in relevant_items:
                query_words = set(query.lower().split())
                content_words = set(item.content.lower().split())
                common_words = query_words.intersection(content_words)
                score = len(common_words) / len(query_words) if query_words else 0
                relevance_scores.append(score)
            
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
            
            # Log the first result
            if relevant_items:
                logger.info(f"Top result: {relevant_items[0].content[:100]}...")
                logger.info(f"Relevance score: {relevance_scores[0]:.2f}")
            else:
                logger.info("No results found")
            
            # Store the results
            query_results.append({
                "query": query,
                "retrieval_time": retrieval_time,
                "result_count": len(relevant_items),
                "relevance_scores": relevance_scores,
                "avg_relevance": avg_relevance
            })
        
        # Calculate average metrics for this query type
        avg_retrieval_time = sum(r["retrieval_time"] for r in query_results) / len(query_results)
        avg_result_count = sum(r["result_count"] for r in query_results) / len(query_results)
        avg_relevance = sum(r["avg_relevance"] for r in query_results) / len(query_results)
        
        results[query_type] = {
            "queries": query_results,
            "avg_retrieval_time": avg_retrieval_time,
            "avg_result_count": avg_result_count,
            "avg_relevance": avg_relevance
        }
        
        logger.info(f"Average relevance for {query_type} queries: {avg_relevance:.2f}")
    
    return results


def visualize_results(results: Dict[str, Any], output_dir: str = "reports") -> None:
    """
    Visualize the retrieval test results.
    
    Args:
        results (Dict[str, Any]): Test results
        output_dir (str): Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    query_types = list(results.keys())
    avg_relevance = [results[qt]["avg_relevance"] for qt in query_types]
    avg_retrieval_time = [results[qt]["avg_retrieval_time"] for qt in query_types]
    
    # Plot average relevance by query type
    plt.figure(figsize=(10, 6))
    bars = plt.bar(query_types, avg_relevance, color='skyblue')
    plt.xlabel('Query Type')
    plt.ylabel('Average Relevance Score')
    plt.title('Retrieval Relevance by Query Type')
    plt.ylim(0, 1.0)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'retrieval_relevance.png'))
    plt.close()
    
    # Plot retrieval time by query type
    plt.figure(figsize=(10, 6))
    bars = plt.bar(query_types, avg_retrieval_time, color='lightgreen')
    plt.xlabel('Query Type')
    plt.ylabel('Average Retrieval Time (seconds)')
    plt.title('Retrieval Time by Query Type')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'retrieval_time.png'))
    plt.close()
    
    # Save the raw data
    with open(os.path.join(output_dir, 'retrieval_test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Main function to run the memory retrieval tests."""
    
    # Set up test memory
    memory_manager = setup_test_memory()
    
    # Run the tests
    logger.info("Starting retrieval relevance tests")
    results = test_query_types(memory_manager)
    
    # Visualize results
    visualize_results(results)
    
    # Print summary
    print("\n=== Memory Retrieval Test Summary ===")
    for query_type, data in results.items():
        print(f"{query_type}: {data['avg_relevance']:.2f} relevance score, {data['avg_retrieval_time']:.3f}s avg retrieval time")
    
    print(f"\nDetailed results saved to reports/retrieval_test_results.json")


if __name__ == "__main__":
    main() 