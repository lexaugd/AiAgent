#!/usr/bin/env python3
"""
Test script for evaluating the effects of different temperature and sampling parameters
on model output quality and hallucination rates.
"""

import os
import sys
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

from models.llm_interface import LocalLLM, get_llm
from config import MODEL_CONFIG

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")


def evaluate_response_quality(prompt: str, response: str) -> Dict[str, float]:
    """
    Evaluate response quality based on simple text analysis heuristics.
    
    Args:
        prompt (str): The prompt that generated the response
        response (str): The response to evaluate
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    # Simple length-based metrics
    char_count = len(response)
    word_count = len(response.split())
    
    # Check for repetition (a basic hallucination signal)
    words = response.lower().split()
    unique_words = len(set(words))
    repetition_ratio = unique_words / word_count if word_count > 0 else 0
    
    # Check for uncertainty markers (often indicates hallucination)
    uncertainty_markers = [
        "i think", "i believe", "probably", "might be", "could be", 
        "possibly", "perhaps", "may be", "not sure", "uncertain"
    ]
    uncertainty_count = sum(1 for marker in uncertainty_markers if marker in response.lower())
    uncertainty_score = uncertainty_count / (word_count / 50) if word_count > 0 else 0
    
    # Check for prompt keywords in response (relevance)
    prompt_words = set(prompt.lower().split())
    prompt_words = {w for w in prompt_words if len(w) > 3}  # Filter short words
    response_words = set(response.lower().split())
    keyword_overlap = len(prompt_words.intersection(response_words)) / len(prompt_words) if prompt_words else 0
    
    # Check for code blocks (usually good sign in coding assistant)
    code_block_count = response.count("```")
    has_code = code_block_count >= 2
    
    # Return all metrics
    return {
        "char_count": char_count,
        "word_count": word_count,
        "repetition_ratio": repetition_ratio,
        "uncertainty_score": uncertainty_score,
        "keyword_overlap": keyword_overlap,
        "has_code": has_code
    }


def evaluate_temperature_effects(
    llm: LocalLLM,
    prompts: List[Dict[str, Any]],
    temperatures: List[float],
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Test how different temperature settings affect model outputs.
    
    Args:
        llm (LocalLLM): The LLM instance to use
        prompts (List[Dict[str, Any]]): List of test prompts and metadata
        temperatures (List[float]): List of temperatures to test
        output_dir (str): Directory to save results
        
    Returns:
        Dict[str, Any]: Test results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for temp in temperatures:
        logger.info(f"Testing temperature {temp}")
        temp_results = []
        
        for prompt_data in prompts:
            prompt = prompt_data["prompt"]
            category = prompt_data["category"]
            
            logger.info(f"  Prompt category: {category}")
            
            # Generate response with this temperature
            start_time = time.time()
            response = llm.generate(
                prompt=prompt,
                temperature=temp,
                max_tokens=500
            )
            generation_time = time.time() - start_time
            
            # Evaluate response quality
            quality_metrics = evaluate_response_quality(prompt, response)
            
            # Store results
            temp_results.append({
                "prompt": prompt,
                "category": category,
                "temperature": temp,
                "response": response,
                "generation_time": generation_time,
                "metrics": quality_metrics
            })
            
            logger.info(f"  Response length: {quality_metrics['word_count']} words")
            logger.info(f"  Repetition ratio: {quality_metrics['repetition_ratio']:.2f}")
            logger.info(f"  Keyword overlap: {quality_metrics['keyword_overlap']:.2f}")
            
            # Small pause to avoid rate limiting
            time.sleep(1)
        
        # Store all results for this temperature
        results[temp] = temp_results
    
    # Save complete results
    with open(os.path.join(output_dir, "temperature_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def evaluate_sampling_params(
    llm: LocalLLM,
    prompt: str,
    param_combinations: List[Dict[str, Any]],
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Test how different sampling parameter combinations affect model outputs.
    
    Args:
        llm (LocalLLM): The LLM instance to use
        prompt (str): Test prompt to use for all tests
        param_combinations (List[Dict[str, Any]]): Parameter combinations to test
        output_dir (str): Directory to save results
        
    Returns:
        Dict[str, Any]: Test results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for params in param_combinations:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        logger.info(f"Testing parameters: {param_str}")
        
        # Generate response with these parameters
        start_time = time.time()
        response = llm.generate(
            prompt=prompt,
            max_tokens=500,
            **params
        )
        generation_time = time.time() - start_time
        
        # Evaluate response quality
        quality_metrics = evaluate_response_quality(prompt, response)
        
        # Store results
        results.append({
            "parameters": params,
            "response": response,
            "generation_time": generation_time,
            "metrics": quality_metrics
        })
        
        logger.info(f"  Response length: {quality_metrics['word_count']} words")
        logger.info(f"  Repetition ratio: {quality_metrics['repetition_ratio']:.2f}")
        logger.info(f"  Keyword overlap: {quality_metrics['keyword_overlap']:.2f}")
        
        # Small pause to avoid rate limiting
        time.sleep(1)
    
    # Save results
    with open(os.path.join(output_dir, "sampling_params_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def visualize_temperature_results(results: Dict[str, List[Dict[str, Any]]], output_dir: str = "reports") -> None:
    """
    Visualize the effects of temperature on model outputs.
    
    Args:
        results (Dict[str, List[Dict[str, Any]]]): Temperature test results
        output_dir (str): Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for visualization
    temperatures = sorted(results.keys())
    
    # Aggregate metrics across all prompts for each temperature
    metrics = {
        "word_count": [],
        "repetition_ratio": [],
        "uncertainty_score": [],
        "keyword_overlap": []
    }
    
    for temp in temperatures:
        # Average metrics across all prompts for this temperature
        temp_metrics = {
            "word_count": [],
            "repetition_ratio": [],
            "uncertainty_score": [],
            "keyword_overlap": []
        }
        
        for result in results[temp]:
            for metric in temp_metrics:
                temp_metrics[metric].append(result["metrics"][metric])
        
        # Calculate averages
        for metric in metrics:
            metrics[metric].append(np.mean(temp_metrics[metric]))
    
    # Plot word count vs temperature
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, metrics["word_count"], marker='o', linewidth=2)
    plt.xlabel('Temperature')
    plt.ylabel('Average Word Count')
    plt.title('Response Length vs. Temperature')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'word_count_vs_temperature.png'))
    plt.close()
    
    # Plot repetition ratio vs temperature
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, metrics["repetition_ratio"], marker='o', linewidth=2)
    plt.xlabel('Temperature')
    plt.ylabel('Average Repetition Ratio (higher is better)')
    plt.title('Repetition Ratio vs. Temperature')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'repetition_vs_temperature.png'))
    plt.close()
    
    # Plot uncertainty score vs temperature
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, metrics["uncertainty_score"], marker='o', linewidth=2)
    plt.xlabel('Temperature')
    plt.ylabel('Average Uncertainty Score (lower is better)')
    plt.title('Uncertainty Score vs. Temperature')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_vs_temperature.png'))
    plt.close()
    
    # Plot keyword overlap vs temperature
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, metrics["keyword_overlap"], marker='o', linewidth=2)
    plt.xlabel('Temperature')
    plt.ylabel('Average Keyword Overlap (higher is better)')
    plt.title('Keyword Overlap vs. Temperature')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overlap_vs_temperature.png'))
    plt.close()
    
    logger.info(f"Temperature visualization saved to {output_dir}")


def main():
    """Main function to run the temperature and sampling parameter tests."""
    
    # Initialize LLM
    llm = get_llm()
    model_name = llm.model_name
    logger.info(f"Using model: {model_name}")
    
    # Define test prompts with different hallucination risks
    test_prompts = [
        {
            "category": "specific_technical",
            "prompt": "Write a function to merge two sorted arrays in Python."
        },
        {
            "category": "ambiguous",
            "prompt": "Make my code more efficient."
        },
        {
            "category": "novel_concept",
            "prompt": "Explain how to use the QuantumState library in Python for quantum computing."
        },
        {
            "category": "context_dependent",
            "prompt": "Can you elaborate on that approach more?"
        },
        {
            "category": "specific_with_constraint",
            "prompt": "Write a function to calculate Fibonacci numbers without using recursion or memoization."
        }
    ]
    
    # Define temperatures to test
    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Run temperature tests
    logger.info("Starting temperature effect tests")
    temp_results = evaluate_temperature_effects(llm, test_prompts, temperatures)
    
    # Visualize temperature results
    visualize_temperature_results(temp_results)
    
    # Define sampling parameter combinations to test
    param_combinations = [
        {"temperature": 0.7, "top_p": 0.95, "frequency_penalty": 0.0, "presence_penalty": 0.0},
        {"temperature": 0.7, "top_p": 0.8, "frequency_penalty": 0.0, "presence_penalty": 0.0},
        {"temperature": 0.7, "top_p": 0.95, "frequency_penalty": 0.5, "presence_penalty": 0.0},
        {"temperature": 0.7, "top_p": 0.95, "frequency_penalty": 0.0, "presence_penalty": 0.5},
        {"temperature": 0.7, "top_p": 0.95, "frequency_penalty": 0.5, "presence_penalty": 0.5},
        {"temperature": 0.4, "top_p": 0.9, "frequency_penalty": 0.3, "presence_penalty": 0.3}
    ]
    
    # Run sampling parameter tests
    logger.info("Starting sampling parameter tests")
    common_prompt = "Explain the difference between multiprocessing and multithreading in Python."
    sampling_results = evaluate_sampling_params(llm, common_prompt, param_combinations)
    
    # Print summary
    print("\n=== Temperature and Sampling Parameter Test Summary ===")
    
    # Print temperature test summary
    print("\nTemperature effects:")
    for temp in sorted(temp_results.keys()):
        avg_word_count = np.mean([r["metrics"]["word_count"] for r in temp_results[temp]])
        avg_overlap = np.mean([r["metrics"]["keyword_overlap"] for r in temp_results[temp]])
        avg_uncertainty = np.mean([r["metrics"]["uncertainty_score"] for r in temp_results[temp]])
        
        print(f"  Temperature {temp}: {avg_word_count:.1f} words, {avg_overlap:.2f} overlap, {avg_uncertainty:.2f} uncertainty")
    
    # Print best temperature finding
    metrics_by_temp = {}
    for temp in sorted(temp_results.keys()):
        metrics_by_temp[temp] = {
            "word_count": np.mean([r["metrics"]["word_count"] for r in temp_results[temp]]),
            "repetition_ratio": np.mean([r["metrics"]["repetition_ratio"] for r in temp_results[temp]]),
            "uncertainty_score": np.mean([r["metrics"]["uncertainty_score"] for r in temp_results[temp]]),
            "keyword_overlap": np.mean([r["metrics"]["keyword_overlap"] for r in temp_results[temp]])
        }
    
    # Find best temperature as a balanced score
    scores = {}
    for temp, metrics in metrics_by_temp.items():
        # Higher word count (up to a point) is good
        word_score = min(metrics["word_count"] / 300, 1.0)
        # Higher repetition ratio is good
        repetition_score = metrics["repetition_ratio"]
        # Lower uncertainty is good
        uncertainty_score = 1.0 - min(metrics["uncertainty_score"], 1.0)
        # Higher keyword overlap is good
        overlap_score = metrics["keyword_overlap"]
        
        # Calculate weighted score
        scores[temp] = (word_score * 0.2 + repetition_score * 0.3 + 
                       uncertainty_score * 0.2 + overlap_score * 0.3)
    
    best_temp = max(scores, key=scores.get)
    print(f"\nBest temperature: {best_temp} (score: {scores[best_temp]:.2f})")
    
    # Print sampling parameter test summary
    print("\nSampling parameter effects:")
    for i, result in enumerate(sampling_results):
        params = ", ".join(f"{k}={v}" for k, v in result["parameters"].items())
        print(f"  Combination {i+1} ({params}):")
        print(f"    Word count: {result['metrics']['word_count']}")
        print(f"    Repetition ratio: {result['metrics']['repetition_ratio']:.2f}")
        print(f"    Uncertainty score: {result['metrics']['uncertainty_score']:.2f}")
        print(f"    Keyword overlap: {result['metrics']['keyword_overlap']:.2f}")
    
    print(f"\nDetailed results saved to reports/temperature_test_results.json and reports/sampling_params_test_results.json")


if __name__ == "__main__":
    main() 