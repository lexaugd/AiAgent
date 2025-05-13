#!/usr/bin/env python3
"""
Script for analyzing model interaction logs to identify hallucination patterns.
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from config import LOGGING_CONFIG


def setup_logger():
    """Set up the script logger."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    return logger


def load_model_interactions(log_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load all model interaction logs from the specified directory.
    
    Args:
        log_dir (str, optional): Directory containing model interaction logs
        
    Returns:
        List[Dict[str, Any]]: List of model interaction data
    """
    # Default log directory if not specified
    if log_dir is None:
        log_dir = Path(LOGGING_CONFIG.get("file", "logs")).parent / "model_interactions"
    
    # Ensure directory exists
    if not os.path.exists(log_dir):
        logger.error(f"Log directory does not exist: {log_dir}")
        return []
    
    # Load all JSON files in the directory
    interactions = []
    for log_file in glob.glob(os.path.join(log_dir, "*.json")):
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
                interactions.append(data)
        except Exception as e:
            logger.error(f"Error loading log file {log_file}: {e}")
    
    return interactions


def classify_hallucinations(interactions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Classify different types of hallucinations based on keywords and patterns.
    
    Args:
        interactions (List[Dict[str, Any]]): List of model interaction data
        
    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary mapping hallucination types to interactions
    """
    # Define hallucination classification patterns
    hallucination_patterns = {
        "fictional_libraries": [
            "import", "from", "library", "module", "package",
            "quantumstate", "neural_synthesis", "brain_sim", "quantum"
        ],
        "api_endpoints": [
            "api", "endpoint", "/api", "POST", "GET", "request", "response",
            "neural_synthesis", "brain"
        ],
        "nonexistent_patterns": [
            "pattern", "design pattern", "architecture", "framework",
            "Observer-Commander", "Commander-Adapter"
        ],
        "ambiguous_solutions": [
            "efficient", "all the things", "everything", "any", "general purpose"
        ],
        "imprecise_answers": [
            "that popular", "the latest", "recently", "commonly", "usually"
        ]
    }
    
    # Create a dictionary to hold classified interactions
    classified = defaultdict(list)
    
    # Classify each interaction
    for interaction in interactions:
        prompt = interaction["prompt"].lower()
        response = interaction["response"].lower()
        
        # Check each hallucination pattern
        matched = False
        for hallucination_type, keywords in hallucination_patterns.items():
            # Check if keywords appear in the prompt
            prompt_matched = any(keyword.lower() in prompt for keyword in keywords)
            
            # Only classify if prompt is relevant to this hallucination type
            if prompt_matched:
                # Look for fictional things in response
                for keyword in keywords:
                    if keyword.lower() in response:
                        classified[hallucination_type].append(interaction)
                        matched = True
                        break
            
            if matched:
                break
        
        # If not matched to any specific type, check for general hallucination indicators
        if not matched:
            # Check for uncertain language suggesting hallucination
            uncertainty_markers = ["i believe", "i think", "might be", "could be", "probably"]
            if any(marker in response for marker in uncertainty_markers):
                classified["uncertain_responses"].append(interaction)
                
            # Check for completely off-topic responses
            topic_drift = False
            prompt_words = set(prompt.split())
            response_words = set(response.split())
            common_words = prompt_words.intersection(response_words)
            if len(common_words) < 3 and len(prompt_words) > 5:
                classified["topic_drift"].append(interaction)
    
    return classified


def generate_hallucination_report(classified: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Generate a summary report of hallucination patterns.
    
    Args:
        classified (Dict[str, List[Dict[str, Any]]]): Classified hallucinations
        
    Returns:
        Dict[str, Any]: Report data
    """
    total_hallucinations = sum(len(items) for items in classified.values())
    
    # Create a summary report
    report = {
        "total_hallucinations": total_hallucinations,
        "hallucination_types": {k: len(v) for k, v in classified.items()},
        "hallucination_samples": {},
        "common_patterns": {}
    }
    
    # Add examples for each hallucination type
    for h_type, interactions in classified.items():
        if interactions:
            # Take first example for each type
            report["hallucination_samples"][h_type] = {
                "prompt": interactions[0]["prompt"],
                "response_excerpt": interactions[0]["response"][:100] + "..." if len(interactions[0]["response"]) > 100 else interactions[0]["response"]
            }
            
            # Extract common patterns in this hallucination type
            common_phrases = defaultdict(int)
            for interaction in interactions:
                response = interaction["response"].lower()
                for phrase in ["import", "library", "module", "api", "endpoint", 
                              "function", "class", "pattern", "framework", "version"]:
                    if phrase in response:
                        common_phrases[phrase] += 1
            
            report["common_patterns"][h_type] = dict(common_phrases)
    
    return report


def visualize_hallucinations(classified: Dict[str, List[Dict[str, Any]]], output_dir: str = "reports"):
    """
    Create visualizations of hallucination patterns.
    
    Args:
        classified (Dict[str, List[Dict[str, Any]]]): Classified hallucinations
        output_dir (str): Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame for visualization
    data = [(h_type, len(interactions)) for h_type, interactions in classified.items()]
    df = pd.DataFrame(data, columns=["Hallucination Type", "Count"])
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["Hallucination Type"], df["Count"])
    plt.xlabel("Hallucination Type")
    plt.ylabel("Count")
    plt.title("Hallucination Type Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f"{height}", 
                 ha="center", va="bottom")
    
    # Save the chart
    plt.savefig(os.path.join(output_dir, "hallucination_types.png"))
    plt.close()
    
    # Log the completion
    logger.info(f"Saved hallucination visualization to {output_dir}/hallucination_types.png")


def main():
    """Main function to analyze hallucinations."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Analyze model interaction logs for hallucination patterns")
    parser.add_argument("--log-dir", help="Directory containing model interaction logs")
    parser.add_argument("--output-dir", default="reports", help="Directory to save reports and visualizations")
    args = parser.parse_args()
    
    # Set up logger
    setup_logger()
    logger.info("Starting hallucination analysis")
    
    # Load model interactions
    interactions = load_model_interactions(args.log_dir)
    logger.info(f"Loaded {len(interactions)} model interactions")
    
    if not interactions:
        logger.error("No model interactions found. Make sure the log directory exists and contains JSON files.")
        return
    
    # Classify hallucinations
    classified = classify_hallucinations(interactions)
    total_hallucinations = sum(len(items) for items in classified.values())
    logger.info(f"Classified {total_hallucinations} potential hallucinations")
    
    # Generate report
    report = generate_hallucination_report(classified)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save report to JSON file
    report_file = os.path.join(args.output_dir, "hallucination_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved hallucination report to {report_file}")
    
    # Create visualizations
    visualize_hallucinations(classified, args.output_dir)
    
    # Print summary to console
    print("\n=== Hallucination Analysis Summary ===")
    print(f"Total interactions analyzed: {len(interactions)}")
    print(f"Total potential hallucinations: {total_hallucinations}")
    print("\nHallucination types:")
    for h_type, count in report["hallucination_types"].items():
        print(f"  - {h_type}: {count}")
    
    print(f"\nDetailed report saved to {report_file}")


if __name__ == "__main__":
    main() 