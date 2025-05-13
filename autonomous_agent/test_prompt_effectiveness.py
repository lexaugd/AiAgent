#!/usr/bin/env python3
"""
Test script to evaluate the effectiveness of different system prompts 
in reducing hallucinations.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger

from models.llm_interface import get_llm
from models.prompt_templates import get_system_prompt, create_prompt_with_context

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")


def evaluate_prompt_on_hallucination_triggers(
    hallucination_triggers: List[Dict[str, str]],
    prompt_types: List[str],
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Evaluate different system prompts on hallucination-prone queries.
    
    Args:
        hallucination_triggers (List[Dict[str, str]]): List of queries known to trigger hallucinations
        prompt_types (List[str]): List of system prompt types to test
        output_dir (str): Directory to save results
        
    Returns:
        Dict[str, Any]: Test results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize LLM
    llm = get_llm()
    model_name = llm.model_name
    logger.info(f"Using model: {model_name}")
    
    results = {}
    
    # Test each prompt type
    for prompt_type in prompt_types:
        logger.info(f"Testing prompt type: {prompt_type}")
        prompt_results = []
        
        # Get the system prompt
        system_prompt = get_system_prompt(prompt_type)
        
        # Test each hallucination trigger
        for trigger in hallucination_triggers:
            query = trigger["query"]
            category = trigger["category"]
            
            logger.info(f"  Testing query: {query[:50]}...")
            
            # Generate response with this system prompt
            start_time = time.time()
            response = llm.generate(
                prompt=query,
                system_prompt=system_prompt,
                temperature=0.3,  # Using lower temperature based on our findings
                max_tokens=500
            )
            generation_time = time.time() - start_time
            
            # Simple hallucination detection heuristics
            hallucination_indicators = detect_hallucination_indicators(response, query)
            
            # Store results
            prompt_results.append({
                "query": query,
                "category": category,
                "response": response,
                "generation_time": generation_time,
                "hallucination_indicators": hallucination_indicators,
                "hallucination_score": calculate_hallucination_score(hallucination_indicators)
            })
            
            logger.info(f"  Hallucination score: {calculate_hallucination_score(hallucination_indicators):.2f}")
            if hallucination_indicators["uncertainty_acknowledged"]:
                logger.info(f"  ✓ Uncertainty acknowledged")
            
            # Small pause to avoid rate limiting
            time.sleep(1)
        
        # Store all results for this prompt type
        results[prompt_type] = prompt_results
    
    # Calculate aggregate metrics
    aggregate_results = {}
    for prompt_type, prompt_results in results.items():
        avg_hallucination_score = sum(r["hallucination_score"] for r in prompt_results) / len(prompt_results)
        uncertainty_acknowledged_count = sum(1 for r in prompt_results if r["hallucination_indicators"]["uncertainty_acknowledged"])
        fictional_references_count = sum(1 for r in prompt_results if r["hallucination_indicators"]["fictional_references"])
        
        aggregate_results[prompt_type] = {
            "avg_hallucination_score": avg_hallucination_score,
            "uncertainty_acknowledged_rate": uncertainty_acknowledged_count / len(prompt_results),
            "fictional_references_rate": fictional_references_count / len(prompt_results)
        }
    
    # Determine the best prompt type
    best_prompt = min(aggregate_results.items(), key=lambda x: x[1]["avg_hallucination_score"])[0]
    
    # Save complete results
    all_results = {
        "detailed_results": results,
        "aggregate_results": aggregate_results,
        "best_prompt": best_prompt
    }
    
    with open(os.path.join(output_dir, "prompt_effectiveness_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    return all_results


def detect_hallucination_indicators(response: str, query: str) -> Dict[str, Any]:
    """
    Detect indicators of hallucination in a response.
    
    Args:
        response (str): The model's response
        query (str): The query that generated the response
        
    Returns:
        Dict[str, Any]: Indicators of hallucination
    """
    # Convert to lowercase for case-insensitive matching
    response_lower = response.lower()
    
    # Check for uncertainty acknowledgment
    uncertainty_phrases = [
        "i'm not sure", "i am not sure", 
        "i don't know", "i do not know",
        "i'm uncertain", "i am uncertain", 
        "i'm not certain", "i am not certain",
        "i don't have information", "i do not have information",
        "i'm not familiar", "i am not familiar"
    ]
    uncertainty_acknowledged = any(phrase in response_lower for phrase in uncertainty_phrases)
    
    # Check for fictional libraries/APIs
    fictional_libraries = [
        "quantumstate", "brainjs", "neuralflow", "quantumnet", 
        "hyperml", "cosmicnn", "mindlib", "cerebrojs", "quantumpy"
    ]
    fictional_references = any(lib in response_lower for lib in fictional_libraries)
    
    # Check for hallucination-prone phrases
    hallucination_phrases = [
        "as we all know", "as you know", "common knowledge",
        "widely used", "standard practice", "industry standard",
        "everyone uses", "official documentation states",
        "conventional wisdom", "best practice"
    ]
    uses_hallucination_phrases = any(phrase in response_lower for phrase in hallucination_phrases)
    
    # Check for excessive hedging
    hedging_phrases = [
        "probably", "likely", "possibly", "might be", "could be",
        "perhaps", "maybe", "i believe", "i think", "in my opinion"
    ]
    hedging_count = sum(response_lower.count(phrase) for phrase in hedging_phrases)
    excessive_hedging = hedging_count >= 3
    
    # Return all indicators
    return {
        "uncertainty_acknowledged": uncertainty_acknowledged,
        "fictional_references": fictional_references,
        "uses_hallucination_phrases": uses_hallucination_phrases,
        "excessive_hedging": excessive_hedging,
        "hedging_count": hedging_count
    }


def calculate_hallucination_score(indicators: Dict[str, Any]) -> float:
    """
    Calculate a hallucination score based on indicators.
    Lower is better (less hallucination).
    
    Args:
        indicators (Dict[str, Any]): Hallucination indicators
        
    Returns:
        float: Hallucination score (0.0 to 1.0)
    """
    score = 0.0
    
    # Uncertainty acknowledgment reduces score (good thing)
    if indicators["uncertainty_acknowledged"]:
        score -= 0.3
    
    # Fictional references increases score (bad thing)
    if indicators["fictional_references"]:
        score += 0.5
    
    # Using hallucination-prone phrases increases score
    if indicators["uses_hallucination_phrases"]:
        score += 0.2
    
    # Excessive hedging increases score
    if indicators["excessive_hedging"]:
        score += 0.2
    else:
        # Some hedging is normal
        score += min(indicators["hedging_count"] * 0.05, 0.2)
    
    # Normalize to 0.0-1.0 range
    return max(0.0, min(1.0, score + 0.3))  # Baseline of 0.3


def test_context_enrichment(output_dir: str = "reports") -> Dict[str, Any]:
    """
    Test how providing additional context affects hallucination rates.
    
    Args:
        output_dir (str): Directory to save results
        
    Returns:
        Dict[str, Any]: Test results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize LLM
    llm = get_llm()
    model_name = llm.model_name
    logger.info(f"Using model: {model_name}")
    
    # Define a hallucination-prone query
    query = "Explain how to use the QuantumState library for quantum computing in Python."
    
    # Define different context levels
    context_levels = [
        {
            "name": "no_context",
            "context": None
        },
        {
            "name": "minimal_context",
            "context": {
                "docs": [
                    "Note: There is no widely-used Python library called 'QuantumState' for quantum computing."
                ]
            }
        },
        {
            "name": "alternative_context",
            "context": {
                "docs": [
                    "Instead of 'QuantumState', consider using established quantum computing libraries like Qiskit, Cirq, or PennyLane."
                ],
                "examples": [
                    "# Example with Qiskit\nimport qiskit\nfrom qiskit import QuantumCircuit\n\n# Create a quantum circuit with 2 qubits\nqc = QuantumCircuit(2)"
                ]
            }
        },
        {
            "name": "detailed_context",
            "context": {
                "docs": [
                    "Note: There is no widely-used Python library called 'QuantumState' for quantum computing.",
                    "Established quantum computing libraries include:\n- Qiskit (by IBM)\n- Cirq (by Google)\n- PennyLane (by Xanadu)\n- QuTiP (Quantum Toolbox in Python)"
                ],
                "examples": [
                    "# Example with Qiskit\nimport qiskit\nfrom qiskit import QuantumCircuit\n\n# Create a quantum circuit with 2 qubits\nqc = QuantumCircuit(2)",
                    "# Example with Cirq\nimport cirq\n\n# Create qubits\nq0, q1 = cirq.LineQubit.range(2)\n\n# Create a circuit\ncircuit = cirq.Circuit()\ncircuit.append(cirq.H(q0))"
                ],
                "conversation": [
                    {"role": "user", "content": "I'm interested in quantum computing with Python"},
                    {"role": "assistant", "content": "That's great! There are several established libraries for quantum computing in Python, including Qiskit, Cirq, and PennyLane."}
                ]
            }
        }
    ]
    
    results = []
    system_prompt = get_system_prompt("refined_coding")
    
    # Test each context level
    for context_level in context_levels:
        name = context_level["name"]
        context = context_level["context"]
        
        logger.info(f"Testing context level: {name}")
        
        # Create the complete prompt
        if context:
            prompt = create_prompt_with_context(query, system_prompt, context)
        else:
            prompt = create_prompt_with_context(query, system_prompt)
        
        # Generate response
        start_time = time.time()
        response = llm.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=500
        )
        generation_time = time.time() - start_time
        
        # Detect hallucination indicators
        hallucination_indicators = detect_hallucination_indicators(response, query)
        hallucination_score = calculate_hallucination_score(hallucination_indicators)
        
        # Store results
        results.append({
            "context_level": name,
            "response": response,
            "generation_time": generation_time,
            "hallucination_indicators": hallucination_indicators,
            "hallucination_score": hallucination_score
        })
        
        logger.info(f"  Hallucination score: {hallucination_score:.2f}")
        if hallucination_indicators["uncertainty_acknowledged"]:
            logger.info(f"  ✓ Uncertainty acknowledged")
        if hallucination_indicators["fictional_references"]:
            logger.info(f"  ✗ Contains fictional references")
        
        # Small pause to avoid rate limiting
        time.sleep(1)
    
    # Save complete results
    with open(os.path.join(output_dir, "context_enrichment_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """Main function to run the system prompt effectiveness tests."""
    
    # Define hallucination-prone queries for testing
    hallucination_triggers = [
        {
            "category": "fictional_library",
            "query": "Explain how to use the QuantumState library for quantum computing in Python."
        },
        {
            "category": "api_endpoint",
            "query": "How do I call the /api/v2/neural_synthesis endpoint in my application?"
        },
        {
            "category": "nonexistent_pattern",
            "query": "Implement the Observer-Commander-Adapter design pattern in JavaScript."
        },
        {
            "category": "ambiguous",
            "query": "Create a data structure for managing all the things efficiently."
        },
        {
            "category": "imprecise",
            "query": "How does the latest version of that popular web framework handle state?"
        }
    ]
    
    # Test different prompt types
    prompt_types = ["default", "coding", "refined_coding"]
    
    # Run the tests
    logger.info("Starting system prompt effectiveness tests")
    results = evaluate_prompt_on_hallucination_triggers(hallucination_triggers, prompt_types)
    
    # Test context enrichment
    logger.info("Starting context enrichment tests")
    context_results = test_context_enrichment()
    
    # Print summary
    print("\n=== System Prompt Effectiveness Summary ===")
    
    # Print aggregate results
    print("\nHallucination scores by prompt type (lower is better):")
    for prompt_type, metrics in results["aggregate_results"].items():
        print(f"  {prompt_type}: {metrics['avg_hallucination_score']:.2f} hallucination score")
        print(f"    Uncertainty acknowledged: {metrics['uncertainty_acknowledged_rate']:.0%}")
        print(f"    Fictional references: {metrics['fictional_references_rate']:.0%}")
    
    # Print best prompt
    print(f"\nBest prompt type: {results['best_prompt']}")
    
    # Print context enrichment results
    print("\nContext enrichment results:")
    for result in context_results:
        print(f"  {result['context_level']}: {result['hallucination_score']:.2f} hallucination score")
    
    # Print recommendation
    print("\nRecommendation based on test results:")
    print(f"  Use the '{results['best_prompt']}' prompt type with detailed context enrichment for best results.")
    
    print(f"\nDetailed results saved to reports/prompt_effectiveness_results.json and reports/context_enrichment_results.json")


if __name__ == "__main__":
    main() 