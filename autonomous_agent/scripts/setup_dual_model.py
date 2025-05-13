#!/usr/bin/env python3
"""
Setup script for the dual-model architecture.

This script checks if the necessary models (DeepSeek-Coder and Phi-3-mini) are installed,
and downloads them if they're not.
"""

import os
import sys
import subprocess
import time
from typing import List, Tuple

# Add the parent directory to the path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def check_ollama_installed() -> bool:
    """Check if Ollama is installed."""
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False


def get_installed_models() -> List[str]:
    """Get the list of installed Ollama models."""
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        print(f"Error getting installed models: {result.stderr}")
        return []
    
    lines = result.stdout.strip().split("\n")
    
    # Skip the header line if it exists
    if lines and "NAME" in lines[0].upper():
        lines = lines[1:]
    
    # Extract model names (first column)
    models = []
    for line in lines:
        if line.strip():
            parts = line.split()
            if parts:
                models.append(parts[0])
    
    return models


def install_model(model_name: str) -> bool:
    """Install an Ollama model."""
    print(f"Installing {model_name}... (this may take a few minutes)")
    
    result = subprocess.run(["ollama", "pull", model_name], capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        print(f"Error installing {model_name}: {result.stderr}")
        return False
    
    print(f"Successfully installed {model_name}")
    return True


def test_model(model_name: str) -> bool:
    """Test if a model works properly."""
    print(f"Testing {model_name}...")
    
    test_prompt = "Respond with a single word: 'working'"
    
    try:
        result = subprocess.run(
            ["ollama", "run", model_name, test_prompt], 
            capture_output=True, 
            text=True, 
            check=False,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"Error testing {model_name}: {result.stderr}")
            return False
        
        if "working" in result.stdout.lower():
            print(f"{model_name} is working properly")
            return True
        else:
            print(f"{model_name} did not respond as expected. Response: {result.stdout[:100]}...")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"Test for {model_name} timed out after 30 seconds")
        return False


def setup_models() -> Tuple[bool, bool]:
    """Set up the necessary models for the dual-model architecture."""
    models_to_install = {
        "coding": "deepseek-coder:6.7b-instruct",
        "reasoning": "phi3:mini"
    }
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("Ollama is not installed. Please install Ollama first:")
        print("  Mac/Linux: curl -fsSL https://ollama.com/install.sh | sh")
        print("  Windows: Visit https://ollama.com/download")
        return False, False
    
    print("Ollama is installed")
    
    # Get installed models
    installed_models = get_installed_models()
    print(f"Currently installed models: {', '.join(installed_models) if installed_models else 'None'}")
    
    # Set up coding model
    coding_model = models_to_install["coding"]
    coding_installed = any(m.startswith("deepseek-coder") for m in installed_models)
    
    if coding_installed:
        print(f"Coding model ({coding_model}) is already installed")
    else:
        coding_installed = install_model(coding_model)
    
    # Set up reasoning model
    reasoning_model = models_to_install["reasoning"]
    reasoning_installed = any(m.startswith("phi3") for m in installed_models)
    
    if reasoning_installed:
        print(f"Reasoning model ({reasoning_model}) is already installed")
    else:
        reasoning_installed = install_model(reasoning_model)
    
    # Test models if they're installed
    if coding_installed:
        coding_working = test_model(coding_model)
    else:
        coding_working = False
    
    if reasoning_installed:
        reasoning_working = test_model(reasoning_model)
    else:
        reasoning_working = False
    
    return coding_working, reasoning_working


def main():
    """Main function."""
    print("=== Dual-Model Architecture Setup ===\n")
    
    coding_working, reasoning_working = setup_models()
    
    if coding_working and reasoning_working:
        print("\n✅ Both models are installed and working properly")
        print("You can now use the dual-model architecture")
    else:
        if not coding_working:
            print("\n❌ Coding model is not installed or not working properly")
        if not reasoning_working:
            print("\n❌ Reasoning model is not installed or not working properly")
        print("\nPlease fix the issues and try again")
    
    print("\nFor more information on how to use the dual-model architecture, see:")
    print("  autonomous_agent/docs/dual_model_architecture.md")


if __name__ == "__main__":
    main() 