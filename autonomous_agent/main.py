#!/usr/bin/env python3
"""
Autonomous Coding Agent

A locally-hosted autonomous coding agent with advanced memory systems,
self-improvement capabilities, and multi-agent collaboration.
"""

import os
import sys
import click
import time
from loguru import logger
import re
from typing import Dict, List, Optional, Tuple

# Fix import paths when run directly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Import project modules
from config import LOGGING_CONFIG, DEV_MODE, LEARNING_CONFIG
from utils.logger import setup_logger


@click.group()
@click.option('--debug/--no-debug', default=DEV_MODE, help='Enable debug mode')
def cli(debug):
    """Autonomous Coding Agent CLI."""
    # Configure logging
    setup_logger(debug)
    if debug:
        logger.info("Debug mode enabled")


def check_response_quality(response: str) -> Tuple[bool, str]:
    """
    Check the quality of an agent response to detect issues.
    
    Args:
        response (str): The response to check.
        
    Returns:
        Tuple[bool, str]: A tuple of (is_acceptable, reason)
    """
    # Empty responses are unacceptable
    if not response:
        return False, "Empty response"
    
    if len(response.strip()) < 10:
        return False, "Extremely short response"
    
    # Check for single character patterns
    if response.strip() in ['?', '.', ',', ':', ';']:
        return False, "Single character response"
    
    # Check for email-only responses (simple pattern)
    if '@' in response and '.' in response and len(response.strip().split()) <= 2:
        return False, "Email address only"
    
    # Check if response is just question marks with newlines
    if all(c in '?\n ' for c in response):
        return False, "Question mark only response"
    
    # Check for coherent sentences
    sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 3]
    if not sentences:
        return False, "No coherent sentences"
    
    # Check for code-only responses
    code_blocks = response.count('```')
    if code_blocks >= 2 and len(response.replace('```', '').strip()) < 20:
        return False, "Code-only response without explanation"
    
    return True, "Response appears acceptable"


@cli.command()
@click.option('--model', default=None, help='Custom model name to use')
@click.option('--enable-learning/--disable-learning', default=True, help='Enable or disable learning system')
def interactive(model, enable_learning):
    """Start the interactive chat session with the agent."""
    try:
        logger.info("Starting interactive session")
        # Import here to avoid circular imports
        from models.llm_interface import get_llm
        from agents.orchestrator import AgentOrchestrator
        
        # Initialize the LLM
        llm = get_llm(model)
        logger.info(f"Using model: {llm.model_name}")
        
        # Initialize the learning system if enabled
        learning_manager = None
        if enable_learning:
            try:
                from learning.manager import get_learning_manager
                learning_manager = get_learning_manager(LEARNING_CONFIG)
                logger.info("Learning system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize learning system: {e}")
                print("Note: Learning system could not be initialized. Continuing without learning capabilities.")
        
        # Initialize the orchestrator
        orchestrator = AgentOrchestrator(llm=llm, learning_manager=learning_manager)
        
        # Start the interactive session
        logger.info("Interactive session ready. Type 'exit' to quit.")
        print("\n===== Autonomous Coding Agent =====")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'reset' to reset the conversation.")
        if enable_learning:
            print("Learning system: Enabled")
        else:
            print("Learning system: Disabled")
        print("======================================\n")
        
        # Track conversation for learning
        conversation = []
        conversation_id = f"conversation_{int(time.time())}"
        
        # Track low-quality responses
        consecutive_low_quality = 0
        max_low_quality = 2  # Maximum consecutive low-quality responses before warning
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            if user_input.lower() == 'reset':
                # Reset conversation
                orchestrator.clear_memory()
                conversation = []
                conversation_id = f"conversation_{int(time.time())}"
                consecutive_low_quality = 0
                print("\nConversation has been reset.\n")
                continue
                
            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})
            
            response = orchestrator.process_input(user_input)
            
            # Check response quality
            is_acceptable, reason = check_response_quality(response)
            
            if not is_acceptable:
                consecutive_low_quality += 1
                logger.warning(f"Low quality response detected: {reason}")
                
                if consecutive_low_quality >= max_low_quality:
                    print("\n[WARNING] Multiple low-quality responses detected. The model may be experiencing issues.")
                    print("[WARNING] Consider typing 'reset' to start a fresh conversation.\n")
            else:
                consecutive_low_quality = 0
            
            print(f"\nAgent: {response}\n")
            
            # Add agent response to conversation
            conversation.append({"role": "assistant", "content": response})
            
        # Process the conversation for learning if enabled
        if enable_learning and learning_manager and conversation:
            print("\nProcessing conversation for learning...")
            try:
                learning_results = learning_manager.learn_from_conversation(
                    messages=conversation,
                    conversation_id=conversation_id
                )
                print(f"Extracted {len(learning_results['experiences'])} experiences")
                print(f"Detected {len(learning_results['feedback'])} feedback items")
                print(f"Extracted {len(learning_results['knowledge_items'])} knowledge items")
            except Exception as e:
                logger.error(f"Error processing conversation for learning: {e}")
                print("Failed to process conversation for learning.")
            
    except KeyboardInterrupt:
        print("\nSession terminated by user.")
    except Exception as e:
        logger.exception(f"Error in interactive session: {e}")
        print(f"\nAn error occurred: {e}")
    finally:
        logger.info("Interactive session ended")
        print("\nSession ended. Goodbye!")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--model', default=None, help='Custom model name to use')
def process_file(input_file, output_file, model):
    """Process a file containing code or instructions."""
    try:
        logger.info(f"Processing file: {input_file}")
        # Import here to avoid circular imports
        from models.llm_interface import get_llm
        from agents.coding_agent import CodingAgent
        
        # Initialize the LLM
        llm = get_llm(model)
        logger.info(f"Using model: {llm.model_name}")
        
        # Initialize the coding agent
        agent = CodingAgent(llm=llm)
        
        # Read the input file
        with open(input_file, 'r') as f:
            content = f.read()
            
        # Process the content
        result = agent.process_code(content)
        
        # Write the output
        with open(output_file, 'w') as f:
            f.write(result)
            
        logger.info(f"Output written to: {output_file}")
        print(f"\nProcessed {input_file} and wrote results to {output_file}")
        
    except Exception as e:
        logger.exception(f"Error processing file: {e}")
        print(f"\nAn error occurred: {e}")


@cli.command()
def install_model():
    """Install or update the local model."""
    try:
        import subprocess
        logger.info("Installing/updating the model with Ollama")
        
        print("\nChecking for Ollama installation...")
        try:
            subprocess.run(["ollama", "--version"], check=True, capture_output=True)
            print("Ollama is installed.")
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Ollama is not installed or not in PATH.")
            print("Please install Ollama from: https://ollama.ai/")
            return
        
        print("\nPulling the wizard-vicuna model...")
        subprocess.run(["ollama", "pull", "wizard-vicuna"], check=True)
        print("Model installed/updated successfully.")
        
    except Exception as e:
        logger.exception(f"Error installing model: {e}")
        print(f"\nAn error occurred: {e}")


@cli.command()
def setup():
    """Set up the environment for the agent."""
    try:
        logger.info("Setting up the environment")
        
        # Ensure all required directories exist
        from config import BASE_DIR, DATA_DIR, VECTOR_DB_DIR, CONVERSATION_HISTORY_DIR
        from config import LEARNING_CONFIG
        os.makedirs(BASE_DIR / "logs", exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        os.makedirs(CONVERSATION_HISTORY_DIR, exist_ok=True)
        
        # Create learning directories
        for dir_path in [
            LEARNING_CONFIG["experience_storage_dir"],
            LEARNING_CONFIG["feedback_storage_dir"],
            LEARNING_CONFIG["reflection_storage_dir"]
        ]:
            os.makedirs(dir_path, exist_ok=True)
        
        print("\n===== Setting up Autonomous Coding Agent =====")
        print("Created necessary directories.")
        
        # Check for required Python packages
        import importlib.util
        missing_packages = []
        for package in ["langchain", "openai", "chromadb", "sentence_transformers"]:
            if importlib.util.find_spec(package) is None:
                missing_packages.append(package)
        
        if missing_packages:
            print("\nThe following required packages are missing:")
            for package in missing_packages:
                print(f"  - {package}")
            print("\nPlease install them using:")
            print("  pip install -r requirements.txt")
        else:
            print("All required Python packages are installed.")
        
        # Check for Ollama
        import shutil
        if shutil.which("ollama") is None:
            print("\nOllama is not installed or not in PATH.")
            print("Please install Ollama from: https://ollama.ai/")
        else:
            print("Ollama is installed.")
            
        print("\nSetup complete!")
        
    except Exception as e:
        logger.exception(f"Error during setup: {e}")
        print(f"\nAn error occurred during setup: {e}")


@cli.command()
def learning_demo():
    """Run the learning system demonstration."""
    try:
        logger.info("Starting learning system demo")
        
        # Run the demo script
        import learning_demo
        learning_demo.main()
        
    except Exception as e:
        logger.exception(f"Error running learning demo: {e}")
        print(f"\nAn error occurred: {e}")


@cli.command()
def test_learning():
    """Run tests for the learning system."""
    try:
        logger.info("Running learning system tests")
        
        # Import unittest module
        import unittest
        import test_learning
        
        # Run the tests
        unittest.main(module=test_learning)
        
    except Exception as e:
        logger.exception(f"Error running learning tests: {e}")
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    # Import this here to avoid circular imports
    import time
    cli() 