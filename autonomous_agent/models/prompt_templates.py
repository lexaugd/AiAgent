"""
Prompt templates for the Autonomous Coding Agent.

This module defines system prompts and templates for various agent functions.
"""

import os
from typing import Dict, Optional
from pathlib import Path


def get_system_prompt(prompt_type: str = "default") -> str:
    """
    Get a system prompt based on the specified type.
    
    Args:
        prompt_type (str): Type of system prompt to retrieve.
            Options: "default", "coding", "refined_coding", "review", "debugging"
        
    Returns:
        str: The system prompt text.
    """
    # Map prompt types to their file locations
    prompt_files = {
        "default": "default_prompt.md",
        "coding": "coding_prompt.md",
        "refined_coding": "refined_coding_prompt.md",
        "review": "review_prompt.md",
        "debugging": "debugging_prompt.md"
    }
    
    # Get the corresponding file
    prompt_file = prompt_files.get(prompt_type, "default_prompt.md")
    
    # Construct the path to the prompt file
    prompts_dir = Path(__file__).parent.parent / "system_prompts"
    prompt_path = prompts_dir / prompt_file
    
    # Check if the file exists
    if not prompt_path.exists():
        # Return a basic default prompt if file not found
        return "You are an autonomous coding agent. Answer concisely and accurately. If you're uncertain, acknowledge your uncertainty rather than providing potentially incorrect information."
    
    # Read and return the prompt
    with open(prompt_path, "r") as f:
        return f.read()


def create_prompt_with_context(
    user_message: str,
    system_prompt: Optional[str] = None,
    context: Optional[Dict] = None
) -> str:
    """
    Create a prompt with system instructions and relevant context.
    
    Args:
        user_message (str): The user's message.
        system_prompt (str, optional): System prompt to use.
        context (Dict, optional): Additional context to include.
            Can contain: relevant_code, docs, examples, etc.
            
    Returns:
        str: The complete prompt.
    """
    # Use default system prompt if none provided
    if system_prompt is None:
        system_prompt = get_system_prompt("refined_coding")
    
    # Start with the system prompt
    prompt = f"{system_prompt}\n\n"
    
    # Add context if provided
    if context:
        prompt += "## Context Information\n\n"
        
        # Add relevant code snippets
        if "relevant_code" in context:
            prompt += "### Relevant Code\n\n"
            for code_item in context["relevant_code"]:
                file_path = code_item.get("file_path", "")
                content = code_item.get("content", "")
                if file_path and content:
                    prompt += f"File: {file_path}\n```\n{content}\n```\n\n"
            
        # Add documentation
        if "docs" in context:
            prompt += "### Documentation\n\n"
            for doc in context["docs"]:
                prompt += f"{doc}\n\n"
        
        # Add examples
        if "examples" in context:
            prompt += "### Examples\n\n"
            for example in context["examples"]:
                prompt += f"{example}\n\n"
        
        # Add conversation history
        if "conversation" in context:
            prompt += "### Conversation History\n\n"
            for msg in context["conversation"]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    prompt += f"{role.capitalize()}: {content}\n\n"
    
    # Add the user's current message
    prompt += f"## Current User Query\n\n{user_message}\n\n"
    
    # Add a final instruction to reduce hallucination
    prompt += "Remember to provide accurate information only. If you're unsure about something, acknowledge your uncertainty."
    
    return prompt 