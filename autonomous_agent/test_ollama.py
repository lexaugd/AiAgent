#!/usr/bin/env python3
"""
Simple test script to verify Ollama connection.
"""

import sys
from openai import OpenAI

def test_ollama_connection():
    print("Testing connection to Ollama...")
    
    try:
        # Initialize the OpenAI client with Ollama endpoint
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="not-needed"  # Ollama doesn't require an API key
        )
        
        # Get available models
        print("Fetching models...")
        models = client.models.list()
        print(f"Available models: {models}")
        
        # Test chat completion
        print("\nTesting chat completion with wizard-vicuna-13b...")
        response = client.chat.completions.create(
            model="wizard-vicuna-13b",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            max_tokens=50
        )
        
        print(f"Response from model: {response.choices[0].message.content}")
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print(f"Error type: {type(e)}")
        return False

if __name__ == "__main__":
    success = test_ollama_connection()
    sys.exit(0 if success else 1) 