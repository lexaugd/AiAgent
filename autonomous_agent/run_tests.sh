#!/bin/bash
# Script to run tests for the autonomous agent fixes

echo "===== Autonomous Coding Agent Test Runner ====="
echo "Running tests to verify LLM interface fixes..."

# Check for dependencies
if ! pip show openai >/dev/null 2>&1; then
  echo "Installing required dependencies..."
  pip install openai loguru
fi

# Check if Ollama is running
echo "Checking Ollama service..."
if curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
  echo "Ollama is running."
else
  echo "WARNING: Ollama does not appear to be running!"
  echo "Please start Ollama with 'ollama serve' before running tests."
  exit 1
fi

# Run the tests
echo -e "\n=== Running LLM Interface Test ==="
python test_llm_fix.py

echo -e "\n=== Running Debug Agent Test ==="
python debug_agent.py

echo -e "\n=== Running Ollama Connection Test ==="
python test_ollama.py

# Run a quick interactive test
echo -e "\n=== Running Quick Interactive Test ==="
echo "This will start the agent in interactive mode for a quick test."
echo "Type 'exit' after testing a few messages to continue."
echo ""
python main.py interactive --model wizard-vicuna-13b

echo -e "\n===== All tests completed! ====="
echo "Check the outputs above for any errors or unexpected behavior." 