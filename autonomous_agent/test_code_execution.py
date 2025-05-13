#!/usr/bin/env python3
"""
Test script for code execution tools.
"""

import os
import sys
import tempfile
from loguru import logger

from tools.code_execution import CodeExecutionTool, FileExecutionTool

def test_python_execution():
    """Test executing Python code."""
    print("Testing Python code execution...")
    
    code_execution = CodeExecutionTool()
    
    # Test simple code
    python_code = """
print("Hello from Python!")
x = 10
y = 20
print(f"The sum of {x} and {y} is {x + y}")
"""

    result = code_execution._execute({}, python_code, language="python")
    print(f"Success: {result['success']}")
    print(f"Output: {result['result']}")
    
    # Test code with error
    error_code = """
print("This will run")
x = 1 / 0  # This will cause an error
print("This won't run")
"""

    result = code_execution._execute({}, error_code, language="python")
    print(f"Error Test - Success: {result['success']}")
    print(f"Error Test - Error: {result.get('stderr', '')}")
    
    # Test timeout
    timeout_code = """
import time
print("Starting long operation...")
time.sleep(20)  # This should time out
print("Finished long operation")
"""

    result = code_execution._execute({}, timeout_code, language="python", timeout=2)
    print(f"Timeout Test - Success: {result['success']}")
    print(f"Timeout Test - Error: {result.get('error', '')}")

def test_file_execution():
    """Test executing code from a file."""
    print("\nTesting file execution...")
    
    # Create a temporary file with code
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
        temp_file.write("""
import sys
print("Hello from a file!")
print(f"Python version: {sys.version}")
print(f"Arguments: {sys.argv[1:]}")
""")
        file_path = temp_file.name
    
    try:
        file_execution = FileExecutionTool()
        
        # Set up context with allowed directories
        context = {
            "allowed_directories": [os.path.dirname(file_path)],
            "permissions": ["code_execution", "file_read"]
        }
        
        # Test file execution
        result = file_execution._execute(context, file_path, args=["arg1", "arg2"])
        print(f"File Execution - Success: {result['success']}")
        print(f"File Execution - Output: {result['result']}")
        print(f"File Execution - File Path: {result['file_path']}")
        print(f"File Execution - File Size: {result['file_size']} bytes")
        
    finally:
        # Clean up the temporary file
        os.unlink(file_path)

def test_javascript_execution():
    """Test executing JavaScript code if Node.js is available."""
    print("\nTesting JavaScript code execution...")
    
    code_execution = CodeExecutionTool()
    
    # Test if Node.js is available
    js_code = """
console.log("Hello from JavaScript!");
const x = 10;
const y = 20;
console.log(`The sum of ${x} and ${y} is ${x + y}`);
"""

    result = code_execution._execute({}, js_code, language="javascript")
    
    if result.get('error') and "Node.js is not installed" in result['error']:
        print("Skipping JavaScript test: Node.js is not installed")
        return
        
    print(f"Success: {result['success']}")
    print(f"Output: {result['result']}")

if __name__ == "__main__":
    print("=== Code Execution Tool Tests ===")
    test_python_execution()
    test_file_execution()
    test_javascript_execution()
    print("=== Tests Completed ===") 