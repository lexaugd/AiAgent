"""
Code execution tools for the Autonomous Coding Agent.

This module provides tools for safely executing code in isolated environments.
"""

import os
import sys
import tempfile
import subprocess
import shutil
import json
import uuid
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger

from tools.base import Tool


class CodeExecutionTool(Tool):
    """Tool for executing code in a sandboxed environment."""
    
    def __init__(self, workspace_dir: Optional[str] = None):
        """
        Initialize the code execution tool.
        
        Args:
            workspace_dir (str, optional): Directory to use for code execution. Default is a temp directory.
        """
        super().__init__(
            name="execute_code",
            description="Execute code in a sandboxed environment",
            required_permissions=["code_execution"]
        )
        self.workspace_dir = workspace_dir or tempfile.mkdtemp(prefix="agent_code_execution_")
        self.execution_timeouts = {
            "python": 10,  # 10 seconds for Python
            "javascript": 5,
            "default": 5   # Default timeout for other languages
        }
        
        # Create workspace directory if it doesn't exist
        os.makedirs(self.workspace_dir, exist_ok=True)
        logger.info(f"Code execution workspace: {self.workspace_dir}")
    
    def _execute(self, context: Dict[str, Any], code: str, language: str = "python", timeout: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute code in a sandboxed environment.
        
        Args:
            context (Dict[str, Any]): The context in which the tool is being executed.
            code (str): The code to execute.
            language (str, optional): The programming language. Defaults to "python".
            timeout (int, optional): Maximum execution time in seconds. Defaults to language-specific timeout.
            
        Returns:
            Dict[str, Any]: The result of the tool execution.
        """
        try:
            # Create a unique execution ID
            execution_id = str(uuid.uuid4())
            execution_dir = os.path.join(self.workspace_dir, execution_id)
            os.makedirs(execution_dir, exist_ok=True)
            
            # Set timeout based on language if not specified
            if timeout is None:
                timeout = self.execution_timeouts.get(language.lower(), self.execution_timeouts["default"])
            
            # Choose execution method based on language
            if language.lower() == "python":
                result = self._execute_python(code, execution_dir, timeout)
            elif language.lower() in ["javascript", "js", "node"]:
                result = self._execute_javascript(code, execution_dir, timeout)
            # Add other language handlers as needed
            else:
                return {
                    "success": False,
                    "error": f"Unsupported language: {language}"
                }
            
            # Return execution result
            return result
            
        except Exception as e:
            logger.exception(f"Error executing code: {e}")
            return {
                "success": False,
                "error": f"Error executing code: {str(e)}"
            }
        finally:
            # Clean up execution directory if not in debug mode
            if not context.get("debug_mode", False) and os.path.exists(execution_dir):
                try:
                    shutil.rmtree(execution_dir)
                except Exception as e:
                    logger.error(f"Failed to clean up execution directory: {e}")
    
    def _execute_python(self, code: str, execution_dir: str, timeout: int) -> Dict[str, Any]:
        """
        Execute Python code in a sandboxed environment.
        
        Args:
            code (str): The Python code to execute.
            execution_dir (str): Directory for execution.
            timeout (int): Maximum execution time in seconds.
            
        Returns:
            Dict[str, Any]: The result of the execution.
        """
        # Write code to a temporary file
        code_file = os.path.join(execution_dir, "code.py")
        with open(code_file, 'w') as f:
            f.write(code)
        
        # Create a file for output capture
        output_file = os.path.join(execution_dir, "output.txt")
        
        try:
            # Execute the code in a separate process with resource limits
            process = subprocess.run(
                [sys.executable, code_file],
                cwd=execution_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                text=True,
                check=False
            )
            
            # Capture the output
            stdout = process.stdout
            stderr = process.stderr
            
            # Check for execution status
            success = process.returncode == 0
            
            return {
                "success": success,
                "result": stdout,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": process.returncode,
                "execution_dir": execution_dir if success else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution timed out after {timeout} seconds",
                "timeout": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error during execution: {str(e)}"
            }
    
    def _execute_javascript(self, code: str, execution_dir: str, timeout: int) -> Dict[str, Any]:
        """
        Execute JavaScript code using Node.js in a sandboxed environment.
        
        Args:
            code (str): The JavaScript code to execute.
            execution_dir (str): Directory for execution.
            timeout (int): Maximum execution time in seconds.
            
        Returns:
            Dict[str, Any]: The result of the execution.
        """
        # Check if Node.js is installed
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            return {
                "success": False,
                "error": "Node.js is not installed or not in PATH"
            }
        
        # Write code to a temporary file
        code_file = os.path.join(execution_dir, "code.js")
        with open(code_file, 'w') as f:
            f.write(code)
        
        try:
            # Execute the code using Node.js
            process = subprocess.run(
                ["node", code_file],
                cwd=execution_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                text=True,
                check=False
            )
            
            # Capture the output
            stdout = process.stdout
            stderr = process.stderr
            
            # Check for execution status
            success = process.returncode == 0
            
            return {
                "success": success,
                "result": stdout,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": process.returncode,
                "execution_dir": execution_dir if success else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution timed out after {timeout} seconds",
                "timeout": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error during execution: {str(e)}"
            }
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Return a schema describing the tool's parameters and return value.
        
        Returns:
            Dict[str, Any]: The schema for the tool.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "code": "Code to execute",
                "language": "Programming language of the code (default: python)",
                "timeout": "Maximum execution time in seconds (optional)"
            },
            "returns": {
                "success": "Boolean indicating whether execution was successful",
                "result": "Output of the code execution if successful",
                "stdout": "Standard output from the execution",
                "stderr": "Standard error from the execution",
                "return_code": "Process return code",
                "error": "Error message if execution failed",
                "timeout": "Boolean indicating if execution timed out"
            }
        }


class FileExecutionTool(Tool):
    """Tool for executing code from a file in a sandboxed environment."""
    
    def __init__(self):
        """Initialize the file execution tool."""
        super().__init__(
            name="execute_file",
            description="Execute code from a file in a sandboxed environment",
            required_permissions=["code_execution", "file_read"]
        )
        # Use the same workspace as CodeExecutionTool
        self.code_execution_tool = CodeExecutionTool()
    
    def _execute(self, context: Dict[str, Any], file_path: str, language: Optional[str] = None, 
                 timeout: Optional[int] = None, args: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute code from a file in a sandboxed environment.
        
        Args:
            context (Dict[str, Any]): The context in which the tool is being executed.
            file_path (str): Path to the file containing code to execute.
            language (str, optional): The programming language. If None, inferred from file extension.
            timeout (int, optional): Maximum execution time in seconds.
            args (List[str], optional): Command line arguments to pass to the program.
            
        Returns:
            Dict[str, Any]: The result of the tool execution.
        """
        try:
            # Validate the file path
            validated_path = self._validate_path(file_path, context)
            if validated_path is None:
                return {
                    "success": False,
                    "error": f"Access denied: {file_path} is outside allowed directories"
                }
            
            # Check if the file exists
            if not os.path.isfile(validated_path):
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }
            
            # Determine language from file extension if not specified
            if language is None:
                ext = os.path.splitext(file_path)[1].lower()
                language = self._get_language_from_extension(ext)
                if language is None:
                    return {
                        "success": False,
                        "error": f"Could not determine language for file with extension {ext}"
                    }
            
            # Read the file
            with open(validated_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Execute the code
            execution_result = self.code_execution_tool._execute(
                context, 
                code, 
                language=language, 
                timeout=timeout, 
                args=args or []
            )
            
            # Add file information to the result
            execution_result["file_path"] = file_path
            execution_result["file_size"] = os.path.getsize(validated_path)
            
            return execution_result
            
        except Exception as e:
            logger.exception(f"Error executing file {file_path}: {e}")
            return {
                "success": False,
                "error": f"Error executing file: {str(e)}"
            }
    
    def _validate_path(self, path: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Validate the path to ensure it's within allowed directories.
        
        Args:
            path (str): The path to validate.
            context (Dict[str, Any]): The context containing allowed directories.
            
        Returns:
            Optional[str]: The validated absolute path, or None if invalid.
        """
        # Get the allowed directories from context, default to current working directory
        allowed_dirs = context.get("allowed_directories", [os.getcwd()])
        
        # Convert to absolute path
        abs_path = os.path.abspath(path)
        
        # Check if the path is within any allowed directory
        for allowed_dir in allowed_dirs:
            if abs_path.startswith(os.path.abspath(allowed_dir)):
                return abs_path
        
        return None
    
    def _get_language_from_extension(self, ext: str) -> Optional[str]:
        """
        Determine language from file extension.
        
        Args:
            ext (str): File extension including the dot.
            
        Returns:
            Optional[str]: The language name, or None if unknown.
        """
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.sh': 'bash',
            '.rb': 'ruby',
            '.php': 'php',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust'
        }
        
        return language_map.get(ext.lower())
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Return a schema describing the tool's parameters and return value.
        
        Returns:
            Dict[str, Any]: The schema for the tool.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "file_path": "Path to the file containing code to execute",
                "language": "Programming language of the code (optional, inferred from file extension if not provided)",
                "timeout": "Maximum execution time in seconds (optional)",
                "args": "List of command line arguments to pass to the program (optional)"
            },
            "returns": {
                "success": "Boolean indicating whether execution was successful",
                "result": "Output of the code execution if successful",
                "stdout": "Standard output from the execution",
                "stderr": "Standard error from the execution",
                "return_code": "Process return code",
                "file_path": "Path to the executed file",
                "file_size": "Size of the executed file in bytes",
                "error": "Error message if execution failed",
                "timeout": "Boolean indicating if execution timed out"
            }
        } 