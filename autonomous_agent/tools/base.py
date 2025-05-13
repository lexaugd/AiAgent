"""
Base Tool implementation for the Autonomous Coding Agent.

This module provides the base Tool class that all tools must extend.
"""

from typing import Dict, List, Any, Optional
from loguru import logger


class Tool:
    """Base class for all tools available to the agent."""
    
    def __init__(self, name: str, description: str, required_permissions: List[str] = None):
        """
        Initialize the tool.
        
        Args:
            name (str): The name of the tool.
            description (str): A description of what the tool does.
            required_permissions (List[str], optional): List of permissions required to use this tool.
        """
        self.name = name
        self.description = description
        self.required_permissions = required_permissions or []
        
    def execute(self, context: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with the given arguments.
        
        Args:
            context (Dict[str, Any], optional): The context in which the tool is being executed.
            **kwargs: Tool-specific arguments.
            
        Returns:
            Dict[str, Any]: The result of the tool execution with at least 'success' and 'result' or 'error' keys.
        """
        try:
            # Check permissions
            if not self.check_permissions(context or {}):
                return {
                    "success": False,
                    "error": f"Permission denied: missing {', '.join(self.get_missing_permissions(context or {}))}"
                }
            
            # Log the tool execution
            logger.debug(f"Executing tool: {self.name} with args: {kwargs}")
            
            # Execute the tool-specific implementation
            result = self._execute(context or {}, **kwargs)
            
            # Log the result
            logger.debug(f"Tool execution result: {self.name} -> success={result.get('success', False)}")
            
            return result
        except Exception as e:
            logger.exception(f"Error executing tool {self.name}: {e}")
            return {
                "success": False,
                "error": f"Error executing tool: {str(e)}"
            }
    
    def _execute(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Tool-specific implementation of the execution logic. Must be overridden by subclasses.
        
        Args:
            context (Dict[str, Any]): The context in which the tool is being executed.
            **kwargs: Tool-specific arguments.
            
        Returns:
            Dict[str, Any]: The result of the tool execution.
        """
        raise NotImplementedError("Tool subclasses must implement _execute()")
        
    def get_schema(self) -> Dict[str, Any]:
        """
        Return a schema describing the tool's parameters and return value.
        
        Returns:
            Dict[str, Any]: The schema for the tool.
        """
        # Default implementation provides a basic schema
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {},
            "returns": {
                "success": "Boolean indicating whether the tool execution was successful",
                "result": "Result of the tool execution if successful",
                "error": "Error message if the tool execution failed"
            }
        }
        
    def check_permissions(self, context: Dict[str, Any]) -> bool:
        """
        Check if the tool has the required permissions to execute.
        
        Args:
            context (Dict[str, Any]): The context containing permission information.
            
        Returns:
            bool: True if the tool has all required permissions, False otherwise.
        """
        if not self.required_permissions:
            return True  # No permissions required
            
        granted_permissions = context.get("permissions", [])
        return all(perm in granted_permissions for perm in self.required_permissions)
    
    def get_missing_permissions(self, context: Dict[str, Any]) -> List[str]:
        """
        Get the list of missing permissions.
        
        Args:
            context (Dict[str, Any]): The context containing permission information.
            
        Returns:
            List[str]: The list of missing permissions.
        """
        granted_permissions = context.get("permissions", [])
        return [perm for perm in self.required_permissions if perm not in granted_permissions] 