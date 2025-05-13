"""
File operation tools for the Autonomous Coding Agent.

This module provides tools for file operations such as reading, writing, and listing files.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

from tools.base import Tool


class FileReadTool(Tool):
    """Tool for reading file contents."""
    
    def __init__(self):
        """Initialize the file read tool."""
        super().__init__(
            name="file_read",
            description="Read the contents of a file",
            required_permissions=["file_read"]
        )
    
    def _execute(self, context: Dict[str, Any], path: str, **kwargs) -> Dict[str, Any]:
        """
        Read the contents of a file.
        
        Args:
            context (Dict[str, Any]): The context in which the tool is being executed.
            path (str): The path to the file to read.
            
        Returns:
            Dict[str, Any]: The result of the tool execution.
        """
        try:
            # Ensure the path is valid and within allowed directories
            validated_path = self._validate_path(path, context)
            if validated_path is None:
                return {
                    "success": False,
                    "error": f"Access denied: {path} is outside allowed directories"
                }
            
            # Check if the file exists
            if not os.path.isfile(validated_path):
                return {
                    "success": False,
                    "error": f"File not found: {path}"
                }
            
            # Read the file
            with open(validated_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "success": True,
                "result": content,
                "path": validated_path
            }
        except Exception as e:
            logger.exception(f"Error reading file {path}: {e}")
            return {
                "success": False,
                "error": f"Error reading file: {str(e)}"
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
                "path": "Path to the file to read"
            },
            "returns": {
                "success": "Boolean indicating whether the file was read successfully",
                "result": "Content of the file if successful",
                "path": "The validated path that was read",
                "error": "Error message if the file could not be read"
            }
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


class FileWriteTool(Tool):
    """Tool for writing content to a file."""
    
    def __init__(self):
        """Initialize the file write tool."""
        super().__init__(
            name="file_write",
            description="Write content to a file",
            required_permissions=["file_write"]
        )
    
    def _execute(self, context: Dict[str, Any], path: str, content: str, mode: str = "w", **kwargs) -> Dict[str, Any]:
        """
        Write content to a file.
        
        Args:
            context (Dict[str, Any]): The context in which the tool is being executed.
            path (str): The path to the file to write.
            content (str): The content to write to the file.
            mode (str, optional): The file mode ('w' for write, 'a' for append). Defaults to "w".
            
        Returns:
            Dict[str, Any]: The result of the tool execution.
        """
        try:
            # Ensure the path is valid and within allowed directories
            validated_path = self._validate_path(path, context)
            if validated_path is None:
                return {
                    "success": False,
                    "error": f"Access denied: {path} is outside allowed directories"
                }
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(validated_path), exist_ok=True)
            
            # Validate the mode
            if mode not in ('w', 'a'):
                return {
                    "success": False,
                    "error": f"Invalid mode: {mode}. Must be 'w' or 'a'."
                }
            
            # Check if the file exists in write mode and we're not supposed to overwrite
            if mode == 'w' and os.path.exists(validated_path) and not context.get("allow_overwrite", False):
                return {
                    "success": False,
                    "error": f"File already exists: {path}. Set allow_overwrite=True to overwrite."
                }
            
            # Write to the file
            with open(validated_path, mode, encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "result": f"Content written to {path}",
                "path": validated_path,
                "bytes_written": len(content.encode('utf-8'))
            }
        except Exception as e:
            logger.exception(f"Error writing to file {path}: {e}")
            return {
                "success": False,
                "error": f"Error writing to file: {str(e)}"
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
                "path": "Path to the file to write",
                "content": "Content to write to the file",
                "mode": "File mode ('w' for write, 'a' for append). Defaults to 'w'."
            },
            "returns": {
                "success": "Boolean indicating whether the file was written successfully",
                "result": "Message indicating the result of the operation",
                "path": "The validated path that was written",
                "bytes_written": "Number of bytes written to the file",
                "error": "Error message if the file could not be written"
            }
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


class FileListTool(Tool):
    """Tool for listing files in a directory."""
    
    def __init__(self):
        """Initialize the file list tool."""
        super().__init__(
            name="file_list",
            description="List files in a directory",
            required_permissions=["file_read"]
        )
    
    def _execute(self, context: Dict[str, Any], directory: str = ".", include_hidden: bool = False, **kwargs) -> Dict[str, Any]:
        """
        List files in a directory.
        
        Args:
            context (Dict[str, Any]): The context in which the tool is being executed.
            directory (str, optional): The directory to list. Defaults to ".".
            include_hidden (bool, optional): Whether to include hidden files. Defaults to False.
            
        Returns:
            Dict[str, Any]: The result of the tool execution.
        """
        try:
            # Ensure the path is valid and within allowed directories
            validated_path = self._validate_path(directory, context)
            if validated_path is None:
                return {
                    "success": False,
                    "error": f"Access denied: {directory} is outside allowed directories"
                }
            
            # Check if the directory exists
            if not os.path.isdir(validated_path):
                return {
                    "success": False,
                    "error": f"Directory not found: {directory}"
                }
            
            # List files
            files = []
            directories = []
            
            for item in os.listdir(validated_path):
                # Skip hidden files if not requested
                if not include_hidden and item.startswith('.'):
                    continue
                
                item_path = os.path.join(validated_path, item)
                if os.path.isfile(item_path):
                    files.append({
                        "name": item,
                        "path": os.path.join(directory, item),
                        "size": os.path.getsize(item_path)
                    })
                elif os.path.isdir(item_path):
                    directories.append({
                        "name": item,
                        "path": os.path.join(directory, item)
                    })
            
            return {
                "success": True,
                "result": {
                    "files": files,
                    "directories": directories,
                    "path": directory
                }
            }
        except Exception as e:
            logger.exception(f"Error listing directory {directory}: {e}")
            return {
                "success": False,
                "error": f"Error listing directory: {str(e)}"
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
                "directory": "Path to the directory to list. Defaults to '.'.",
                "include_hidden": "Whether to include hidden files. Defaults to False."
            },
            "returns": {
                "success": "Boolean indicating whether the directory was listed successfully",
                "result": {
                    "files": "List of files in the directory",
                    "directories": "List of subdirectories in the directory",
                    "path": "The directory that was listed"
                },
                "error": "Error message if the directory could not be listed"
            }
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


class FileInfoTool(Tool):
    """Tool for getting information about a file."""
    
    def __init__(self):
        """Initialize the file info tool."""
        super().__init__(
            name="file_info",
            description="Get information about a file",
            required_permissions=["file_read"]
        )
    
    def _execute(self, context: Dict[str, Any], path: str, **kwargs) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            context (Dict[str, Any]): The context in which the tool is being executed.
            path (str): The path to the file to get information about.
            
        Returns:
            Dict[str, Any]: The result of the tool execution.
        """
        try:
            # Ensure the path is valid and within allowed directories
            validated_path = self._validate_path(path, context)
            if validated_path is None:
                return {
                    "success": False,
                    "error": f"Access denied: {path} is outside allowed directories"
                }
            
            # Check if the path exists
            if not os.path.exists(validated_path):
                return {
                    "success": False,
                    "error": f"Path not found: {path}"
                }
            
            # Get file information
            stat_info = os.stat(validated_path)
            
            result = {
                "path": path,
                "absolute_path": validated_path,
                "exists": True,
                "type": "directory" if os.path.isdir(validated_path) else "file",
                "size": stat_info.st_size,
                "last_modified": stat_info.st_mtime,
                "permissions": stat_info.st_mode
            }
            
            # If it's a file, add extension and first/last few lines
            if os.path.isfile(validated_path):
                # Get file extension
                _, ext = os.path.splitext(validated_path)
                result["extension"] = ext.lstrip('.')
                
                # Read first and last few lines for text files
                try:
                    if self._is_text_file(validated_path):
                        with open(validated_path, 'r', encoding='utf-8') as f:
                            # Read first 5 lines
                            first_lines = []
                            for _ in range(5):
                                line = f.readline().rstrip()
                                if not line:
                                    break
                                first_lines.append(line)
                            
                            result["first_lines"] = first_lines
                            
                            # Try to read last 5 lines if file is not too large
                            if stat_info.st_size < 1024 * 1024:  # 1MB limit
                                result["last_lines"] = self._get_last_lines(validated_path, 5)
                except:
                    # If we can't read the file, just skip these fields
                    pass
            
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            logger.exception(f"Error getting file info for {path}: {e}")
            return {
                "success": False,
                "error": f"Error getting file info: {str(e)}"
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
                "path": "Path to the file to get information about"
            },
            "returns": {
                "success": "Boolean indicating whether the file info was retrieved successfully",
                "result": {
                    "path": "The path that was provided",
                    "absolute_path": "The absolute path to the file",
                    "exists": "Whether the file exists",
                    "type": "The type of the path (file or directory)",
                    "size": "The size of the file in bytes",
                    "last_modified": "The last modified timestamp",
                    "permissions": "The file permissions",
                    "extension": "The file extension (for files only)",
                    "first_lines": "The first few lines of the file (for text files only)",
                    "last_lines": "The last few lines of the file (for text files only)"
                },
                "error": "Error message if the file info could not be retrieved"
            }
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
    
    def _is_text_file(self, path: str) -> bool:
        """
        Check if a file is a text file.
        
        Args:
            path (str): The path to the file.
            
        Returns:
            bool: True if the file is a text file, False otherwise.
        """
        # Simple heuristic based on extension
        text_extensions = {
            'txt', 'md', 'py', 'js', 'html', 'css', 'json', 'xml', 'yaml', 'yml',
            'c', 'cpp', 'h', 'hpp', 'java', 'go', 'rb', 'php', 'ts', 'sh', 'bat',
            'rs', 'scala', 'kt', 'swift', 'dart', 'lua', 'pl', 'pm', 'r'
        }
        
        _, ext = os.path.splitext(path)
        if ext.lstrip('.').lower() in text_extensions:
            return True
        
        # Try reading a small portion of the file to check if it's text
        try:
            with open(path, 'r', encoding='utf-8') as f:
                f.read(1024)
            return True
        except:
            return False
    
    def _get_last_lines(self, path: str, n: int) -> List[str]:
        """
        Get the last n lines of a file.
        
        Args:
            path (str): The path to the file.
            n (int): The number of lines to get.
            
        Returns:
            List[str]: The last n lines of the file.
        """
        # Simple implementation - read the whole file and get the last n lines
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            return [line.rstrip() for line in lines[-n:] if line.strip()]
        except:
            return [] 