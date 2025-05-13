# Autonomous Agent Tool System

This directory contains the tool implementations that allow the autonomous agent to interact with its environment.

## Implementation Roadmap

### Phase 1: Basic File Operations

1. **Create the Base Tool Interface**
   - Implement the `Tool` base class with standard interface
   - Add context handling for tool execution
   - Create security and permission system
   - Build error handling and result processing

2. **Implement File Operation Tools**
   - `FileReadTool`: Read the contents of files
   - `FileWriteTool`: Write content to files
   - `FileListTool`: List directory contents
   - `FileInfoTool`: Get information about files

### Phase 2: Code Execution

1. **Create Sandbox Environment**
   - Implement isolated execution environment
   - Add resource limitations (time, memory)
   - Create result capturing mechanism
   - Build error handling and recovery

2. **Implement Code Execution Tools**
   - `CodeExecuteTool`: Execute code in a safe environment
   - `TestRunTool`: Run tests on code
   - `DependencyTool`: Manage project dependencies
   - `EnvironmentTool`: Configure execution environment

### Phase 3: Advanced Tools

1. **Web Interaction Tools**
   - `WebSearchTool`: Perform web searches for information
   - `DocLookupTool`: Find documentation for libraries
   - `ApiTool`: Interact with web APIs

2. **Project Management Tools**
   - `ProjectInitTool`: Initialize new projects
   - `GitTool`: Interface with Git repositories
   - `PackagingTool`: Create distributable packages

## Tool Design Principles

1. **Safety First**: All tools must implement proper error handling and security measures
2. **Consistent Interface**: All tools follow the same basic interface for predictability
3. **Clear Feedback**: Tools provide detailed information about their execution results
4. **Permission System**: Tools operate within a permission framework to prevent unintended actions
5. **Dry Run Option**: All destructive tools support a dry-run mode for testing

## Tool Base Class Interface

```python
class Tool:
    """Base class for all tools available to the agent."""
    
    def __init__(self, name: str, description: str, required_permissions: List[str] = None):
        self.name = name
        self.description = description
        self.required_permissions = required_permissions or []
        
    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute the tool with the given arguments."""
        raise NotImplementedError("Tool subclasses must implement execute()")
        
    def get_schema(self) -> Dict[str, Any]:
        """Return a schema describing the tool's parameters and return value."""
        raise NotImplementedError("Tool subclasses must implement get_schema()")
        
    def check_permissions(self, context: Dict[str, Any]) -> bool:
        """Check if the tool has the required permissions to execute."""
        # Implementation will verify permissions from context
        pass
```

## Usage Example

```python
# Registering tools with the agent
file_tools = [
    FileReadTool(),
    FileWriteTool(),
    FileListTool()
]

agent = AutonomousCodeAgent(
    name="Coder",
    tools=file_tools
)

# Tool usage in the agent
def _execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a plan using available tools."""
    results = {}
    
    for step in plan['steps']:
        tool_name = step['tool']
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            
            # Check permissions
            if not tool.check_permissions(self.context):
                results[step['id']] = {
                    "success": False, 
                    "error": "Permission denied"
                }
                continue
                
            # Execute the tool
            try:
                results[step['id']] = tool.execute(**step['args'])
            except Exception as e:
                results[step['id']] = {
                    "success": False,
                    "error": str(e)
                }
        else:
            results[step['id']] = {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }
            
    return results
```

## Implementation Guidelines

1. Always include proper error handling in tool implementations
2. Use dry-run mode for potentially destructive operations
3. Implement proper resource management (open/close files, etc.)
4. Log all tool operations for debugging and auditing
5. Provide detailed error messages for failed operations
6. Include examples in the tool documentation
7. Add unit tests for each tool 



cd autonomous_agent && python main.py interactive --model wizard-vicuna-13b