# Dual-Model Architecture

## Overview

The Autonomous Coding Agent now features a dual-model architecture that leverages two specialized language models, each optimized for different types of tasks:

1. **Reasoning Model (Phi-3-mini)**: Handles tasks requiring deep reasoning, planning, explanation, and problem-solving.
2. **Coding Model (DeepSeek-Coder-6.7B-Instruct)**: Specializes in code generation, implementation, and code review.

This architecture allows the agent to excel at both high-level thinking tasks and concrete implementation tasks by routing queries to the most appropriate model.

## How It Works

### Task Classification

When a query is received, the ModelManager:

1. Uses keyword-based classification to determine the task type
   - Planning tasks → Reasoning model
   - Code generation tasks → Coding model
   - Code review tasks → Coding model
   - Problem-solving tasks → Reasoning model
   - Explanation tasks → Reasoning model
   - General tasks → Reasoning model (default)

2. Dynamically selects the appropriate model for the task

3. Formats the prompt with model-specific enhancements

4. Processes the response and returns the result

### Combined Planning and Implementation

For complex tasks that require both planning and implementation, the ModelManager provides a `generate_with_planning` method that:

1. Uses the reasoning model to create a detailed plan
2. Passes the plan to the coding model for implementation
3. Returns both the plan and the implementation

## Usage

### Basic Usage

```python
from autonomous_agent.models import get_model_manager

# Initialize the model manager
model_manager = get_model_manager()

# Simple queries are automatically routed to the appropriate model
response = model_manager.generate_response(
    messages=[{"role": "user", "content": "Explain how a hash table works"}]
)

# You can explicitly specify the task type
from autonomous_agent.models import TaskType

response = model_manager.generate_response(
    messages=[{"role": "user", "content": "Create a Python function to validate email addresses"}],
    task_type=TaskType.CODE_GENERATION
)
```

### Planning and Implementation

```python
# For complex tasks requiring both planning and implementation
plan, implementation = model_manager.generate_with_planning(
    messages=[{"role": "user", "content": "Create a Python class for a shopping cart"}]
)

print("Plan:")
print(plan)
print("\nImplementation:")
print(implementation)
```

### Advanced Configuration

You can configure the models used by the ModelManager in `config.py`:

```python
# Standard model configuration (used for coding tasks)
MODEL_CONFIG = {
    "name": "deepseek-coder:6.7b-instruct",
    "base_url": "http://localhost:11434/v1",
    "max_tokens": 4096,
    "temperature": 0.35,  # Optimized for code generation
    "top_p": 0.95,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1,
}

# Reasoning model configuration
REASONING_MODEL_CONFIG = {
    "name": "phi3:mini",
    "base_url": "http://localhost:11434/v1",
    "max_tokens": 4096,
    "temperature": 0.7,  # Higher for more creative reasoning
    "top_p": 0.95,
    "frequency_penalty": 0.2,
    "presence_penalty": 0.2,
}
```

You can also dynamically update model configurations:

```python
from autonomous_agent.models import ModelType

# Update reasoning model configuration
model_manager.update_model_config(
    model_type=ModelType.REASONING,
    config_updates={"temperature": 0.8}
)
```

## Performance Tracking

The ModelManager keeps track of performance statistics:

```python
# Get usage statistics
stats = model_manager.get_stats()
print(stats)
```

## Technical Details

### Model Specifications

1. **Reasoning Model (Phi-3-mini)**
   - Parameters: 3.8B
   - Context Length: Up to 128K tokens
   - Optimized for reasoning tasks
   - Strong on mathematical and logical tasks
   - Available via Ollama

2. **Coding Model (DeepSeek-Coder-6.7B-Instruct)**
   - Parameters: 6.7B
   - Specialized for code generation and understanding
   - Available via Ollama

### Implementation Considerations

- Both models run locally, requiring no API keys or external connections
- Resource requirements:
  - Phi-3-mini: ~8GB VRAM
  - DeepSeek-Coder: ~10GB VRAM
  - Running both models simultaneously requires adequate GPU memory or CPU capacity

## Integration with Memory System

The dual-model architecture integrates with the existing memory system:
- Both models share access to the same memory pools
- Memory manager provides consistent context retrieval
- Conversations are tracked across both models

## Future Improvements

- Enhanced task classification using semantic similarity
- Fine-tuned context retrieval based on model-specific needs
- Parallel model inference for complex tasks
- Fallback mechanisms for when models exceed resource constraints 