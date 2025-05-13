# Autonomous Coding Agent

A locally-hosted autonomous coding agent module with advanced memory systems, self-improvement capabilities, and a robust toolset, designed to be part of the larger Agnet project. It primarily uses the `deepseek-coder:6.7b-instruct` model for coding tasks and is being developed towards a dual-model architecture incorporating `phi3:mini` for reasoning.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Status](#project-status)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Development Roadmap](#development-roadmap)
- [Current Development Focus](#current-development-focus)
- [How to Contribute](#how-to-contribute)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project provides the core functionalities for an autonomous coding assistant. It leverages locally-hosted Large Language Models via Ollama, primarily using `deepseek-coder:6.7b-instruct` for code generation, explanation, and review. The system is evolving towards a dual-model architecture, with `phi3:mini` planned for enhanced reasoning capabilities. The agent is equipped with sophisticated memory systems (short-term and long-term) to maintain context and learn from interactions, aiming for consistent and contextually relevant responses. This module is a key component of the Agnet autonomous agent system.

## Key Features

-   **Local LLM Execution**: All processing via Ollama ensures privacy and eliminates API costs, primarily using `deepseek-coder:6.7b-instruct` for coding tasks, with `phi3:mini` planned for reasoning.
-   **Core Infrastructure**:
    -   Centralized configuration (`config.py`).
    -   Structured logging with Loguru.
    -   Command-Line Interface (CLI) powered by Click.
-   **Advanced Model Interface**:
    -   Abstraction layer for LLMs.
    -   Supports synchronous and asynchronous API calls.
    -   Streaming response capabilities for interactive feedback.
-   **Sophisticated Memory System**:
    -   **Short-Term Memory**: Conversation buffers with token limit management and persistence.
    -   **Long-Term Memory**: ChromaDB-backed vector storage for code and conceptual knowledge, with embedding generation and context-aware similarity search.
-   **Comprehensive Learning System**:
    -   Tracks experiences and user feedback.
    -   Extracts valuable knowledge from interactions.
    -   Self-reflection capabilities for planning improvements.
    -   Unified `LearningManager` to coordinate learning components.
-   **Flexible Agent System**:
    -   `BaseAgent` class with core functionalities.
    -   Specialized `CodingAgent` for code-related tasks.
    -   `AgentOrchestrator` to coordinate between different agent types (and planned for multi-model orchestration).
    -   Reflection capabilities for self-improvement.
-   **Code-Related Capabilities**:
    -   Code generation based on natural language requirements.
    -   Language detection for various programming languages.
    -   Code explanation and review functionalities.
-   **Extensible Tool System**:
    -   Base `Tool` class and permission system.
    -   File operation tools (read, write, list directories).
    -   Sandboxed code execution environment.
-   **Autonomous Operation**:
    -   Implements an observe-plan-act-reflect cycle.
    -   Task queue management and continuous operation mode.

## Project Status

The project has implemented core autonomous features, including advanced memory and learning systems using `deepseek-coder:6.7b-instruct` as the primary coding model. The initial phases focusing on basic autonomy, enhanced capabilities, and memory/learning integration are complete. Current development is focused on fully implementing and integrating a dual-model architecture (with `phi3:mini` for reasoning) and further enhancements to tooling and decision-making.

## Technology Stack

-   **Python**: 3.8+
-   **LLM Engine**: Ollama (with `deepseek-coder:6.7b-instruct` as primary coding model; `phi3:mini` as reasoning model for the planned dual-model architecture)
-   **Vector Database**: ChromaDB
-   **Core Libraries**:
    -   Langchain
    -   OpenAI (SDK for API compatibility)
    -   Sentence-Transformers (for embeddings)
    -   Pydantic (for data validation)
-   **CLI**: Click
-   **Logging**: Loguru
-   **API (for potential future extensions)**: FastAPI, Uvicorn
-   **Testing**: Pytest
-   (See `autonomous_agent/requirements.txt` for a full list of dependencies)

## Project Structure

The `autonomous_agent` module is organized as follows:

```
autonomous_agent/
├── agents/                # Agent implementations (base, coding, orchestrator)
├── config.py              # Centralized configuration including model settings
├── data/                  # Data storage (e.g., conversation history, vector_db for long-term memory)
├── learning/              # Learning system components (experience, feedback, knowledge, reflection)
├── logs/                  # Log files generated by the agent
├── main.py                # CLI entry point for interacting with the agent
├── memory/                # Memory implementations (short-term, long-term, embeddings)
├── models/                # Model interface (LLM communication)
├── requirements.txt       # Python package dependencies for this module
├── setup.py               # Package installation script for this module
├── system_prompts/        # Stores system prompts for different agent personalities/tasks
├── tools/                 # Tools for agent use (file operations, code execution)
├── utils/                 # Utility functions (e.g., logger setup)
└── README.md              # (This file is now at the project root)
```

## Installation

1.  **Clone the Agnet Repository**:
    ```bash
    git clone git@github.com:lexaugd/AiAgent.git
    cd AiAgent
    ```

2.  **Navigate to the Agent Module**:
    ```bash
    cd autonomous_agent 
    ```

3.  **Set up a Python Virtual Environment** (Recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate 
    ```

4.  **Install Dependencies**:
    From within the `autonomous_agent` directory:
    ```bash
    pip install -r requirements.txt
    ```
    If you intend to use this as an installable package, you can also run `pip install -e .` from within the `autonomous_agent` directory.

5.  **Install Ollama**:
    Ensure Ollama is installed and running. Visit [Ollama's website](https://ollama.ai/) for instructions.

6.  **Pull the LLM Models via Ollama**:
    The agent uses `deepseek-coder:6.7b-instruct` as the primary coding model and `phi3:mini` for reasoning (as part of the dual-model architecture). You can pull them using the provided script or manually:
    Using the script (from the `autonomous_agent` directory):
    ```bash
    python main.py install-model
    ```
    This command will attempt to pull `deepseek-coder:6.7b-instruct` and `phi3:mini`.
    Alternatively, pull manually:
    ```bash
    ollama pull deepseek-coder:6.7b-instruct
    ollama pull phi3:mini
    ```
    Refer to `autonomous_agent/config.py` for the exact model names used.

## Usage

The primary way to interact with the agent module is through its Command-Line Interface (`main.py`). Navigate to the `autonomous_agent` directory.

Available commands:

-   **Interactive Chat**:
    Uses the configured default model (currently `deepseek-coder:6.7b-instruct`).
    ```bash
    python main.py interactive
    ```
    Options:
    `--model <model_name>`: Specify a custom Ollama model (e.g., `phi3:mini` or other compatible models).
    `--enable-learning` / `--disable-learning`: Toggle the learning system (default: enabled).

-   **Process a File**:
    ```bash
    python main.py process-file <input_path.py> <output_path.py>
    ```
    Example: `python main.py process-file ../dev_log/summary.md processed_summary.txt`
    Option: `--model <model_name>`

-   **Install/Update Models**:
    Pulls the default `deepseek-coder:6.7b-instruct` and `phi3:mini` models via Ollama.
    ```bash
    python main.py install-model
    ```

-   **Environment Setup**:
    ```bash
    python main.py setup
    ```
-   **Learning System Demo**:
    ```bash
    python main.py learning-demo
    ```
-   **Test Learning Components**:
    ```bash
    python main.py test-learning
    ```

For debugging, `debug_agent.py` might offer specific testing functionalities.

## Development Roadmap

-   **Phase 1: Basic Autonomy** [COMPLETED]
-   **Phase 2: Enhanced Capabilities** [COMPLETED]
-   **Phase 3: Memory and Learning** [COMPLETED]
    (Using `deepseek-coder:6.7b-instruct` as the coding model)
-   **Phase 4: Dual-Model Architecture** [IN PROGRESS/PLANNED]
    -   Full integration and orchestration of `deepseek-coder:6.7b-instruct` (coding) and `phi3:mini` (reasoning).
    -   Model manager for coordinating multiple LLMs.
    -   Task classification and routing to appropriate models.

## Current Development Focus

-   Completing the dual-model architecture with `deepseek-coder:6.7b-instruct` and `phi3:mini`.
-   Expanding the tool ecosystem with more specialized capabilities (e.g., web search).
-   Implementing more sophisticated planning and reasoning, leveraging the dual-model setup.
-   Fine-tuning the learning system for better knowledge extraction and self-improvement.
-   Comprehensive integration testing with real-world development scenarios.

## How to Contribute

Contributions are welcome! Please feel free to submit a Pull Request. Ensure your contributions align with the project's coding standards and include relevant tests.

## License

MIT License. (Note: A `LICENSE` file should be added to the repository to formalize this.)

## Acknowledgements

-   The [LangChain](https://github.com/langchain-ai/langchain) project for concepts and inspiration.
-   The [Ollama](https://ollama.ai/) project for enabling local LLM execution.
-   The creators and contributors to the DeepSeek-Coder, Phi-3, and other open-source LLMs. 