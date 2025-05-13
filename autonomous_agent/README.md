# Autonomous Coding Agent

A locally-hosted autonomous coding agent with advanced memory systems, self-improvement capabilities, and multi-agent collaboration.

## Overview

This project creates an autonomous coding assistant using a locally-hosted Wizard-Vicuna-13B-Uncensored model. The agent can write, review, and refactor code, while maintaining both short-term and long-term memory to provide consistent and contextually relevant responses.

Key features:
- **100% Local Execution**: All processing happens on your machine, ensuring privacy and eliminating API costs
- **Advanced Memory Systems**: Both short-term conversation memory and long-term knowledge storage
- **Multi-Agent Collaboration**: Multiple specialized agents working together
- **Self-Improvement**: Mechanisms for learning from past interactions
- **Comprehensive Tooling**: File operations, code execution, and system command capabilities

## Requirements

- Python 3.8+
- 16GB+ RAM recommended (8GB minimum)
- Ollama installed (for running the LLM locally)
- GGUF-format model file (Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/autonomous-agent.git
cd autonomous-agent
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama (if not already installed):
   - Visit [Ollama's website](https://ollama.ai/) for installation instructions

4. Add your model to Ollama:
```bash
# If you already have the GGUF file
ollama create wizard-vicuna-13b -f /path/to/Wizard-Vicuna-13B-Uncensored.Q4_K_M.gguf

# OR pull a pre-built version
ollama pull wizard-vicuna
```

## Usage

1. Start the agent:
```bash
python main.py
```

2. Interact with the agent through the provided interface.

3. For advanced usage and configuration options, see the documentation.

## Project Structure

```
autonomous_agent/
├── models/           # Model configuration and interface
├── agents/           # Agent implementations
├── memory/           # Memory systems
├── tools/            # External tools integration
├── utils/            # Utility functions
└── data/             # Data storage
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgements

- The [LangChain](https://github.com/langchain-ai/langchain) project
- The [Ollama](https://ollama.ai/) project
- All contributors to the Wizard-Vicuna model 