"""
Setup script for the Autonomous Coding Agent package.
"""

from setuptools import setup, find_packages
import os
import re

# Read the version from __init__.py
with open(os.path.join(os.path.dirname(__file__), "__init__.py")) as f:
    version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

# Read the requirements from requirements.txt
with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
    requirements = f.read().splitlines()

setup(
    name="autonomous_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.0",
        "loguru>=0.6.0",
        "openai>=1.0.0",
        "langchain>=0.0.27",
        "chromadb>=0.4.15",
        "sentence_transformers>=2.2.2",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "tiktoken>=0.5.1",
        "pyyaml>=6.0.0"
    ],
    entry_points={
        "console_scripts": [
            "agent=autonomous_agent.main:cli",
        ],
    },
    python_requires=">=3.8",
    author="AI Agent Developer",
    author_email="dev@example.com",
    description="An autonomous coding agent with local LLM capabilities",
    keywords="ai, llm, autonomous, agent, coding",
    url="https://github.com/user/autonomous_agent",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 