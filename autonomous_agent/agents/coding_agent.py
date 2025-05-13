"""
Coding agent implementation for the Autonomous Coding Agent.

This module provides a specialized agent for code generation and manipulation.
"""

import json
import re
from typing import Dict, List, Optional, Any, Callable
from loguru import logger

from models.llm_interface import LocalLLM
from agents.base_agent import BaseAgent

# Singleton instance
_coding_agent = None

class CodingAgent(BaseAgent):
    """
    Specialized agent for code generation and manipulation.
    """
    
    def __init__(
        self,
        name: str = "Code Assistant",
        llm: Optional[LocalLLM] = None,
        memory_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        learning_manager = None
    ):
        """
        Initialize the coding agent.
        
        Args:
            name (str): The name of the agent.
            llm (LocalLLM, optional): The LLM to use.
            memory_id (str, optional): The ID for the agent's memory.
            system_prompt (str, optional): The system prompt to use.
            tools (List[Callable], optional): The tools available to the agent.
            learning_manager: The learning manager instance.
        """
        # Ensure name is properly initialized before using it in _get_coding_system_prompt
        self.name = name
        
        # Initialize learning manager
        self.learning_manager = learning_manager
        
        # Use a coding-specific system prompt if none provided
        if system_prompt is None:
            system_prompt = self._get_coding_system_prompt()
            
        # Initialize the base agent
        super().__init__(
            name=name,
            llm=llm,
            memory_id=memory_id,
            system_prompt=system_prompt,
            tools=tools
        )
        
        logger.info(f"Initialized coding agent: {self.name} ({self.id})")
    
    def _get_coding_system_prompt(self) -> str:
        """
        Get a specialized system prompt for the coding agent.
        
        Returns:
            str: The system prompt.
        """
        return f"""You are {self.name}, an expert coding assistant.

IMPORTANT INSTRUCTIONS:
1. Provide clear, concise, and accurate code solutions
2. Explain complex concepts in simple terms
3. When answering coding questions, prioritize:
   - Correctness
   - Efficiency
   - Readability
   - Best practices
4. If you're unsure about something, acknowledge the uncertainty
5. Format your code with appropriate syntax highlighting
6. If asked to modify existing code, maintain the coding style
7. If asked about libraries or frameworks you're not familiar with, be upfront about limitations

You're designed to assist with programming questions, explain code, review code, and help with debugging."""
    
    def process_code(self, code_content: str) -> str:
        """
        Process a code snippet or file.
        
        Args:
            code_content (str): The code content to process.
            
        Returns:
            str: The processed code or analysis.
        """
        # Determine the language from the code content
        language = self._detect_language(code_content)
        
        # Create a specific prompt for code processing
        prompt = f"""Please analyze and improve the following {language} code:

```{language}
{code_content}
```

Focus on:
1. Improving code quality and readability
2. Fixing any bugs or issues
3. Optimizing performance where possible
4. Enhancing documentation and comments
5. Applying best practices and design patterns

Provide the improved code with detailed explanations of your changes."""
        
        # Process the code
        result = self.process(prompt)
        
        logger.debug(f"Processed code in language: {language}")
        return result
    
    def generate_code(self, requirements: str, language: str) -> str:
        """
        Generate code based on requirements.
        
        Args:
            requirements (str): The requirements for the code.
            language (str): The programming language to use.
            
        Returns:
            str: The generated code.
        """
        # Create a specific prompt for code generation
        prompt = f"""Generate {language} code that meets these requirements:

{requirements}

The code should be:
1. Well-structured and organized
2. Properly documented with comments
3. Following best practices for {language}
4. Efficient and optimized
5. Easy to understand and maintain

Provide only the code without additional explanations."""
        
        # Generate the code
        result = self.process(prompt)
        
        # Extract code blocks from the result
        code_blocks = self._extract_code_blocks(result, language)
        
        if code_blocks:
            # Return the largest code block (assuming it's the main implementation)
            return max(code_blocks, key=len)
        else:
            # If no code blocks found, return the raw result
            return result
    
    def explain_code(self, code_content: str) -> str:
        """
        Explain a code snippet.
        
        Args:
            code_content (str): The code content to explain.
            
        Returns:
            str: The explanation.
        """
        # Determine the language from the code content
        language = self._detect_language(code_content)
        
        # Create a specific prompt for code explanation
        prompt = f"""Explain the following {language} code in detail:

```{language}
{code_content}
```

Include:
1. Overall purpose and functionality
2. Explanation of the key components and how they work
3. Notable algorithms or design patterns used
4. Potential edge cases or limitations
5. Suggestions for improvements (if any)

Make your explanation clear and accessible to developers of different skill levels."""
        
        # Generate the explanation
        result = self.process(prompt)
        
        logger.debug(f"Explained code in language: {language}")
        return result
    
    def review_code(self, code_content: str) -> str:
        """
        Review code for issues and improvements.
        
        Args:
            code_content (str): The code content to review.
            
        Returns:
            str: The review.
        """
        # Determine the language from the code content
        language = self._detect_language(code_content)
        
        # Create a specific prompt for code review
        prompt = f"""Perform a thorough code review of the following {language} code:

```{language}
{code_content}
```

In your review, focus on:
1. Code quality issues
2. Potential bugs or errors
3. Security vulnerabilities
4. Performance concerns
5. Readability and maintainability
6. Adherence to best practices and design patterns

For each issue, suggest a specific improvement or solution."""
        
        # Generate the review
        result = self.process(prompt)
        
        logger.debug(f"Reviewed code in language: {language}")
        return result
    
    def _detect_language(self, code_content: str) -> str:
        """
        Detect the programming language of the code content.
        
        Args:
            code_content (str): The code content to analyze.
            
        Returns:
            str: The detected programming language.
        """
        # This is a very basic detection; in a real implementation,
        # we would use more sophisticated heuristics or a dedicated library
        
        # Check for language indicators
        if re.search(r"import\s+(\w+)\s+from", code_content) or "def " in code_content and ":" in code_content:
            return "python"
        elif re.search(r"function\s+\w+\s*\(", code_content) and "{" in code_content:
            if ("import React" in code_content or "export " in code_content or "const " in code_content):
                return "javascript"
            elif "cout" in code_content or "cin" in code_content or "#include" in code_content:
                return "cpp"
            else:
                return "javascript"
        elif "public class" in code_content or "private class" in code_content:
            return "java"
        elif "<html" in code_content.lower():
            return "html"
        elif "@Controller" in code_content or "public class" in code_content and "void main" in code_content:
            return "java"
        elif "func " in code_content and "{" in code_content:
            return "go"
        elif "#include" in code_content and "int main" in code_content:
            return "c"
        elif "use strict" in code_content or "function" in code_content:
            return "javascript"
        else:
            # Default to a generic "code" if we can't detect the language
            return "code"
    
    def _extract_code_blocks(self, text: str, language: str) -> List[str]:
        """
        Extract code blocks from the text.
        
        Args:
            text (str): The text to extract code blocks from.
            language (str): The programming language.
            
        Returns:
            List[str]: The extracted code blocks.
        """
        # Look for code blocks with language identifier
        pattern = f"```(?:{language})?\n(.*?)```"
        blocks = re.findall(pattern, text, re.DOTALL)
        
        # If no blocks found with the language, try without language identifier
        if not blocks:
            pattern = "```\n(.*?)```"
            blocks = re.findall(pattern, text, re.DOTALL)
        
        # If still no blocks, try with single line code blocks
        if not blocks:
            pattern = "`(.*?)`"
            blocks = re.findall(pattern, text, re.DOTALL)
        
        return blocks

    def process(self, query: str) -> str:
        """
        Process a query and generate a response.
        
        Args:
            query (str): The user query to process.
            
        Returns:
            str: The generated response.
        """
        # Get context from memory
        context = self._get_context()
        
        # Process the query with the LLM
        response = self._generate_response(query, context)
        
        # Store the interaction in memory
        self._update_memory(query, response)
        
        # If learning is enabled, record the experience
        if self.learning_manager:
            try:
                # Determine experience type based on query content
                experience_type = self._determine_experience_type(query)
                
                # Record the experience
                experience_id = self.learning_manager.record_experience(
                    context=context,
                    query=query,
                    response=response,
                    experience_type=experience_type,
                    metadata={"agent_id": self.id, "agent_type": "coding"},
                    extract_knowledge=True
                )
                
                logger.debug(f"Recorded experience: {experience_id}")
            except Exception as e:
                logger.error(f"Error recording experience: {e}")
        
        return response
        
    def _determine_experience_type(self, query: str) -> str:
        """
        Determine the type of experience based on the query content.
        
        Args:
            query (str): The user query.
            
        Returns:
            str: The experience type.
        """
        from learning.types import ExperienceType
        
        query_lower = query.lower()
        
        # Check for common patterns in queries
        if any(keyword in query_lower for keyword in ["generate", "create", "write", "implement"]):
            return ExperienceType.CODE_GENERATION
        elif any(keyword in query_lower for keyword in ["explain", "understand", "describe", "how does"]):
            return ExperienceType.CODE_EXPLANATION
        elif any(keyword in query_lower for keyword in ["error", "bug", "fix", "issue", "problem", "doesn't work"]):
            return ExperienceType.ERROR_RESOLUTION
        else:
            # Default to CODE_EXPLANATION as a fallback
            return ExperienceType.CODE_EXPLANATION

    def _get_context(self) -> str:
        """Get the conversation context from memory."""
        # Get the last few messages for context
        messages = self.memory.get_messages_for_llm()
        if not messages:
            return ""
            
        # Format messages into a string
        context = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in messages[-5:]  # Use last 5 messages for context
        ])
        
        return context
        
    def _generate_response(self, query: str, context: str) -> str:
        """Generate a response using the LLM."""
        # Use a specialized prompt if needed based on query content
        # For now, just use the query as-is
        response = self.llm.generate(
            prompt=query,
            system_prompt=self.system_prompt,
            temperature=0.7
        )
        
        return response
        
    def _update_memory(self, query: str, response: str) -> None:
        """Update memory with the query and response."""
        # Add the user message
        self.memory.add_message("user", query)
        
        # Add the assistant message
        self.memory.add_message("assistant", response)


def get_coding_agent(
    llm: Optional[LocalLLM] = None,
    memory_id: Optional[str] = None,
    learning_manager = None
) -> CodingAgent:
    """
    Get a singleton coding agent instance.
    
    Args:
        llm (LocalLLM, optional): The LLM to use.
        memory_id (str, optional): The ID for the agent's memory.
        learning_manager: The learning manager instance.
        
    Returns:
        CodingAgent: The coding agent instance.
    """
    global _coding_agent
    if _coding_agent is None:
        _coding_agent = CodingAgent(
            llm=llm,
            memory_id=memory_id,
            learning_manager=learning_manager
        )
    return _coding_agent 