"""
Base agent implementation for the Autonomous Coding Agent.

This module provides a base agent class that other agent implementations can extend.
"""

import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from loguru import logger

from models.llm_interface import LocalLLM, get_llm
from memory.short_term import ShortTermMemory, get_memory
from memory.manager import MemoryManager, get_memory_manager


class BaseAgent:
    """
    Base agent class with core functionalities.
    """
    
    def __init__(
        self,
        name: str,
        llm: Optional[LocalLLM] = None,
        memory_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Callable]] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            name (str): The name of the agent.
            llm (LocalLLM, optional): The LLM to use.
            memory_id (str, optional): The ID for the agent's memory.
            system_prompt (str, optional): The system prompt to use.
            tools (List[Callable], optional): A list of tools available to the agent.
        """
        self.name = name
        self.id = str(uuid.uuid4())
        self.llm = llm or get_llm()
        
        # Use the memory_id for both short-term memory and memory manager
        self.memory_id = memory_id or f"agent_{self.id}"
        self.memory = get_memory(self.memory_id)
        
        # Initialize the memory manager for optimized context retrieval
        self.memory_manager = get_memory_manager(self.memory_id)
        
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.tools = tools or []
        
        # Add the system prompt to memory if provided
        if self.system_prompt:
            self.memory.add_system_message(self.system_prompt)
        
        logger.info(f"Initialized agent: {self.name} ({self.id})")
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the agent."""
        return f"""You are {self.name}, an AI assistant that helps with coding tasks.
You are helpful, concise, and provide accurate information.
If you don't know something, you admit it instead of making things up."""
    
    def process(self, user_input: str) -> str:
        """
        Process a user input and generate a response using optimized context.
        
        Args:
            user_input (str): The user input to process.
            
        Returns:
            str: The agent's response.
        """
        logger.debug(f"Processing input: {user_input[:50]}...")
        
        # Add the user input to both memory systems
        self.memory.add_user_message(user_input)
        self.memory_manager.add_message("user", user_input)
        
        # Get optimized context for the query
        start_time = time.time()
        optimized_context = self.memory_manager.refresh_context(
            query=user_input,
            system_prompt=self.system_prompt
        )
        context_time = time.time() - start_time
        logger.debug(f"Retrieved optimized context in {context_time:.2f} seconds")
        
        # Generate response using the optimized context
        start_time = time.time()
        response = self._generate_response_with_context(user_input, optimized_context)
        generation_time = time.time() - start_time
        
        # Add the response to both memory systems
        self.memory.add_assistant_message(response)
        self.memory_manager.add_message("assistant", response)
        
        # Log metrics
        metrics = optimized_context.get("metrics", {})
        logger.info(f"Generated response in {generation_time:.2f} seconds")
        logger.info(f"Context utilization: {metrics.get('utilization', 0):.1%}")
        logger.info(f"Query type: {metrics.get('query_type', 'unknown')}, Complexity: {metrics.get('complexity', 'unknown')}")
        
        return response
    
    def _generate_response_with_context(self, query: str, optimized_context: Dict[str, Any]) -> str:
        """
        Generate a response using the optimized context.
        
        Args:
            query (str): The user query.
            optimized_context (Dict[str, Any]): The optimized context from memory manager.
            
        Returns:
            str: The generated response.
        """
        try:
            # Use the new generate_with_context method
            return self.llm.generate_with_context(
                query=query,
                context=optimized_context,
                interaction_type="agent_response"
            )
        except Exception as e:
            logger.exception(f"Error generating response with context: {e}")
            
            # Fallback to standard generation if context-based generation fails
            logger.info("Falling back to standard response generation")
            return self._generate_response_fallback(query)
    
    def _generate_response_fallback(self, query: str) -> str:
        """
        Fallback method to generate a response using the standard approach.
        
        Args:
            query (str): The user query.
            
        Returns:
            str: The generated response.
        """
        try:
            # Get the conversation history for context
            messages = self.memory.get_messages_for_llm()
            
            # Try using direct llm.generate method
            for msg in messages:
                if msg['role'] == 'system':
                    system_content = msg['content']
                    break
            else:
                system_content = self.system_prompt
            
            return self.llm.generate(
                prompt=query,
                system_prompt=system_content,
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens
            )
            
        except Exception as e:
            logger.exception(f"Error in fallback response generation: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Legacy method to generate a response using the LLM.
        This is kept for backwards compatibility.
        
        Args:
            messages (List[Dict[str, str]]): The messages to send to the LLM.
            
        Returns:
            str: The generated response.
        """
        try:
            # Print the message list for debugging
            logger.debug(f"Using legacy response generation with {len(messages)} messages")
            
            # Try using direct llm.generate method first
            if len(messages) > 0:
                last_message = messages[-1]['content']
                system_content = None
                
                # Find system message if exists
                for msg in messages:
                    if msg['role'] == 'system':
                        system_content = msg['content']
                        break
                
                return self.llm.generate(
                    prompt=last_message,
                    system_prompt=system_content,
                    temperature=self.llm.temperature,
                    max_tokens=self.llm.max_tokens
                )
            
            # Fallback to chat completion API
            logger.debug("Falling back to chat completion API")
            response = self.llm.client.chat.completions.create(
                model=self.llm.model_name,
                messages=messages,
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens
            )
            
            # Log and return the response
            content = response.choices[0].message.content
            return content
            
        except Exception as e:
            logger.exception(f"Error generating response: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def reflect(self) -> str:
        """
        Reflect on past interactions and improve the agent's performance.
        
        Returns:
            str: The result of the reflection.
        """
        # Get recent messages for reflection
        recent_messages = self.memory.get_last_k_messages(10)
        recent_interactions = "\n".join([
            f"{msg.role.upper()}: {msg.content}" 
            for msg in recent_messages
            if msg.role != "system"
        ])
        
        # Create a reflection prompt
        reflection_prompt = f"""
Review the following recent interactions:

{recent_interactions}

Based on these interactions, reflect on your performance:
1. What did you do well?
2. What could you improve?
3. What knowledge or skills would help you provide better assistance?
4. How can you better understand the user's needs?

Provide a concise self-evaluation and plan for improvement.
"""
        
        # Generate a reflection using optimized context
        optimized_context = self.memory_manager.refresh_context(
            query="Reflect on my performance and suggest improvements",
            system_prompt="You are performing a self-evaluation to improve your capabilities."
        )
        
        reflection = self.llm.generate_with_context(
            query=reflection_prompt,
            context=optimized_context
        )
        
        logger.info(f"Agent reflection generated: {reflection[:100]}...")
        return reflection
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save the agent's current state.
        
        Returns:
            Dict[str, Any]: The agent's state.
        """
        # In a more complex implementation, we would save more state information
        return {
            "id": self.id,
            "name": self.name,
            "system_prompt": self.system_prompt,
            "memory": self.memory.get_messages_as_dict(),
            "memory_id": self.memory_id
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load the agent's state.
        
        Args:
            state (Dict[str, Any]): The state to load.
        """
        self.id = state.get("id", self.id)
        self.name = state.get("name", self.name)
        self.system_prompt = state.get("system_prompt", self.system_prompt)
        self.memory_id = state.get("memory_id", self.memory_id)
        
        # Reinitialize memory systems with the loaded ID
        if self.memory_id != state.get("memory_id"):
            self.memory = get_memory(self.memory_id)
            self.memory_manager = get_memory_manager(self.memory_id)
        
        logger.info(f"Loaded agent state: {self.name} ({self.id})")
    
    def clear_memory(self) -> None:
        """Clear the agent's memory."""
        self.memory.clear()
        self.memory_manager.clear_short_term()
        
        # Re-add the system prompt
        if self.system_prompt:
            self.memory.add_system_message(self.system_prompt)
            self.memory_manager.add_message("system", self.system_prompt)
        
        logger.info(f"Cleared memory for agent: {self.name} ({self.id})")


def get_agent(
    name: str,
    llm: Optional[LocalLLM] = None,
    memory_id: Optional[str] = None,
    system_prompt: Optional[str] = None
) -> BaseAgent:
    """
    Create a base agent instance.
    
    Args:
        name (str): The name of the agent.
        llm (LocalLLM, optional): The LLM to use.
        memory_id (str, optional): The ID for the agent's memory.
        system_prompt (str, optional): The system prompt to use.
        
    Returns:
        BaseAgent: The agent instance.
    """
    return BaseAgent(
        name=name,
        llm=llm,
        memory_id=memory_id,
        system_prompt=system_prompt
    ) 