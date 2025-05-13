"""
Orchestrator agent implementation for the Autonomous Coding Agent.

This module provides an orchestrator agent that coordinates between different agent types.
"""

import json
import re
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from loguru import logger

from models.llm_interface import LocalLLM, get_llm
from agents.base_agent import BaseAgent
from agents.coding_agent import CodingAgent, get_coding_agent

# Singleton instance
_orchestrator = None

class AgentOrchestrator:
    """
    Orchestrator agent that coordinates between different agent types.
    """
    
    def __init__(
        self,
        llm: Optional[LocalLLM] = None,
        memory_id: Optional[str] = None,
        learning_manager = None
    ):
        """
        Initialize the orchestrator agent.
        
        Args:
            llm (LocalLLM, optional): The LLM to use.
            memory_id (str, optional): The ID for the orchestrator's memory.
            learning_manager: The learning manager instance.
        """
        self.llm = llm or get_llm()
        self.memory_id = memory_id or "orchestrator"
        self.learning_manager = learning_manager
        
        # Initialize specialized agents
        self.coding_agent = get_coding_agent(
            llm=self.llm, 
            memory_id=f"{self.memory_id}_coding",
            learning_manager=self.learning_manager
        )
        
        # Later we can add more specialized agents here
        
        logger.info(f"Initialized AgentOrchestrator with memory_id: {self.memory_id}")
    
    def process_input(self, user_input: str) -> str:
        """
        Process user input and route to the appropriate agent.
        
        Args:
            user_input (str): The user input to process.
            
        Returns:
            str: The response from the appropriate agent.
        """
        logger.debug(f"Processing input: {user_input[:50]}...")
        
        # Determine which agent should handle the request
        agent_type, confidence = self._determine_agent_type(user_input)
        
        # Log the routing decision
        logger.debug(f"Routing to agent: {agent_type} (confidence: {confidence:.2f})")
        
        # Route to the appropriate agent
        if agent_type == "coding" and confidence >= 0.6:
            return self.coding_agent.process(user_input)
        else:
            # Default to coding agent for now
            # In a more complex implementation, we would have more agent types
            return self.coding_agent.process(user_input)
    
    def _determine_agent_type(self, user_input: str) -> Tuple[str, float]:
        """
        Determine which agent type should handle the request.
        
        Args:
            user_input (str): The user input to process.
            
        Returns:
            Tuple[str, float]: The agent type and confidence score.
        """
        # Create a prompt to determine the agent type
        prompt = f"""Determine the most appropriate agent type to handle this user request.
You must respond with a valid JSON object containing ONLY two fields:
1. "agent_type": Either "coding" or "general"
2. "confidence": A number between 0 and 1 indicating your confidence

User request: {user_input}

Valid response example:
{{"agent_type": "coding", "confidence": 0.9}}

DO NOT include any other text, explanation, or formatting in your response.
ONLY return the JSON object.
"""
        
        try:
            # Generate a response with lower temperature for more deterministic results
            response = self.llm.generate(
                prompt=prompt, 
                system_prompt="You are an agent router. Your ONLY job is to return a valid JSON object.",
                temperature=0.1,
                max_retries=2
            )
            
            # Try to find and extract JSON from the response
            json_pattern = r"\{.*?\}"
            json_matches = re.findall(json_pattern, response, re.DOTALL)
            
            if json_matches:
                # Try each match until we find a valid JSON
                for json_str in json_matches:
                    try:
                        parsed = json.loads(json_str)
                        if "agent_type" in parsed and "confidence" in parsed:
                            agent_type = parsed["agent_type"]
                            # Validate agent type
                            if agent_type not in ["coding", "general"]:
                                agent_type = "coding"  # Default to coding
                            
                            confidence = float(parsed["confidence"])
                            # Ensure confidence is within bounds
                            confidence = max(0.0, min(1.0, confidence))
                            
                            return agent_type, confidence
                    except (json.JSONDecodeError, ValueError):
                        continue
            
            # If we've reached here, we couldn't parse valid JSON
            logger.warning(f"Failed to parse agent type response: {response[:100]}...")
            
            # Use a heuristic approach as fallback
            coding_keywords = ["code", "program", "function", "class", "variable", 
                              "python", "javascript", "java", "c++", "algorithm",
                              "error", "bug", "debug", "api", "library"]
            
            # Simple keyword matching
            user_input_lower = user_input.lower()
            matches = sum(1 for keyword in coding_keywords if keyword in user_input_lower)
            
            if matches > 0:
                confidence = min(0.5 + (matches * 0.1), 0.9)  # Scale confidence with matches
                return "coding", confidence
            else:
                return "coding", 0.5  # Default to coding with moderate confidence
                
        except Exception as e:
            logger.exception(f"Error determining agent type: {e}")
            # Default to coding agent if an error occurs
            return "coding", 0.5
    
    def generate_code(self, requirements: str, language: str) -> str:
        """
        Generate code based on requirements.
        
        Args:
            requirements (str): The requirements for the code.
            language (str): The programming language to use.
            
        Returns:
            str: The generated code.
        """
        return self.coding_agent.generate_code(requirements, language)
    
    def explain_code(self, code_content: str) -> str:
        """
        Explain a code snippet.
        
        Args:
            code_content (str): The code content to explain.
            
        Returns:
            str: The explanation.
        """
        return self.coding_agent.explain_code(code_content)
    
    def review_code(self, code_content: str) -> str:
        """
        Review code for issues and improvements.
        
        Args:
            code_content (str): The code content to review.
            
        Returns:
            str: The review.
        """
        return self.coding_agent.review_code(code_content)
    
    def reflect(self) -> Dict[str, str]:
        """
        Perform reflection for all agents.
        
        Returns:
            Dict[str, str]: The reflection results for each agent.
        """
        results = {
            "coding_agent": self.coding_agent.reflect()
            # Add more agents here as we implement them
        }
        
        logger.info("Performed reflection for all agents")
        return results
    
    def clear_memory(self) -> None:
        """
        Clear memory for all agents.
        """
        self.coding_agent.clear_memory()
        # Add more agents here as we implement them
        
        logger.info("Cleared memory for all agents")
    
    def clone_agent(self, agent_type: str = "coding") -> BaseAgent:
        """
        Create a clone of an agent.
        
        Args:
            agent_type (str): The type of agent to clone.
            
        Returns:
            BaseAgent: The cloned agent.
        """
        if agent_type == "coding":
            # Create a new coding agent with the same LLM but different memory
            cloned_agent = get_coding_agent(
                llm=self.llm,
                memory_id=f"{self.memory_id}_coding_clone_{len(self.get_clones())}"
            )
            
            logger.info(f"Cloned {agent_type} agent")
            return cloned_agent
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def get_clones(self) -> List[str]:
        """
        Get a list of all cloned agent memory IDs.
        
        Returns:
            List[str]: The memory IDs of all cloned agents.
        """
        # In a real implementation, we would track this properly
        # For now, we just return an empty list
        return []


def get_orchestrator(
    llm: Optional[LocalLLM] = None,
    memory_id: Optional[str] = None,
    learning_manager = None
) -> AgentOrchestrator:
    """
    Get or create the singleton instance of the orchestrator agent.
    
    Args:
        llm (LocalLLM, optional): The LLM to use.
        memory_id (str, optional): The ID for the orchestrator's memory.
        learning_manager: The learning manager instance.
        
    Returns:
        AgentOrchestrator: The orchestrator agent instance.
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator(
            llm=llm, 
            memory_id=memory_id,
            learning_manager=learning_manager
        )
    return _orchestrator 