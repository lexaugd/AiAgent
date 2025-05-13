"""
Short-term memory implementation for the Autonomous Coding Agent.

This module provides a short-term memory implementation using conversation buffers.
"""

import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger

import sys
sys.path.append("../..")
from config import MEMORY_CONFIG, CONVERSATION_HISTORY_DIR


class Message:
    """Class to represent a message in the conversation."""
    
    def __init__(self, role: str, content: str, timestamp: Optional[float] = None):
        """
        Initialize a message.
        
        Args:
            role (str): The role of the message sender (user, assistant, system).
            content (str): The content of the message.
            timestamp (float, optional): The timestamp of the message.
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a message from a dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time())
        )
    
    def to_langchain_format(self) -> Dict[str, str]:
        """Convert the message to LangChain message format."""
        return {
            "role": self.role,
            "content": self.content
        }
    
    def to_openai_format(self) -> Dict[str, str]:
        """Convert the message to OpenAI message format."""
        return {
            "role": self.role,
            "content": self.content
        }


class ShortTermMemory:
    """
    Short-term memory implementation using conversation buffers.
    """
    
    def __init__(
        self,
        conversation_id: str = "default",
        max_token_limit: Optional[int] = None
    ):
        """
        Initialize the short-term memory.
        
        Args:
            conversation_id (str): The ID of the conversation.
            max_token_limit (int, optional): The maximum number of tokens to store.
        """
        self.conversation_id = conversation_id
        self.max_token_limit = max_token_limit or MEMORY_CONFIG["short_term"]["max_token_limit"]
        self.messages: List[Message] = []
        self.current_token_count = 0
        
        # Load existing conversation if available
        self._load_conversation()
        
        logger.debug(f"Initialized ShortTermMemory with ID: {conversation_id}")
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the memory.
        
        Args:
            role (str): The role of the message sender (user, assistant, system).
            content (str): The content of the message.
        """
        message = Message(role=role, content=content)
        
        # Add the message to the buffer
        self.messages.append(message)
        
        # Approximate token count (very basic approximation)
        # In a real implementation, use a proper tokenizer
        approx_tokens = len(content.split())
        self.current_token_count += approx_tokens
        
        # If we exceed the token limit, remove old messages until we're under the limit
        if self.current_token_count > self.max_token_limit:
            self._trim_to_token_limit()
        
        # Save the conversation
        self._save_conversation()
        
        logger.debug(f"Added message (role: {role}, approx tokens: {approx_tokens})")
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the memory."""
        self.add_message("user", content)
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the memory."""
        self.add_message("assistant", content)
    
    def add_system_message(self, content: str) -> None:
        """Add a system message to the memory."""
        self.add_message("system", content)
    
    def get_messages(self) -> List[Message]:
        """Get all messages in the memory."""
        return self.messages
    
    def get_last_k_messages(self, k: int) -> List[Message]:
        """Get the last k messages in the memory."""
        return self.messages[-k:] if k < len(self.messages) else self.messages
    
    def get_messages_as_dict(self) -> List[Dict[str, Any]]:
        """Get all messages in the memory as dictionaries."""
        return [message.to_dict() for message in self.messages]
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Get all messages in the format expected by the LLM."""
        return [message.to_openai_format() for message in self.messages]
    
    def clear(self) -> None:
        """Clear all messages from the memory."""
        self.messages = []
        self.current_token_count = 0
        self._save_conversation()
        logger.debug("Cleared short-term memory")
    
    def _trim_to_token_limit(self) -> None:
        """Remove old messages until we're under the token limit."""
        while self.current_token_count > self.max_token_limit and len(self.messages) > 0:
            # Remove the oldest message (excluding system messages if possible)
            non_system_indices = [
                i for i, message in enumerate(self.messages)
                if message.role != "system"
            ]
            
            if non_system_indices:
                idx_to_remove = non_system_indices[0]
            else:
                idx_to_remove = 0
            
            removed_message = self.messages.pop(idx_to_remove)
            approx_tokens = len(removed_message.content.split())
            self.current_token_count -= approx_tokens
            
            logger.debug(f"Trimmed message (role: {removed_message.role}, approx tokens: {approx_tokens})")
    
    def _get_conversation_path(self) -> Path:
        """Get the path to the conversation file."""
        return Path(CONVERSATION_HISTORY_DIR) / f"{self.conversation_id}.json"
    
    def _save_conversation(self) -> None:
        """Save the conversation to a file."""
        try:
            conversation_path = self._get_conversation_path()
            with open(conversation_path, "w") as f:
                json.dump({
                    "conversation_id": self.conversation_id,
                    "messages": self.get_messages_as_dict(),
                    "token_count": self.current_token_count
                }, f, indent=2)
                
            logger.debug(f"Saved conversation to {conversation_path}")
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
    
    def _load_conversation(self) -> None:
        """Load the conversation from a file if it exists."""
        conversation_path = self._get_conversation_path()
        
        if conversation_path.exists():
            try:
                with open(conversation_path, "r") as f:
                    data = json.load(f)
                    
                self.conversation_id = data.get("conversation_id", self.conversation_id)
                self.messages = [Message.from_dict(msg) for msg in data.get("messages", [])]
                self.current_token_count = data.get("token_count", 0)
                
                logger.debug(f"Loaded conversation from {conversation_path} ({len(self.messages)} messages)")
            except Exception as e:
                logger.error(f"Failed to load conversation: {e}")


def get_memory(conversation_id: str = "default") -> ShortTermMemory:
    """
    Get a short-term memory instance.
    
    Args:
        conversation_id (str): The ID of the conversation.
        
    Returns:
        ShortTermMemory: The short-term memory instance.
    """
    return ShortTermMemory(conversation_id=conversation_id) 