"""
Memory manager for the Autonomous Coding Agent.

This module provides a unified interface for accessing and managing different memory systems.
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from loguru import logger

from .short_term import ShortTermMemory, Message, get_memory as get_short_term_memory
from .long_term import LongTermMemory, MemoryItem, get_long_term_memory
from .embeddings import EmbeddingGenerator, CodeChunker, EmbeddingUtils
from .memory_optimizer import MemoryOptimizer

from config import MODEL_CONFIG


class MemoryManager:
    """
    Memory manager that coordinates between short-term and long-term memory systems.
    """
    
    def __init__(
        self,
        agent_id: str = "default",
        short_term_id: Optional[str] = None,
        long_term_collection: str = "code_knowledge",
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the memory manager.
        
        Args:
            agent_id (str): The ID of the agent using this memory manager.
            short_term_id (str, optional): The ID for short-term memory.
            long_term_collection (str): The collection name for long-term memory.
            embedding_model (str, optional): The name of the embedding model to use.
        """
        self.agent_id = agent_id
        
        # Initialize memory systems
        self.short_term = get_short_term_memory(short_term_id or f"{agent_id}_conversation")
        self.long_term = get_long_term_memory(long_term_collection)
        
        # Initialize embedding utilities
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model or "all-MiniLM-L6-v2")
        self.code_chunker = CodeChunker()
        
        # Working memory (temporary items not persisted in vector store)
        self.working_memory = {}
        
        # Memory statistics
        self.stats = {
            "short_term_size": 0,
            "long_term_size": 0,
            "retrieval_count": 0,
            "storage_count": 0,
            "working_memory_size": 0
        }
        
        # Initialize the memory optimizer
        model_context_size = MODEL_CONFIG.get("max_tokens", 4096)
        self.memory_optimizer = MemoryOptimizer(
            model_context_size=model_context_size,
            embedding_generator=self.embedding_generator
        )
        
        # Update stats
        self._update_stats()
        
        logger.info(f"Initialized MemoryManager for agent: {agent_id}")
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to short-term memory.
        
        Args:
            role (str): The role of the message sender.
            content (str): The content of the message.
        """
        self.short_term.add_message(role, content)
        self._update_stats()
    
    def get_conversation_history(self, k: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get the conversation history from short-term memory.
        
        Args:
            k (int, optional): The number of messages to retrieve.
            
        Returns:
            List[Dict[str, str]]: The conversation history.
        """
        messages = self.short_term.get_messages()
        if k is not None:
            messages = messages[-k:]
        
        return [msg.to_dict() for msg in messages]
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """
        Get all messages in the format expected by the LLM.
        
        Returns:
            List[Dict[str, str]]: The messages in LLM format.
        """
        return self.short_term.get_messages_for_llm()
    
    def add_to_long_term(self, content: str, item_type: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add an item to long-term memory.
        
        Args:
            content (str): The content to store.
            item_type (str): The type of content.
            metadata (Dict[str, Any], optional): Additional metadata.
            
        Returns:
            str: The ID of the stored item.
        """
        # Add agent_id to metadata
        meta = metadata or {}
        meta["agent_id"] = self.agent_id
        
        # Add item to long-term memory
        item_id = self.long_term.add_item(content, item_type=item_type, metadata=meta)
        
        # Update stats
        self.stats["storage_count"] += 1
        self._update_stats()
        
        return item_id
    
    def add_code_to_long_term(self, code: str, language: str, 
                            metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add code to long-term memory with semantic chunking.
        
        Args:
            code (str): The code to store.
            language (str): The programming language.
            metadata (Dict[str, Any], optional): Additional metadata.
            
        Returns:
            List[str]: The IDs of the stored chunks.
        """
        # Add agent_id and language to metadata
        meta = metadata or {}
        meta["agent_id"] = self.agent_id
        meta["language"] = language
        
        # Chunk the code
        chunks = self.code_chunker.chunk(code, language=language)
        
        # Store each chunk
        chunk_ids = []
        for chunk in chunks:
            # Add chunk metadata to the item metadata
            chunk_meta = {**meta, **chunk["metadata"]}
            
            # Add to long-term memory
            item_id = self.long_term.add_item(
                chunk["content"], 
                item_type="code", 
                metadata=chunk_meta
            )
            chunk_ids.append(item_id)
        
        # Update stats
        self.stats["storage_count"] += len(chunks)
        self._update_stats()
        
        return chunk_ids
    
    def retrieve_relevant(self, query: str, item_type: Optional[str] = None,
                        n_results: int = 5) -> List[MemoryItem]:
        """
        Retrieve relevant items from long-term memory.
        
        Args:
            query (str): The query text.
            item_type (str, optional): The type of items to retrieve.
            n_results (int): The number of results to retrieve.
            
        Returns:
            List[MemoryItem]: The retrieved memory items.
        """
        # Query long-term memory
        items = self.long_term.query(query, item_type=item_type, n_results=n_results)
        
        # Update stats
        self.stats["retrieval_count"] += 1
        
        return items
    
    def retrieve_code_examples(self, query: str, language: Optional[str] = None, 
                             n_results: int = 5) -> List[MemoryItem]:
        """
        Retrieve relevant code examples from long-term memory.
        
        Args:
            query (str): The query text.
            language (str, optional): The programming language.
            n_results (int): The number of results to retrieve.
            
        Returns:
            List[MemoryItem]: The retrieved code examples.
        """
        # Prepare filters
        filters = {"item_type": "code"}
        if language:
            filters["language"] = language
        
        # Process query for better code retrieval
        processed_query = self.embedding_generator._preprocess_query(query)
        
        # Query long-term memory
        items = self.long_term.query(
            processed_query, 
            filters=filters, 
            n_results=n_results
        )
        
        # Update stats
        self.stats["retrieval_count"] += 1
        
        return items
    
    def set_working_memory(self, key: str, value: Any) -> None:
        """
        Store an item in working memory.
        
        Args:
            key (str): The key to store the item under.
            value (Any): The value to store.
        """
        self.working_memory[key] = {
            "value": value,
            "timestamp": time.time()
        }
        self.stats["working_memory_size"] = len(self.working_memory)
    
    def get_working_memory(self, key: str, default: Any = None) -> Any:
        """
        Retrieve an item from working memory.
        
        Args:
            key (str): The key to retrieve.
            default (Any, optional): The default value if key is not found.
            
        Returns:
            Any: The retrieved value or default.
        """
        item = self.working_memory.get(key)
        if item is not None:
            return item["value"]
        return default
    
    def forget_working_memory(self, key: str) -> None:
        """
        Remove an item from working memory.
        
        Args:
            key (str): The key to remove.
        """
        if key in self.working_memory:
            del self.working_memory[key]
            self.stats["working_memory_size"] = len(self.working_memory)
    
    def clear_working_memory(self) -> None:
        """Clear all items from working memory."""
        self.working_memory.clear()
        self.stats["working_memory_size"] = 0
    
    def refresh_context(self, query: str, context_size: int = 10, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Refresh the agent's context with relevant information from long-term memory,
        using dynamic token allocation and optimized context composition.
        
        Args:
            query (str): The current query or task.
            context_size (int): The number of relevant items to include.
            system_prompt (str, optional): The system prompt to include in context optimization.
            
        Returns:
            Dict[str, Any]: The refreshed context.
        """
        # Get the conversation history
        messages = self.short_term.get_messages()
        
        # Get relevant items from long-term memory with increased context_size
        # We'll retrieve more than we need and let the optimizer choose the best ones
        relevant_items = self.long_term.query(query, n_results=context_size * 2)
        
        # Use a default system prompt if none provided
        sys_prompt = system_prompt or "You are a helpful AI assistant focusing on coding tasks."
        
        # Use the memory optimizer to build an optimized context
        optimized_context = self.memory_optimizer.build_optimized_context(
            query=query,
            system_prompt=sys_prompt,
            messages=messages,
            knowledge_items=relevant_items,
            working_memory={k: v["value"] for k, v in self.working_memory.items()}
        )
        
        # Create a context dictionary with the same structure as before for backwards compatibility
        context = {
            "recent_messages": [msg.to_dict() for msg in optimized_context["messages"]],
            "relevant_knowledge": [item.to_dict() for item in optimized_context["knowledge_items"]],
            "working_memory": optimized_context["working_memory"],
            "system_prompt": optimized_context["system_prompt"],
            "metrics": optimized_context["metrics"]
        }
        
        # Update stats
        self.stats["retrieval_count"] += 1
        
        # Log context utilization metrics
        metrics = optimized_context["metrics"]
        logger.info(f"Context utilization: {metrics['utilization']:.1%} ({metrics['total_tokens']} tokens)")
        logger.info(f"Query classified as {metrics['query_type']} complexity: {metrics['complexity']}")
        
        return context
    
    def store_conversation_summary(self) -> str:
        """
        Create a summary of the current conversation and store it in long-term memory.
        
        Returns:
            str: The ID of the stored summary.
        """
        # Get all messages
        messages = self.short_term.get_messages()
        
        # Create a summary (concatenate for now, in a real system this would use the LLM)
        summary = f"Conversation ({len(messages)} messages) - {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # Add the first 2 messages and last 3 messages to give context
        if messages:
            summary += "\nStart of conversation:\n"
            for msg in messages[:min(2, len(messages))]:
                summary += f"[{msg.role}]: {msg.content[:100]}...\n"
                
            summary += "\nEnd of conversation:\n"
            for msg in messages[-min(3, len(messages)):]:
                summary += f"[{msg.role}]: {msg.content[:100]}...\n"
        
        # Store in long-term memory
        item_id = self.long_term.add_item(
            summary,
            item_type="conversation_summary",
            metadata={
                "agent_id": self.agent_id,
                "message_count": len(messages),
                "timestamp": time.time()
            }
        )
        
        # Update stats
        self.stats["storage_count"] += 1
        self._update_stats()
        
        return item_id
    
    def _update_stats(self) -> None:
        """Update memory statistics."""
        self.stats["short_term_size"] = len(self.short_term.get_messages())
        collection_stats = self.long_term.get_collection_stats()
        self.stats["long_term_size"] = collection_stats.get("item_count", 0)
        self.stats["working_memory_size"] = len(self.working_memory)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dict[str, Any]: Memory statistics.
        """
        self._update_stats()
        return self.stats
    
    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        self.short_term.clear()
        self._update_stats()


def get_memory_manager(agent_id: str = "default") -> MemoryManager:
    """
    Get a memory manager instance.
    
    Args:
        agent_id (str): The ID of the agent using the memory manager.
        
    Returns:
        MemoryManager: The memory manager instance.
    """
    return MemoryManager(agent_id=agent_id) 