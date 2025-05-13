"""
Memory types and structures for the Autonomous Coding Agent.

This module defines the different types of memories and their structure.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
import time
from dataclasses import dataclass


class MemoryType(Enum):
    """Enumeration of memory types."""
    
    # Episodic memory - experiences and interactions
    CONVERSATION = "conversation"
    SESSION = "session"
    TASK = "task"
    EXPERIENCE = "experience"
    
    # Semantic memory - factual knowledge
    CONCEPT = "concept"
    FACT = "fact"
    DOCUMENTATION = "documentation"
    
    # Procedural memory - code and procedures
    CODE = "code"
    FUNCTION = "function"
    ALGORITHM = "algorithm"
    SOLUTION = "solution"
    PATTERN = "pattern"
    
    # Debug memory - errors and failures
    ERROR = "error"
    EXCEPTION = "exception"
    DEBUG = "debug"
    
    # Meta-memory - agent's own capabilities and reflections
    CAPABILITY = "capability"
    REFLECTION = "reflection"
    FEEDBACK = "feedback"
    
    # User-related memory
    PREFERENCE = "preference"
    INTERACTION_STYLE = "interaction_style"


@dataclass
class MemoryPriority:
    """
    Priority levels for memories, affecting retrieval and forgetting.
    
    Higher values indicate higher priority.
    """
    LOW = 1       # Background knowledge, easily replaceable
    MEDIUM = 2    # Standard memories
    HIGH = 3      # Important memories that should be preserved
    CRITICAL = 4  # Essential memories that should never be forgotten


@dataclass
class MemoryAccess:
    """
    Records of memory access to track usage patterns.
    """
    timestamp: float
    context: Optional[str] = None
    agent_id: Optional[str] = None
    relevance_score: Optional[float] = None
    
    @classmethod
    def record_access(cls, context: Optional[str] = None, 
                     agent_id: Optional[str] = None,
                     relevance_score: Optional[float] = None) -> 'MemoryAccess':
        """
        Create a new memory access record.
        
        Args:
            context (str, optional): The context of the access.
            agent_id (str, optional): The ID of the agent accessing the memory.
            relevance_score (float, optional): The relevance score of the memory to the access context.
            
        Returns:
            MemoryAccess: The memory access record.
        """
        return cls(
            timestamp=time.time(),
            context=context,
            agent_id=agent_id,
            relevance_score=relevance_score
        )


@dataclass
class MemoryAssociation:
    """
    Association between memories to form a semantic network.
    """
    target_id: str
    association_type: str
    strength: float
    created_at: float = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_id": self.target_id,
            "association_type": self.association_type,
            "strength": self.strength,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryAssociation':
        """Create from dictionary."""
        return cls(
            target_id=data["target_id"],
            association_type=data["association_type"],
            strength=data["strength"],
            created_at=data.get("created_at", time.time())
        )


@dataclass
class MemoryMetadata:
    """
    Extended metadata for memory items.
    """
    # Basic information
    item_type: str
    created_at: float
    source: Optional[str] = None
    
    # Memory management
    priority: int = MemoryPriority.MEDIUM
    last_accessed: Optional[float] = None
    access_count: int = 0
    access_history: List[MemoryAccess] = None
    
    # Associations with other memories
    associations: List[MemoryAssociation] = None
    
    # Custom metadata fields
    custom: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.access_history is None:
            self.access_history = []
        if self.associations is None:
            self.associations = []
        if self.custom is None:
            self.custom = {}
    
    def record_access(self, context: Optional[str] = None, 
                     agent_id: Optional[str] = None,
                     relevance_score: Optional[float] = None) -> None:
        """
        Record an access to this memory.
        
        Args:
            context (str, optional): The context of the access.
            agent_id (str, optional): The ID of the agent accessing the memory.
            relevance_score (float, optional): The relevance score of the memory to the access context.
        """
        access = MemoryAccess.record_access(context, agent_id, relevance_score)
        self.access_history.append(access)
        self.last_accessed = access.timestamp
        self.access_count += 1
    
    def add_association(self, target_id: str, association_type: str, strength: float) -> None:
        """
        Add an association to another memory.
        
        Args:
            target_id (str): The ID of the target memory.
            association_type (str): The type of association.
            strength (float): The strength of the association.
        """
        association = MemoryAssociation(
            target_id=target_id,
            association_type=association_type,
            strength=strength
        )
        self.associations.append(association)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_type": self.item_type,
            "created_at": self.created_at,
            "source": self.source,
            "priority": self.priority,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "access_history": [
                {
                    "timestamp": access.timestamp,
                    "context": access.context,
                    "agent_id": access.agent_id,
                    "relevance_score": access.relevance_score
                }
                for access in self.access_history
            ],
            "associations": [assoc.to_dict() for assoc in self.associations],
            "custom": self.custom
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryMetadata':
        """Create from dictionary."""
        metadata = cls(
            item_type=data["item_type"],
            created_at=data["created_at"],
            source=data.get("source"),
            priority=data.get("priority", MemoryPriority.MEDIUM),
            last_accessed=data.get("last_accessed"),
            access_count=data.get("access_count", 0),
            custom=data.get("custom", {})
        )
        
        # Add access history
        if "access_history" in data:
            metadata.access_history = [
                MemoryAccess(
                    timestamp=access["timestamp"],
                    context=access.get("context"),
                    agent_id=access.get("agent_id"),
                    relevance_score=access.get("relevance_score")
                )
                for access in data["access_history"]
            ]
        
        # Add associations
        if "associations" in data:
            metadata.associations = [
                MemoryAssociation.from_dict(assoc)
                for assoc in data["associations"]
            ]
        
        return metadata


@dataclass
class ExtendedMemoryItem:
    """
    Extended memory item with rich metadata and associations.
    """
    content: str
    metadata: MemoryMetadata
    item_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtendedMemoryItem':
        """Create from dictionary."""
        return cls(
            item_id=data.get("item_id"),
            content=data["content"],
            metadata=MemoryMetadata.from_dict(data["metadata"]),
            embedding=data.get("embedding")
        )
    
    def record_access(self, context: Optional[str] = None, 
                     agent_id: Optional[str] = None,
                     relevance_score: Optional[float] = None) -> None:
        """
        Record an access to this memory.
        
        Args:
            context (str, optional): The context of the access.
            agent_id (str, optional): The ID of the agent accessing the memory.
            relevance_score (float, optional): The relevance score of the memory to the access context.
        """
        self.metadata.record_access(context, agent_id, relevance_score)
    
    def add_association(self, target_id: str, association_type: str, strength: float) -> None:
        """
        Add an association to another memory.
        
        Args:
            target_id (str): The ID of the target memory.
            association_type (str): The type of association.
            strength (float): The strength of the association.
        """
        self.metadata.add_association(target_id, association_type, strength) 