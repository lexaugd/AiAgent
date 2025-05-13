"""
Type definitions for the learning system.

This module defines the core data types used in the learning components.
"""

import time
import json
import enum
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from loguru import logger

# Define enums for categorizing different types
class ExperienceType(enum.Enum):
    """Types of agent experiences."""
    CODE_GENERATION = "code_generation"
    CODE_EXPLANATION = "code_explanation"
    ERROR_RESOLUTION = "error_resolution"
    QUESTION_ANSWERING = "question_answering"
    TASK_PLANNING = "task_planning"
    UNKNOWN = "unknown"

class FeedbackType(enum.Enum):
    """Types of user feedback."""
    EXPLICIT_RATING = "explicit_rating"  # User provided a numerical rating
    CORRECTION = "correction"  # User corrected the agent's response
    CONFIRMATION = "confirmation"  # User confirmed the agent's response
    REJECTION = "rejection"  # User rejected the agent's response
    CLARIFICATION = "clarification"  # User asked for clarification
    UNKNOWN = "unknown"

class KnowledgeType(enum.Enum):
    """Types of extracted knowledge."""
    CODE_SNIPPET = "code_snippet"
    CODE_PATTERN = "code_pattern"
    FACT = "fact"
    CONCEPT = "concept"
    ERROR_SOLUTION = "error_solution"
    TOOL_USAGE = "tool_usage"
    UNKNOWN = "unknown"

class Experience:
    """Class to represent an agent experience."""
    
    def __init__(
        self,
        context: str,
        query: str,
        response: str,
        experience_type: Union[ExperienceType, str],
        metadata: Optional[Dict[str, Any]] = None,
        experience_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        outcome: Optional[str] = None,
        feedback: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an experience.
        
        Args:
            context (str): The context in which the experience occurred
            query (str): The user query that triggered the experience
            response (str): The agent's response
            experience_type (Union[ExperienceType, str]): The type of experience
            metadata (Dict[str, Any], optional): Additional metadata
            experience_id (str, optional): Unique ID for the experience
            timestamp (float, optional): The time when the experience occurred
            outcome (str, optional): The outcome of the experience (success/failure)
            feedback (Dict[str, Any], optional): User feedback on the experience
        """
        self.context = context
        self.query = query
        self.response = response
        
        # Convert string to enum if necessary
        if isinstance(experience_type, str):
            try:
                self.experience_type = ExperienceType[experience_type.upper()]
            except KeyError:
                self.experience_type = ExperienceType.UNKNOWN
        else:
            self.experience_type = experience_type
            
        self.metadata = metadata or {}
        self.experience_id = experience_id or f"exp_{int(time.time())}_{id(self)}"
        self.timestamp = timestamp or time.time()
        self.outcome = outcome or "unknown"
        self.feedback = feedback or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the experience to a dictionary."""
        return {
            "experience_id": self.experience_id,
            "context": self.context,
            "query": self.query,
            "response": self.response,
            "experience_type": self.experience_type.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "outcome": self.outcome,
            "feedback": self.feedback
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """Create an experience from a dictionary."""
        return cls(
            context=data["context"],
            query=data["query"],
            response=data["response"],
            experience_type=data["experience_type"],
            metadata=data.get("metadata", {}),
            experience_id=data.get("experience_id"),
            timestamp=data.get("timestamp"),
            outcome=data.get("outcome"),
            feedback=data.get("feedback", {})
        )


class Feedback:
    """Class to represent user feedback on agent responses."""
    
    def __init__(
        self,
        content: str,
        feedback_type: Union[FeedbackType, str],
        rating: Optional[float] = None,
        target_response_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        feedback_id: Optional[str] = None,
        timestamp: Optional[float] = None
    ):
        """
        Initialize a feedback item.
        
        Args:
            content (str): The feedback content
            feedback_type (Union[FeedbackType, str]): The type of feedback
            rating (float, optional): Numerical rating (1-5 scale)
            target_response_id (str, optional): ID of the response this feedback is for
            metadata (Dict[str, Any], optional): Additional metadata
            feedback_id (str, optional): Unique ID for the feedback
            timestamp (float, optional): The time when the feedback was given
        """
        self.content = content
        
        # Convert string to enum if necessary
        if isinstance(feedback_type, str):
            try:
                self.feedback_type = FeedbackType[feedback_type.upper()]
            except KeyError:
                self.feedback_type = FeedbackType.UNKNOWN
        else:
            self.feedback_type = feedback_type
            
        self.rating = rating
        self.target_response_id = target_response_id
        self.metadata = metadata or {}
        self.feedback_id = feedback_id or f"fb_{int(time.time())}_{id(self)}"
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the feedback to a dictionary."""
        return {
            "feedback_id": self.feedback_id,
            "content": self.content,
            "feedback_type": self.feedback_type.value,
            "rating": self.rating,
            "target_response_id": self.target_response_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Feedback':
        """Create a feedback item from a dictionary."""
        return cls(
            content=data["content"],
            feedback_type=data["feedback_type"],
            rating=data.get("rating"),
            target_response_id=data.get("target_response_id"),
            metadata=data.get("metadata", {}),
            feedback_id=data.get("feedback_id"),
            timestamp=data.get("timestamp")
        )


class KnowledgeItem:
    """Class to represent extracted knowledge."""
    
    def __init__(
        self,
        content: str,
        knowledge_type: Union[KnowledgeType, str],
        source: str,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
        knowledge_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        embedding: Optional[List[float]] = None
    ):
        """
        Initialize a knowledge item.
        
        Args:
            content (str): The knowledge content
            knowledge_type (Union[KnowledgeType, str]): The type of knowledge
            source (str): The source of the knowledge (e.g., conversation ID)
            confidence (float): Confidence score for the extracted knowledge (0-1)
            metadata (Dict[str, Any], optional): Additional metadata
            knowledge_id (str, optional): Unique ID for the knowledge item
            timestamp (float, optional): The time when the knowledge was extracted
            embedding (List[float], optional): Vector embedding of the knowledge
        """
        self.content = content
        
        # Convert string to enum if necessary
        if isinstance(knowledge_type, str):
            try:
                self.knowledge_type = KnowledgeType[knowledge_type.upper()]
            except KeyError:
                self.knowledge_type = KnowledgeType.UNKNOWN
        else:
            self.knowledge_type = knowledge_type
            
        self.source = source
        self.confidence = confidence
        self.metadata = metadata or {}
        self.knowledge_id = knowledge_id or f"know_{int(time.time())}_{id(self)}"
        self.timestamp = timestamp or time.time()
        self.embedding = embedding
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the knowledge item to a dictionary."""
        return {
            "knowledge_id": self.knowledge_id,
            "content": self.content,
            "knowledge_type": self.knowledge_type.value,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeItem':
        """Create a knowledge item from a dictionary."""
        return cls(
            content=data["content"],
            knowledge_type=data["knowledge_type"],
            source=data["source"],
            confidence=data["confidence"],
            metadata=data.get("metadata", {}),
            knowledge_id=data.get("knowledge_id"),
            timestamp=data.get("timestamp"),
            embedding=data.get("embedding")
        )


class ReflectionResult:
    """Class to represent the result of a self-reflection."""
    
    def __init__(
        self,
        insights: List[str],
        improvement_areas: List[str],
        action_plan: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        reflection_id: Optional[str] = None,
        timestamp: Optional[float] = None
    ):
        """
        Initialize a reflection result.
        
        Args:
            insights (List[str]): Insights gained from the reflection
            improvement_areas (List[str]): Areas identified for improvement
            action_plan (List[str]): Action items to implement improvements
            metadata (Dict[str, Any], optional): Additional metadata
            reflection_id (str, optional): Unique ID for the reflection
            timestamp (float, optional): The time when the reflection occurred
        """
        self.insights = insights
        self.improvement_areas = improvement_areas
        self.action_plan = action_plan
        self.metadata = metadata or {}
        self.reflection_id = reflection_id or f"refl_{int(time.time())}_{id(self)}"
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the reflection result to a dictionary."""
        return {
            "reflection_id": self.reflection_id,
            "insights": self.insights,
            "improvement_areas": self.improvement_areas,
            "action_plan": self.action_plan,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReflectionResult':
        """Create a reflection result from a dictionary."""
        return cls(
            insights=data["insights"],
            improvement_areas=data["improvement_areas"],
            action_plan=data["action_plan"],
            metadata=data.get("metadata", {}),
            reflection_id=data.get("reflection_id"),
            timestamp=data.get("timestamp")
        ) 