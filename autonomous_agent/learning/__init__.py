"""
Learning-related modules for the Autonomous Coding Agent.

This module provides mechanisms for the agent to learn from interactions,
incorporate feedback, extract knowledge, and improve over time.
"""

from .types import (
    Experience, Feedback, KnowledgeItem, ReflectionResult,
    ExperienceType, FeedbackType, KnowledgeType
)

from .experience import ExperienceTracker, get_experience_tracker
from .feedback import FeedbackProcessor, get_feedback_processor
from .extraction import KnowledgeExtractor, get_knowledge_extractor
from .reflection import Reflector, get_reflector
from .manager import LearningManager, get_learning_manager

# Export the learning components for easy access
__all__ = [
    # Data types
    'Experience',
    'Feedback',
    'KnowledgeItem',
    'ReflectionResult',
    'ExperienceType',
    'FeedbackType',
    'KnowledgeType',
    
    # Experience tracking
    'ExperienceTracker',
    'get_experience_tracker',
    
    # Feedback incorporation
    'FeedbackProcessor',
    'get_feedback_processor',
    
    # Knowledge extraction
    'KnowledgeExtractor',
    'get_knowledge_extractor',
    
    # Self-reflection
    'Reflector',
    'get_reflector',
    
    # Learning manager
    'LearningManager',
    'get_learning_manager',
] 