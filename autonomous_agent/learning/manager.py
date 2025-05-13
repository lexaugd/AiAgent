"""
Learning manager for the Autonomous Coding Agent.

This module provides a unified interface for coordinating learning mechanisms.
"""

import time
import json
import os
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from loguru import logger

# Import memory manager using relative path
try:
    # Try direct import first (when run as a module)
    from memory.manager import get_memory_manager
except ImportError:
    # Try relative import (when run from the autonomous_agent directory)
    from ..memory.manager import get_memory_manager

from .types import Experience, Feedback, KnowledgeItem, ReflectionResult
from .types import ExperienceType, FeedbackType, KnowledgeType
from .experience import get_experience_tracker
from .feedback import get_feedback_processor
from .extraction import get_knowledge_extractor
from .reflection import get_reflector

# Singleton instance
_learning_manager = None

class LearningManager:
    """
    Class to coordinate and integrate all learning components.
    
    This manager serves as the central point for all learning-related functionality,
    providing a unified interface for recording experiences, processing feedback,
    extracting knowledge, and triggering reflection.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the learning manager.
        
        Args:
            config (Dict[str, Any], optional): Configuration parameters
        """
        self.config = config or {}
        
        # Initialize and get component instances
        self.experience_tracker = get_experience_tracker(
            storage_dir=self.config.get("experience_storage_dir"),
            max_cache_size=self.config.get("experience_cache_size", 100)
        )
        
        self.feedback_processor = get_feedback_processor(
            storage_dir=self.config.get("feedback_storage_dir"),
            max_cache_size=self.config.get("feedback_cache_size", 50)
        )
        
        self.knowledge_extractor = get_knowledge_extractor(
            extraction_threshold=self.config.get("knowledge_extraction_threshold", 0.6),
            embedding_batch_size=self.config.get("embedding_batch_size", 5)
        )
        
        self.reflector = get_reflector(
            storage_dir=self.config.get("reflection_storage_dir"),
            reflection_period=self.config.get("reflection_period", 10)
        )
        
        # Get memory manager
        self.memory_manager = get_memory_manager()
        
        logger.info(f"Initialized LearningManager with config: {self.config}")
        
    def record_experience(
        self,
        context: str,
        query: str,
        response: str,
        experience_type: Union[ExperienceType, str],
        metadata: Optional[Dict[str, Any]] = None,
        extract_knowledge: bool = True
    ) -> str:
        """
        Record an agent experience and optionally extract knowledge from it.
        
        Args:
            context (str): The context in which the experience occurred
            query (str): The user query that triggered the experience
            response (str): The agent's response
            experience_type (Union[ExperienceType, str]): The type of experience
            metadata (Dict[str, Any], optional): Additional metadata
            extract_knowledge (bool, optional): Whether to extract knowledge
            
        Returns:
            str: The ID of the recorded experience
        """
        # Create and record the experience
        experience = Experience(
            context=context,
            query=query,
            response=response,
            experience_type=experience_type,
            metadata=metadata or {}
        )
        
        experience_id = self.experience_tracker.record_experience(experience)
        
        # Extract knowledge if requested
        if extract_knowledge:
            self.knowledge_extractor.extract_from_experience(experience)
            
        # Notify reflector about the new experience
        reflection_triggered = self.reflector.notify_new_experience(experience)
        
        if reflection_triggered:
            logger.info(f"Recording experience {experience_id} triggered automatic reflection")
            
        return experience_id
        
    def process_feedback(
        self,
        content: str,
        feedback_type: Union[FeedbackType, str],
        target_response_id: Optional[str] = None,
        rating: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        update_experience: bool = True
    ) -> str:
        """
        Process user feedback and optionally update the associated experience.
        
        Args:
            content (str): The feedback content
            feedback_type (Union[FeedbackType, str]): The type of feedback
            target_response_id (str, optional): ID of the response this feedback is for
            rating (float, optional): Numerical rating (1-5 scale)
            metadata (Dict[str, Any], optional): Additional metadata
            update_experience (bool, optional): Whether to update the associated experience
            
        Returns:
            str: The ID of the processed feedback
        """
        # Create and process the feedback
        feedback = Feedback(
            content=content,
            feedback_type=feedback_type,
            rating=rating,
            target_response_id=target_response_id,
            metadata=metadata or {}
        )
        
        feedback_id = self.feedback_processor.process_feedback(feedback)
        
        # Update the experience outcome based on feedback type
        if update_experience and target_response_id:
            outcome = None
            
            # Determine outcome based on feedback type
            if isinstance(feedback_type, str):
                try:
                    feedback_type_enum = FeedbackType[feedback_type.upper()]
                except KeyError:
                    feedback_type_enum = FeedbackType.UNKNOWN
            else:
                feedback_type_enum = feedback_type
                
            if feedback_type_enum == FeedbackType.CONFIRMATION:
                outcome = "success"
            elif feedback_type_enum == FeedbackType.REJECTION:
                outcome = "failure"
            elif rating is not None:
                # Rating-based outcome (4+ is success, below 3 is failure)
                if rating >= 4:
                    outcome = "success"
                elif rating < 3:
                    outcome = "failure"
                    
            if outcome:
                self.experience_tracker.update_experience(
                    experience_id=target_response_id,
                    outcome=outcome
                )
                
        return feedback_id
        
    def extract_knowledge(
        self,
        text: str,
        source: str,
        knowledge_types: Optional[List[Union[KnowledgeType, str]]] = None
    ) -> List[KnowledgeItem]:
        """
        Extract knowledge directly from text.
        
        Args:
            text (str): The text to extract knowledge from
            source (str): The source of the text
            knowledge_types (List[Union[KnowledgeType, str]], optional): Types of knowledge to extract
            
        Returns:
            List[KnowledgeItem]: Extracted knowledge items
        """
        # Convert string types to enums if necessary
        if knowledge_types:
            enum_types = []
            for kt in knowledge_types:
                if isinstance(kt, str):
                    try:
                        enum_types.append(KnowledgeType[kt.upper()])
                    except KeyError:
                        enum_types.append(KnowledgeType.UNKNOWN)
                else:
                    enum_types.append(kt)
        else:
            enum_types = None
            
        return self.knowledge_extractor.extract_knowledge_from_text(
            text=text,
            source=source,
            knowledge_types=enum_types
        )
        
    def trigger_reflection(
        self,
        days: int = 7,
        experience_types: Optional[List[str]] = None
    ) -> ReflectionResult:
        """
        Trigger a reflection on recent experiences.
        
        Args:
            days (int, optional): Number of days to look back
            experience_types (List[str], optional): Filter by experience types
            
        Returns:
            ReflectionResult: The reflection result
        """
        return self.reflector.reflect_on_period(days, experience_types)
        
    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all learning components.
        
        Returns:
            Dict[str, Any]: Statistics from all learning components
        """
        stats = {
            "experiences": self.experience_tracker.get_statistics(),
            "feedback": self.feedback_processor.get_statistics(),
            "feedback_trends": self.feedback_processor.analyze_feedback_trends(),
            "reflections": {
                "total_reflections": len(self.reflector.reflection_results),
                "latest_reflection": None
            }
        }
        
        # Add latest reflection info if available
        latest_reflection = self.reflector.get_latest_reflection()
        if latest_reflection:
            stats["reflections"]["latest_reflection"] = {
                "reflection_id": latest_reflection.reflection_id,
                "timestamp": latest_reflection.timestamp,
                "insights_count": len(latest_reflection.insights),
                "improvement_areas_count": len(latest_reflection.improvement_areas),
                "action_plan_count": len(latest_reflection.action_plan)
            }
            
        return stats
        
    def integrate_external_knowledge(
        self,
        content: str,
        knowledge_type: Union[KnowledgeType, str],
        source: str,
        confidence: float = 0.9,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Integrate external knowledge directly into the learning system.
        
        Args:
            content (str): The knowledge content
            knowledge_type (Union[KnowledgeType, str]): The type of knowledge
            source (str): The source of the knowledge
            confidence (float, optional): Confidence score (0-1)
            metadata (Dict[str, Any], optional): Additional metadata
            
        Returns:
            str: The ID of the created knowledge item
        """
        # Create a knowledge item
        knowledge_item = KnowledgeItem(
            content=content,
            knowledge_type=knowledge_type,
            source=source,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        # Generate embedding
        self.knowledge_extractor._generate_embeddings([knowledge_item])
        
        # Store in long-term memory
        self.knowledge_extractor._store_knowledge_items([knowledge_item])
        
        return knowledge_item.knowledge_id
        
    def get_improvement_suggestions(self) -> List[str]:
        """
        Get suggestions for improvement based on reflections and feedback.
        
        Returns:
            List[str]: Suggestions for improvement.
        """
        suggestions = []
        
        # Get suggestions from the latest reflection
        latest_reflection = self.reflector.get_latest_reflection()
        if latest_reflection:
            suggestions.extend(latest_reflection.improvement_areas)
        
        # Get suggestions from feedback trends
        feedback_trends = self.feedback_processor.analyze_feedback_trends()
        trend_direction = feedback_trends.get("trend_direction", "stable")
        
        # Add suggestion based on trend direction
        avg_rating = feedback_trends.get("rating_trends", {}).get("all_time", {}).get("average_rating", 0)
        if trend_direction == "declining":
            suggestions.append(f"Recent feedback ratings are declining. Current average is {avg_rating:.1f}. Identify areas of concern.")
        
        # Get experiences to analyze for success rate
        experiences = self.experience_tracker.list_experiences(limit=100)
        if experiences:
            success_count = sum(1 for exp in experiences if exp.outcome == "success")
            success_rate = success_count / len(experiences)
            
            if success_rate < 0.8:
                suggestions.append(f"Current success rate is {success_rate:.1%}. Focus on improving response quality.")
        
        return suggestions
        
    def retrieve(self, query: str, limit: int = 5, include_metadata: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge from memory based on a query.
        
        Args:
            query (str): The query to search for.
            limit (int): Maximum number of results to return.
            include_metadata (bool): Whether to include metadata in the results.
            
        Returns:
            List[Dict[str, Any]]: The retrieved knowledge items.
        """
        # Use memory manager's retrieve_relevant method
        items = self.memory_manager.retrieve_relevant(query=query, n_results=limit)
        
        # Also get code examples if they might be relevant
        code_items = self.memory_manager.retrieve_code_examples(query=query, n_results=limit//2)
        
        # Combine results
        all_items = items + code_items
        
        # Convert to a standardized format
        results = []
        for item in all_items:
            result = {
                "content": item.content,
                "type": item.item_type,
                "id": item.item_id
            }
            
            # Add similarity score if available (not implemented yet, placeholder)
            result["similarity"] = 0.0
            
            # Include metadata if requested
            if include_metadata and hasattr(item, 'metadata'):
                result["metadata"] = item.metadata
                
                # Extract source
                if "source" in item.metadata:
                    result["source"] = item.metadata["source"]
                    
                # Extract confidence
                if "confidence" in item.metadata:
                    result["confidence"] = item.metadata["confidence"]
                    result["similarity"] = float(item.metadata["confidence"])  # Use confidence as similarity
            
            results.append(result)
            
        # Remove duplicates based on content
        unique_results = {}
        for result in results:
            content = result["content"]
            if content not in unique_results:
                unique_results[content] = result
                
        # Sort by similarity score
        sorted_results = sorted(
            unique_results.values(), 
            key=lambda x: x.get("similarity", 0), 
            reverse=True
        )
        
        # Return limited number of results
        return sorted_results[:limit]
        
    def learn_from_conversation(
        self, 
        messages: List[Dict[str, Any]],
        conversation_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an entire conversation to extract experiences, feedback, and knowledge.
        
        Args:
            messages (List[Dict[str, Any]]): List of conversation messages
            conversation_id (str): Identifier for the conversation
            metadata (Dict[str, Any], optional): Additional metadata about the conversation
            
        Returns:
            Dict[str, Any]: Results of the learning process
        """
        results = {
            "experiences": [],
            "feedback": [],
            "knowledge_items": [],
            "conversation_id": conversation_id
        }
        
        if not messages:
            return results
            
        # Process messages to extract experiences and feedback
        current_query = ""
        current_context = ""
        
        for i, message in enumerate(messages):
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "user":
                current_query = content
                # Update context with previous messages
                if i > 0:
                    prev_messages = messages[:i]
                    current_context = "\n".join([f"{m.get('role', '')}: {m.get('content', '')}" for m in prev_messages])
                    
            elif role == "assistant" and current_query:
                # Record an experience
                experience_type = self._determine_experience_type(current_query, content)
                exp_id = self.record_experience(
                    context=current_context,
                    query=current_query,
                    response=content,
                    experience_type=experience_type,
                    metadata={
                        "conversation_id": conversation_id,
                        "message_index": i,
                        **(metadata or {})
                    },
                    extract_knowledge=True
                )
                results["experiences"].append(exp_id)
                
                # Reset the current query
                current_query = ""
                
            elif role == "user" and i > 0 and messages[i-1].get("role") == "assistant":
                # Check if this is feedback to the previous assistant message
                feedback_type = self._determine_feedback_type(content)
                if feedback_type != FeedbackType.UNKNOWN:
                    # This is likely feedback
                    target_response_id = results["experiences"][-1] if results["experiences"] else None
                    if target_response_id:
                        fb_id = self.process_feedback(
                            content=content,
                            feedback_type=feedback_type,
                            target_response_id=target_response_id,
                            metadata={
                                "conversation_id": conversation_id,
                                "message_index": i,
                                **(metadata or {})
                            }
                        )
                        results["feedback"].append(fb_id)
                        
        # Extract overall knowledge from the entire conversation
        conversation_text = "\n".join([f"{m.get('role', '')}: {m.get('content', '')}" for m in messages])
        knowledge_items = self.extract_knowledge(
            text=conversation_text,
            source=f"conversation:{conversation_id}"
        )
        
        results["knowledge_items"] = [item.knowledge_id for item in knowledge_items]
        
        return results
        
    def _determine_experience_type(self, query: str, response: str) -> ExperienceType:
        """Determine the experience type based on the query and response content."""
        # Simple keyword-based heuristics
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ["error", "bug", "fix", "issue", "problem", "fail"]):
            return ExperienceType.ERROR_RESOLUTION
            
        if any(kw in query_lower for kw in ["explain", "how does", "what is", "meaning of"]):
            return ExperienceType.CODE_EXPLANATION
            
        if any(kw in query_lower for kw in ["create", "generate", "write", "implement", "code for"]):
            return ExperienceType.CODE_GENERATION
            
        if any(kw in query_lower for kw in ["plan", "steps to", "how to", "approach"]):
            return ExperienceType.TASK_PLANNING
            
        # Check for code in response as a fallback
        if "```" in response or response.count("    ") > 5:
            return ExperienceType.CODE_GENERATION
            
        return ExperienceType.QUESTION_ANSWERING
        
    def _determine_feedback_type(self, content: str) -> FeedbackType:
        """Determine the feedback type based on the content."""
        content_lower = content.lower()
        
        # Simple keyword-based heuristics
        if any(kw in content_lower for kw in ["thank", "thanks", "perfect", "great", "excellent", "awesome"]):
            return FeedbackType.CONFIRMATION
            
        if any(kw in content_lower for kw in ["wrong", "incorrect", "error", "mistake", "not right", "instead"]):
            return FeedbackType.CORRECTION
            
        if any(kw in content_lower for kw in ["no", "not what i wanted", "doesn't work", "does not work"]):
            return FeedbackType.REJECTION
            
        if any(kw in content_lower for kw in ["what do you mean", "explain", "clarify", "don't understand"]):
            return FeedbackType.CLARIFICATION
            
        # If the message starts with a question, it's likely not feedback
        if content_lower.strip().endswith("?") or content_lower.startswith("why") or content_lower.startswith("how"):
            return FeedbackType.UNKNOWN
            
        return FeedbackType.UNKNOWN


def get_learning_manager(config: Optional[Dict[str, Any]] = None) -> LearningManager:
    """
    Get or create the singleton LearningManager instance.
    
    Args:
        config (Dict[str, Any], optional): Configuration parameters
        
    Returns:
        LearningManager: The singleton LearningManager instance
    """
    global _learning_manager
    if _learning_manager is None:
        _learning_manager = LearningManager(config)
    return _learning_manager 