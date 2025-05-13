"""
Self-reflection for the Autonomous Coding Agent.

This module provides functionality for the agent to reflect on its performance and improve.
"""

import time
import json
import os
from typing import Dict, List, Any, Optional, Union, Iterator
from pathlib import Path
from loguru import logger

import sys
import os
sys.path.append("../..")
try:
    # Try direct import first (when run as a module)
    from memory.manager import get_memory_manager
except ImportError:
    # Try relative import (when run from the autonomous_agent directory)
    from ..memory.manager import get_memory_manager

from .types import ReflectionResult, Experience, KnowledgeItem
from .experience import get_experience_tracker
from .extraction import get_knowledge_extractor
from learning.types import (
    Experience, 
    ExperienceType,
    ReflectionResult,
    KnowledgeType
)

# Singleton instance
_reflector = None

class Reflector:
    """Class to enable agent self-reflection and improvement."""
    
    def __init__(
        self, 
        storage_dir: Optional[Union[str, Path]] = None,
        reflection_period: int = 10
    ):
        """
        Initialize the reflector.
        
        Args:
            storage_dir (Union[str, Path], optional): Directory to store reflection results
            reflection_period (int, optional): Number of experiences before auto-reflection
        """
        if storage_dir is None:
            storage_dir = Path.home() / ".autonomous_agent" / "reflections"
        elif isinstance(storage_dir, str):
            storage_dir = Path(storage_dir)
            
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.reflection_period = reflection_period
        self.reflection_results = {}  # In-memory cache of reflection results
        self._load_reflection_results()
        
        # Get related components
        self.experience_tracker = get_experience_tracker()
        self.memory_manager = get_memory_manager()
        self.knowledge_extractor = get_knowledge_extractor()
        
        # Track the experience count for automatic reflection
        self.experience_count_since_reflection = 0
        
        logger.info(f"Initialized Reflector with storage directory: {storage_dir}")
        
    def reflect_on_period(
        self, 
        days: int = 7,
        experience_types: Optional[List[str]] = None
    ) -> ReflectionResult:
        """
        Reflect on experiences from the last N days.
        
        Args:
            days (int, optional): Number of days to look back
            experience_types (List[str], optional): Filter by experience types
            
        Returns:
            ReflectionResult: The reflection result
        """
        # Get experiences from the last N days
        now = time.time()
        start_time = now - (days * 24 * 60 * 60)
        
        experiences = self.experience_tracker.list_experiences(
            experience_type=experience_types[0] if experience_types else None,
            start_time=start_time
        )
        
        if experience_types and len(experience_types) > 1:
            # Filter by multiple experience types
            experiences = [
                exp for exp in experiences 
                if exp.experience_type.value in experience_types
            ]
            
        return self.reflect_on_experiences(experiences)
        
    def reflect_on_experiences(self, experiences: List[Experience]) -> ReflectionResult:
        """
        Reflect on a specific set of experiences.
        
        Args:
            experiences (List[Experience]): The experiences to reflect on
            
        Returns:
            ReflectionResult: The reflection result
        """
        if not experiences:
            logger.warning("No experiences to reflect on")
            return ReflectionResult(
                insights=["No experiences available for reflection."],
                improvement_areas=["Gather more experiences to enable meaningful reflection."],
                action_plan=["Continue operating to gather experiences."]
            )
            
        # Analyze experiences
        analysis = self._analyze_experiences(experiences)
        
        # Generate insights
        insights = self._generate_insights(analysis)
        
        # Identify areas for improvement
        improvement_areas = self._identify_improvement_areas(analysis)
        
        # Create action plan
        action_plan = self._create_action_plan(improvement_areas)
        
        # Create reflection result
        reflection_result = ReflectionResult(
            insights=insights,
            improvement_areas=improvement_areas,
            action_plan=action_plan,
            metadata={
                "experience_count": len(experiences),
                "experience_types": list(set(exp.experience_type.value for exp in experiences)),
                "time_range": {
                    "start": min(exp.timestamp for exp in experiences),
                    "end": max(exp.timestamp for exp in experiences)
                },
                "analysis": analysis
            }
        )
        
        # Save the reflection result
        self._save_reflection_result(reflection_result)
        
        # Reset the experience counter
        self.experience_count_since_reflection = 0
        
        return reflection_result
        
    def get_latest_reflection(self) -> Optional[ReflectionResult]:
        """
        Get the most recent reflection result.
        
        Returns:
            Optional[ReflectionResult]: The most recent reflection result, or None if none exists
        """
        if not self.reflection_results:
            return None
            
        # Get the most recent reflection by timestamp
        latest_id = max(self.reflection_results.keys(), key=lambda k: self.reflection_results[k].timestamp)
        return self.reflection_results[latest_id]
        
    def get_reflection(self, reflection_id: str) -> Optional[ReflectionResult]:
        """
        Get a reflection result by ID.
        
        Args:
            reflection_id (str): The ID of the reflection result to retrieve
            
        Returns:
            Optional[ReflectionResult]: The reflection result if found, None otherwise
        """
        # Try to get from memory cache
        if reflection_id in self.reflection_results:
            return self.reflection_results[reflection_id]
        
        # Try to load from disk
        reflection_path = self.storage_dir / f"{reflection_id}.json"
        if reflection_path.exists():
            with open(reflection_path, "r") as f:
                data = json.load(f)
                reflection = ReflectionResult(
                    insights=data["insights"],
                    improvement_areas=data["improvement_areas"],
                    action_plan=data["action_plan"],
                    metadata=data.get("metadata", {}),
                    reflection_id=data.get("reflection_id"),
                    timestamp=data.get("timestamp")
                )
                self.reflection_results[reflection_id] = reflection
                return reflection
                
        return None
        
    def list_reflections(
        self, 
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[ReflectionResult]:
        """
        List reflection results with optional filtering.
        
        Args:
            start_time (float, optional): Only include reflections after this time
            end_time (float, optional): Only include reflections before this time
            limit (int, optional): Maximum number of reflections to return
            
        Returns:
            List[ReflectionResult]: Matching reflection results
        """
        # Load all reflection results to ensure we have the latest data
        self._load_reflection_results()
        
        # Filter reflection results
        filtered_reflections = []
        for refl in self.reflection_results.values():
            if start_time and refl.timestamp < start_time:
                continue
                
            if end_time and refl.timestamp > end_time:
                continue
                
            filtered_reflections.append(refl)
            
        # Sort by timestamp (newest first)
        filtered_reflections.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            filtered_reflections = filtered_reflections[:limit]
            
        return filtered_reflections
        
    def notify_new_experience(self, experience: Experience) -> bool:
        """
        Notify the reflector of a new experience and potentially trigger reflection.
        
        Args:
            experience (Experience): The new experience
            
        Returns:
            bool: True if reflection was triggered, False otherwise
        """
        self.experience_count_since_reflection += 1
        
        # Check if we should trigger automatic reflection
        if self.experience_count_since_reflection >= self.reflection_period:
            # Get recent experiences
            experiences = self.experience_tracker.list_experiences(limit=self.reflection_period)
            self.reflect_on_experiences(experiences)
            return True
            
        return False
        
    def _analyze_experiences(self, experiences: List[Experience]) -> Dict[str, Any]:
        """
        Analyze a set of experiences to extract patterns and statistics.
        
        Args:
            experiences (List[Experience]): The experiences to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        analysis = {}
        
        # Outcome analysis
        outcome_counts = {}
        for exp in experiences:
            outcome = exp.outcome or "unknown"
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            
        analysis["outcome_counts"] = outcome_counts
        analysis["success_rate"] = outcome_counts.get("success", 0) / len(experiences) if experiences else 0
        
        # Feedback analysis
        ratings = []
        feedback_types = {}
        for exp in experiences:
            for fb in exp.feedback.values():
                if "rating" in fb and fb["rating"] is not None:
                    ratings.append(fb["rating"])
                if "type" in fb:
                    fb_type = fb["type"]
                    feedback_types[fb_type] = feedback_types.get(fb_type, 0) + 1
                    
        analysis["average_rating"] = sum(ratings) / len(ratings) if ratings else 0
        analysis["feedback_types"] = feedback_types
        
        # Experience type distribution
        type_counts = {}
        for exp in experiences:
            exp_type = exp.experience_type.value
            type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
            
        analysis["experience_type_counts"] = type_counts
        
        # Extract patterns in unsuccessful experiences
        unsuccessful_exps = [exp for exp in experiences if exp.outcome == "failure"]
        unsuccessful_patterns = {}
        
        for exp in unsuccessful_exps:
            # Simple pattern: check for error keywords in response
            error_keywords = ["error", "exception", "failed", "not working", "bug"]
            for keyword in error_keywords:
                if keyword in exp.response.lower():
                    unsuccessful_patterns[keyword] = unsuccessful_patterns.get(keyword, 0) + 1
                    
        analysis["unsuccessful_patterns"] = unsuccessful_patterns
        
        return analysis
        
    def _generate_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate insights from analysis results.
        
        Args:
            analysis (Dict[str, Any]): Analysis results
            
        Returns:
            List[str]: Generated insights
        """
        insights = []
        
        # Success rate insight
        success_rate = analysis.get("success_rate", 0)
        if success_rate > 0.8:
            insights.append(f"High success rate of {success_rate:.1%} indicates effective performance.")
        elif success_rate < 0.5:
            insights.append(f"Low success rate of {success_rate:.1%} suggests significant room for improvement.")
        else:
            insights.append(f"Moderate success rate of {success_rate:.1%} indicates effective but improvable performance.")
            
        # Rating insight
        avg_rating = analysis.get("average_rating", 0)
        if avg_rating > 4:
            insights.append(f"High average rating of {avg_rating:.1f} indicates strong user satisfaction.")
        elif avg_rating < 3:
            insights.append(f"Lower average rating of {avg_rating:.1f} suggests user satisfaction needs improvement.")
            
        # Experience type insights
        type_counts = analysis.get("experience_type_counts", {})
        if type_counts:
            most_common_type = max(type_counts.items(), key=lambda x: x[1])
            insights.append(f"Most common experience type is {most_common_type[0]} ({most_common_type[1]} occurrences), indicating user preference.")
            
        # Unsuccessful patterns
        unsuccessful_patterns = analysis.get("unsuccessful_patterns", {})
        if unsuccessful_patterns:
            most_common_pattern = max(unsuccessful_patterns.items(), key=lambda x: x[1]) if unsuccessful_patterns else (None, 0)
            if most_common_pattern[0]:
                insights.append(f"Most common issue in unsuccessful experiences relates to '{most_common_pattern[0]}' ({most_common_pattern[1]} occurrences).")
                
        return insights
        
    def _identify_improvement_areas(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Identify areas for improvement from analysis results.
        
        Args:
            analysis (Dict[str, Any]): Analysis results
            
        Returns:
            List[str]: Identified improvement areas
        """
        improvement_areas = []
        
        # Success rate improvement
        success_rate = analysis.get("success_rate", 0)
        if success_rate < 0.8:
            improvement_areas.append(f"Increase success rate from current {success_rate:.1%}.")
            
        # Rating improvement
        avg_rating = analysis.get("average_rating", 0)
        if avg_rating < 4:
            improvement_areas.append(f"Improve user satisfaction rating from current {avg_rating:.1f}.")
            
        # Feedback type improvements
        feedback_types = analysis.get("feedback_types", {})
        if "correction" in feedback_types and feedback_types["correction"] > 3:
            improvement_areas.append("Reduce the number of corrections needed by improving initial responses.")
        if "clarification" in feedback_types and feedback_types["clarification"] > 3:
            improvement_areas.append("Provide clearer explanations to reduce clarification requests.")
            
        # Unsuccessful patterns improvements
        unsuccessful_patterns = analysis.get("unsuccessful_patterns", {})
        for pattern, count in unsuccessful_patterns.items():
            if count > 2:
                improvement_areas.append(f"Address recurring issues related to '{pattern}'.")
                
        return improvement_areas
        
    def _create_action_plan(self, improvement_areas: List[str]) -> List[str]:
        """
        Create an action plan based on identified improvement areas.
        
        Args:
            improvement_areas (List[str]): Identified improvement areas
            
        Returns:
            List[str]: Action plan items
        """
        action_plan = []
        
        for area in improvement_areas:
            if "success rate" in area.lower():
                action_plan.append("Review unsuccessful experiences and extract common patterns.")
                action_plan.append("Store solutions to common errors in long-term memory.")
                
            elif "user satisfaction" in area.lower() or "rating" in area.lower():
                action_plan.append("Enhance response quality with more detailed explanations.")
                action_plan.append("Follow up with users to ensure their questions are fully answered.")
                
            elif "correction" in area.lower():
                action_plan.append("Implement additional validation before providing responses.")
                action_plan.append("Extract knowledge from corrections to improve future responses.")
                
            elif "clarification" in area.lower():
                action_plan.append("Be more thorough in initial explanations.")
                action_plan.append("Anticipate common follow-up questions and address them proactively.")
                
            elif "recurring issues" in area.lower():
                pattern = area.split("'")[1] if "'" in area else ""
                action_plan.append(f"Create specialized knowledge items for handling '{pattern}' issues.")
                action_plan.append(f"Prioritize learning resources related to '{pattern}'.")
                
            else:
                # Generic action for other improvement areas
                action_plan.append(f"Create and implement a specific strategy to address: {area}")
                
        # If no specific actions, add generic ones
        if not action_plan:
            action_plan = [
                "Continue gathering experiences to identify patterns.",
                "Extract more knowledge from successful interactions.",
                "Regularly review and consolidate long-term memory."
            ]
            
        return action_plan
        
    def _save_reflection_result(self, reflection: ReflectionResult):
        """
        Save a reflection result to disk and memory cache.
        
        Args:
            reflection (ReflectionResult): The reflection result to save
        """
        # Save to memory cache
        self.reflection_results[reflection.reflection_id] = reflection
        
        # Save to disk
        reflection_path = self.storage_dir / f"{reflection.reflection_id}.json"
        with open(reflection_path, "w") as f:
            json.dump(reflection.to_dict(), f, indent=2)
            
        logger.debug(f"Saved reflection result: {reflection.reflection_id}")
        
        # Also add insights and improvement areas to long-term memory
        self._store_reflection_in_memory(reflection)
        
    def _store_reflection_in_memory(self, reflection: ReflectionResult):
        """
        Store reflection insights and action plans in long-term memory.
        
        Args:
            reflection (ReflectionResult): The reflection result to store
        """
        # Extract insights and store in long-term memory
        for insight in reflection.insights:
            self.knowledge_extractor.extract_knowledge_from_text(
                text=insight,
                source=f"reflection:{reflection.reflection_id}",
                knowledge_types=[KnowledgeType.FACT]
            )
            
        # Extract improvement areas as concepts
        for area in reflection.improvement_areas:
            self.knowledge_extractor.extract_knowledge_from_text(
                text=area,
                source=f"reflection:{reflection.reflection_id}",
                knowledge_types=[KnowledgeType.CONCEPT]
            )
            
        # Store the action plan items
        for action in reflection.action_plan:
            self.memory_manager.add_to_long_term(
                content=action,
                item_type="action_plan",
                metadata={
                    "reflection_id": reflection.reflection_id,
                    "timestamp": reflection.timestamp
                }
            )
            
    def _load_reflection_results(self):
        """Load reflection results from disk into memory cache."""
        # Clear existing cache
        self.reflection_results = {}
        
        # Load from disk
        for reflection_file in self.storage_dir.glob("*.json"):
            try:
                with open(reflection_file, "r") as f:
                    data = json.load(f)
                    reflection = ReflectionResult(
                        insights=data["insights"],
                        improvement_areas=data["improvement_areas"],
                        action_plan=data["action_plan"],
                        metadata=data.get("metadata", {}),
                        reflection_id=data.get("reflection_id"),
                        timestamp=data.get("timestamp")
                    )
                    self.reflection_results[reflection.reflection_id] = reflection
            except Exception as e:
                logger.error(f"Error loading reflection file {reflection_file}: {e}")
                
        logger.debug(f"Loaded {len(self.reflection_results)} reflection results from disk")


def get_reflector(
    storage_dir: Optional[Union[str, Path]] = None,
    reflection_period: int = 10
) -> Reflector:
    """
    Get or create the singleton Reflector instance.
    
    Args:
        storage_dir (Union[str, Path], optional): Directory to store reflection results
        reflection_period (int, optional): Number of experiences before auto-reflection
        
    Returns:
        Reflector: The singleton Reflector instance
    """
    global _reflector
    if _reflector is None:
        _reflector = Reflector(storage_dir, reflection_period)
    return _reflector 