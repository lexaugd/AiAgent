"""
Experience tracking for the Autonomous Coding Agent.

This module provides functionality to track and record agent experiences.
"""

import time
import json
import os
from typing import Dict, List, Any, Optional, Union, Iterator
from pathlib import Path
from loguru import logger

from .types import Experience, ExperienceType

# Singleton instance
_experience_tracker = None

class ExperienceTracker:
    """Class to track and record agent experiences."""
    
    def __init__(
        self, 
        storage_dir: Optional[Union[str, Path]] = None,
        max_cache_size: int = 100
    ):
        """
        Initialize the experience tracker.
        
        Args:
            storage_dir (Union[str, Path], optional): Directory to store experiences
            max_cache_size (int, optional): Maximum number of experiences to keep in memory
        """
        if storage_dir is None:
            storage_dir = Path.home() / ".autonomous_agent" / "experiences"
        elif isinstance(storage_dir, str):
            storage_dir = Path(storage_dir)
            
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size = max_cache_size
        self.experiences = {}  # In-memory cache of experiences
        self._load_experiences()
        
        logger.info(f"Initialized ExperienceTracker with storage directory: {storage_dir}")
        
    def record_experience(self, experience: Experience) -> str:
        """
        Record a new experience.
        
        Args:
            experience (Experience): The experience to record
            
        Returns:
            str: The ID of the recorded experience
        """
        # Save to memory cache
        self.experiences[experience.experience_id] = experience
        
        # Save to disk
        experience_path = self.storage_dir / f"{experience.experience_id}.json"
        with open(experience_path, "w") as f:
            json.dump(experience.to_dict(), f, indent=2)
            
        # If cache is too large, remove oldest experiences
        if len(self.experiences) > self.max_cache_size:
            oldest_id = sorted(
                self.experiences.keys(), 
                key=lambda x: self.experiences[x].timestamp
            )[0]
            del self.experiences[oldest_id]
            
        logger.debug(f"Recorded experience: {experience.experience_id} ({experience.experience_type.value})")
        return experience.experience_id
        
    def get_experience(self, experience_id: str) -> Optional[Experience]:
        """
        Get an experience by ID.
        
        Args:
            experience_id (str): The ID of the experience to retrieve
            
        Returns:
            Optional[Experience]: The experience if found, None otherwise
        """
        # Try to get from memory cache
        if experience_id in self.experiences:
            return self.experiences[experience_id]
        
        # Try to load from disk
        experience_path = self.storage_dir / f"{experience_id}.json"
        if experience_path.exists():
            with open(experience_path, "r") as f:
                data = json.load(f)
                experience = Experience.from_dict(data)
                self.experiences[experience_id] = experience
                return experience
                
        return None
        
    def update_experience(
        self, 
        experience_id: str, 
        outcome: Optional[str] = None,
        feedback: Optional[Dict[str, Any]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing experience.
        
        Args:
            experience_id (str): The ID of the experience to update
            outcome (str, optional): New outcome value
            feedback (Dict[str, Any], optional): Feedback to add to the experience
            metadata_updates (Dict[str, Any], optional): Updates to the metadata
            
        Returns:
            bool: True if the experience was updated, False otherwise
        """
        experience = self.get_experience(experience_id)
        if not experience:
            logger.warning(f"Cannot update experience: {experience_id} not found")
            return False
            
        if outcome:
            experience.outcome = outcome
            
        if feedback:
            experience.feedback.update(feedback)
            
        if metadata_updates:
            experience.metadata.update(metadata_updates)
            
        # Update timestamp to reflect the modification
        experience.metadata["last_modified"] = time.time()
        
        # Save to disk
        experience_path = self.storage_dir / f"{experience_id}.json"
        with open(experience_path, "w") as f:
            json.dump(experience.to_dict(), f, indent=2)
            
        logger.debug(f"Updated experience: {experience_id}")
        return True
        
    def delete_experience(self, experience_id: str) -> bool:
        """
        Delete an experience.
        
        Args:
            experience_id (str): The ID of the experience to delete
            
        Returns:
            bool: True if the experience was deleted, False otherwise
        """
        # Remove from memory cache
        if experience_id in self.experiences:
            del self.experiences[experience_id]
        
        # Remove from disk
        experience_path = self.storage_dir / f"{experience_id}.json"
        if experience_path.exists():
            experience_path.unlink()
            logger.debug(f"Deleted experience: {experience_id}")
            return True
            
        logger.warning(f"Cannot delete experience: {experience_id} not found")
        return False
        
    def list_experiences(
        self, 
        experience_type: Optional[Union[ExperienceType, str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[Experience]:
        """
        List experiences with optional filtering.
        
        Args:
            experience_type (Union[ExperienceType, str], optional): Filter by experience type
            start_time (float, optional): Only include experiences after this time
            end_time (float, optional): Only include experiences before this time
            limit (int, optional): Maximum number of experiences to return
            
        Returns:
            List[Experience]: Matching experiences
        """
        # Convert string to enum if necessary
        if isinstance(experience_type, str):
            try:
                experience_type = ExperienceType[experience_type.upper()]
            except KeyError:
                experience_type = None
                
        # Load all experiences from disk to ensure we have the latest data
        self._load_experiences()
        
        # Filter experiences
        filtered_experiences = []
        for exp in self.experiences.values():
            if experience_type and exp.experience_type != experience_type:
                continue
                
            if start_time and exp.timestamp < start_time:
                continue
                
            if end_time and exp.timestamp > end_time:
                continue
                
            filtered_experiences.append(exp)
            
        # Sort by timestamp (newest first)
        filtered_experiences.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            filtered_experiences = filtered_experiences[:limit]
            
        return filtered_experiences
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the recorded experiences.
        
        Returns:
            Dict[str, Any]: Statistics about the experiences
        """
        # Load all experiences to ensure we have the latest data
        self._load_experiences()
        
        # Count experiences by type
        type_counts = {}
        for exp_type in ExperienceType:
            type_counts[exp_type.value] = 0
            
        # Count experiences by outcome
        outcome_counts = {}
        
        # Calculate time ranges
        timestamps = []
        
        for exp in self.experiences.values():
            type_counts[exp.experience_type.value] += 1
            
            outcome = exp.outcome or "unknown"
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            
            timestamps.append(exp.timestamp)
            
        # Calculate time range
        time_range = {
            "oldest": min(timestamps) if timestamps else None,
            "newest": max(timestamps) if timestamps else None,
            "total_days": (max(timestamps) - min(timestamps)) / (60 * 60 * 24) if timestamps else 0
        }
        
        return {
            "total_experiences": len(self.experiences),
            "type_counts": type_counts,
            "outcome_counts": outcome_counts,
            "time_range": time_range
        }
        
    def _load_experiences(self):
        """Load experiences from disk into memory cache."""
        # Clear existing cache
        self.experiences = {}
        
        # Load from disk
        for experience_file in self.storage_dir.glob("*.json"):
            try:
                with open(experience_file, "r") as f:
                    data = json.load(f)
                    experience = Experience.from_dict(data)
                    self.experiences[experience.experience_id] = experience
            except Exception as e:
                logger.error(f"Error loading experience file {experience_file}: {e}")
                
        logger.debug(f"Loaded {len(self.experiences)} experiences from disk")
        
    def __len__(self) -> int:
        """Return the number of experiences in memory."""
        return len(self.experiences)
        
    def __iter__(self) -> Iterator[Experience]:
        """Iterate over all experiences in memory."""
        return iter(self.experiences.values())


def get_experience_tracker(
    storage_dir: Optional[Union[str, Path]] = None,
    max_cache_size: int = 100
) -> ExperienceTracker:
    """
    Get or create the singleton ExperienceTracker instance.
    
    Args:
        storage_dir (Union[str, Path], optional): Directory to store experiences
        max_cache_size (int, optional): Maximum number of experiences to keep in memory
        
    Returns:
        ExperienceTracker: The singleton ExperienceTracker instance
    """
    global _experience_tracker
    if _experience_tracker is None:
        _experience_tracker = ExperienceTracker(storage_dir, max_cache_size)
    return _experience_tracker 