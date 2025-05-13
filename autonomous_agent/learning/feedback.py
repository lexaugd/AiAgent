"""
Feedback processing for the Autonomous Coding Agent.

This module provides functionality to process and incorporate user feedback.
"""

import time
import json
import os
from typing import Dict, List, Any, Optional, Union, Iterator
from pathlib import Path
from loguru import logger

from .types import Feedback, FeedbackType, Experience
from .experience import get_experience_tracker

# Singleton instance
_feedback_processor = None

class FeedbackProcessor:
    """Class to process and incorporate user feedback."""
    
    def __init__(
        self, 
        storage_dir: Optional[Union[str, Path]] = None,
        max_cache_size: int = 50
    ):
        """
        Initialize the feedback processor.
        
        Args:
            storage_dir (Union[str, Path], optional): Directory to store feedback
            max_cache_size (int, optional): Maximum number of feedback items to keep in memory
        """
        if storage_dir is None:
            storage_dir = Path.home() / ".autonomous_agent" / "feedback"
        elif isinstance(storage_dir, str):
            storage_dir = Path(storage_dir)
            
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size = max_cache_size
        self.feedback_items = {}  # In-memory cache of feedback items
        self._load_feedback_items()
        
        # Get experience tracker instance
        self.experience_tracker = get_experience_tracker()
        
        logger.info(f"Initialized FeedbackProcessor with storage directory: {storage_dir}")
        
    def process_feedback(self, feedback: Feedback) -> str:
        """
        Process a new feedback item.
        
        Args:
            feedback (Feedback): The feedback to process
            
        Returns:
            str: The ID of the processed feedback
        """
        # Save to memory cache
        self.feedback_items[feedback.feedback_id] = feedback
        
        # Save to disk
        feedback_path = self.storage_dir / f"{feedback.feedback_id}.json"
        with open(feedback_path, "w") as f:
            json.dump(feedback.to_dict(), f, indent=2)
            
        # If cache is too large, remove oldest feedback items
        if len(self.feedback_items) > self.max_cache_size:
            oldest_id = sorted(
                self.feedback_items.keys(), 
                key=lambda x: self.feedback_items[x].timestamp
            )[0]
            del self.feedback_items[oldest_id]
            
        # Update the associated experience if applicable
        if feedback.target_response_id:
            self._update_associated_experience(feedback)
            
        logger.debug(f"Processed feedback: {feedback.feedback_id} ({feedback.feedback_type.value})")
        return feedback.feedback_id
        
    def get_feedback(self, feedback_id: str) -> Optional[Feedback]:
        """
        Get a feedback item by ID.
        
        Args:
            feedback_id (str): The ID of the feedback to retrieve
            
        Returns:
            Optional[Feedback]: The feedback if found, None otherwise
        """
        # Try to get from memory cache
        if feedback_id in self.feedback_items:
            return self.feedback_items[feedback_id]
        
        # Try to load from disk
        feedback_path = self.storage_dir / f"{feedback_id}.json"
        if feedback_path.exists():
            with open(feedback_path, "r") as f:
                data = json.load(f)
                feedback = Feedback.from_dict(data)
                self.feedback_items[feedback_id] = feedback
                return feedback
                
        return None
        
    def delete_feedback(self, feedback_id: str) -> bool:
        """
        Delete a feedback item.
        
        Args:
            feedback_id (str): The ID of the feedback to delete
            
        Returns:
            bool: True if the feedback was deleted, False otherwise
        """
        # Remove from memory cache
        if feedback_id in self.feedback_items:
            del self.feedback_items[feedback_id]
        
        # Remove from disk
        feedback_path = self.storage_dir / f"{feedback_id}.json"
        if feedback_path.exists():
            feedback_path.unlink()
            logger.debug(f"Deleted feedback: {feedback_id}")
            return True
            
        logger.warning(f"Cannot delete feedback: {feedback_id} not found")
        return False
        
    def get_feedback_for_experience(self, experience_id: str) -> List[Feedback]:
        """
        Get all feedback items for a specific experience.
        
        Args:
            experience_id (str): The ID of the experience
            
        Returns:
            List[Feedback]: Feedback items for the experience
        """
        # Load all feedback items to ensure we have the latest data
        self._load_feedback_items()
        
        # Filter feedback items by experience ID
        feedback_items = []
        for feedback in self.feedback_items.values():
            if feedback.target_response_id == experience_id:
                feedback_items.append(feedback)
                
        return feedback_items
        
    def list_feedback(
        self, 
        feedback_type: Optional[Union[FeedbackType, str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[Feedback]:
        """
        List feedback items with optional filtering.
        
        Args:
            feedback_type (Union[FeedbackType, str], optional): Filter by feedback type
            start_time (float, optional): Only include feedback after this time
            end_time (float, optional): Only include feedback before this time
            limit (int, optional): Maximum number of feedback items to return
            
        Returns:
            List[Feedback]: Matching feedback items
        """
        # Convert string to enum if necessary
        if isinstance(feedback_type, str):
            try:
                feedback_type = FeedbackType[feedback_type.upper()]
            except KeyError:
                feedback_type = None
                
        # Load all feedback items to ensure we have the latest data
        self._load_feedback_items()
        
        # Filter feedback items
        filtered_feedback = []
        for fb in self.feedback_items.values():
            if feedback_type and fb.feedback_type != feedback_type:
                continue
                
            if start_time and fb.timestamp < start_time:
                continue
                
            if end_time and fb.timestamp > end_time:
                continue
                
            filtered_feedback.append(fb)
            
        # Sort by timestamp (newest first)
        filtered_feedback.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            filtered_feedback = filtered_feedback[:limit]
            
        return filtered_feedback
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the processed feedback.
        
        Returns:
            Dict[str, Any]: Statistics about the feedback
        """
        # Load all feedback items to ensure we have the latest data
        self._load_feedback_items()
        
        # Count feedback by type
        type_counts = {}
        for fb_type in FeedbackType:
            type_counts[fb_type.value] = 0
            
        # Count feedback by rating
        rating_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "unknown": 0}
        
        # Calculate time ranges
        timestamps = []
        
        # Calculate average rating
        ratings = []
        
        for fb in self.feedback_items.values():
            type_counts[fb.feedback_type.value] += 1
            
            if fb.rating:
                ratings.append(fb.rating)
                rating_key = int(fb.rating)
                if rating_key in rating_counts:
                    rating_counts[rating_key] += 1
                else:
                    rating_counts["unknown"] += 1
            else:
                rating_counts["unknown"] += 1
            
            timestamps.append(fb.timestamp)
            
        # Calculate time range
        time_range = {
            "oldest": min(timestamps) if timestamps else None,
            "newest": max(timestamps) if timestamps else None,
            "total_days": (max(timestamps) - min(timestamps)) / (60 * 60 * 24) if timestamps else 0
        }
        
        return {
            "total_feedback": len(self.feedback_items),
            "type_counts": type_counts,
            "rating_counts": rating_counts,
            "average_rating": sum(ratings) / len(ratings) if ratings else 0,
            "time_range": time_range
        }
        
    def _update_associated_experience(self, feedback: Feedback) -> bool:
        """
        Update the experience associated with this feedback.
        
        Args:
            feedback (Feedback): The feedback to incorporate
            
        Returns:
            bool: True if the experience was updated, False otherwise
        """
        if not feedback.target_response_id:
            return False
            
        experience_id = feedback.target_response_id
        
        # Prepare feedback data for the experience
        feedback_data = {
            "feedback_id": feedback.feedback_id,
            "content": feedback.content,
            "type": feedback.feedback_type.value,
            "rating": feedback.rating,
            "timestamp": feedback.timestamp
        }
        
        # Update the experience with the feedback
        return self.experience_tracker.update_experience(
            experience_id=experience_id,
            feedback={feedback.feedback_id: feedback_data}
        )
        
    def _load_feedback_items(self):
        """Load feedback items from disk into memory cache."""
        # Clear existing cache
        self.feedback_items = {}
        
        # Load from disk
        for feedback_file in self.storage_dir.glob("*.json"):
            try:
                with open(feedback_file, "r") as f:
                    data = json.load(f)
                    feedback = Feedback.from_dict(data)
                    self.feedback_items[feedback.feedback_id] = feedback
            except Exception as e:
                logger.error(f"Error loading feedback file {feedback_file}: {e}")
                
        logger.debug(f"Loaded {len(self.feedback_items)} feedback items from disk")
        
    def analyze_feedback_trends(self) -> Dict[str, Any]:
        """
        Analyze trends in the feedback data.
        
        Returns:
            Dict[str, Any]: Analysis of feedback trends
        """
        # Load all feedback items
        self._load_feedback_items()
        
        # Get all feedback sorted by time
        all_feedback = sorted(self.feedback_items.values(), key=lambda x: x.timestamp)
        
        if not all_feedback:
            return {"trends": "No feedback data available for analysis"}
            
        # Analyze rating trends over time (last 7 days, last 30 days, all time)
        now = time.time()
        day_seconds = 60 * 60 * 24
        
        # Last 7 days
        recent_feedback = [fb for fb in all_feedback if fb.timestamp >= now - (7 * day_seconds)]
        recent_ratings = [fb.rating for fb in recent_feedback if fb.rating is not None]
        
        # Last 30 days
        month_feedback = [fb for fb in all_feedback if fb.timestamp >= now - (30 * day_seconds)]
        month_ratings = [fb.rating for fb in month_feedback if fb.rating is not None]
        
        # All time
        all_ratings = [fb.rating for fb in all_feedback if fb.rating is not None]
        
        # Calculate averages
        recent_avg = sum(recent_ratings) / len(recent_ratings) if recent_ratings else 0
        month_avg = sum(month_ratings) / len(month_ratings) if month_ratings else 0
        all_time_avg = sum(all_ratings) / len(all_ratings) if all_ratings else 0
        
        return {
            "total_feedback": len(all_feedback),
            "rating_trends": {
                "last_7_days": {
                    "count": len(recent_feedback),
                    "average_rating": recent_avg
                },
                "last_30_days": {
                    "count": len(month_feedback),
                    "average_rating": month_avg
                },
                "all_time": {
                    "count": len(all_feedback),
                    "average_rating": all_time_avg
                }
            },
            "trend_direction": "improving" if recent_avg > month_avg else "declining" if recent_avg < month_avg else "stable"
        }
        
    def __len__(self) -> int:
        """Return the number of feedback items in memory."""
        return len(self.feedback_items)
        
    def __iter__(self) -> Iterator[Feedback]:
        """Iterate over all feedback items in memory."""
        return iter(self.feedback_items.values())


def get_feedback_processor(
    storage_dir: Optional[Union[str, Path]] = None,
    max_cache_size: int = 50
) -> FeedbackProcessor:
    """
    Get or create the singleton FeedbackProcessor instance.
    
    Args:
        storage_dir (Union[str, Path], optional): Directory to store feedback
        max_cache_size (int, optional): Maximum number of feedback items to keep in memory
        
    Returns:
        FeedbackProcessor: The singleton FeedbackProcessor instance
    """
    global _feedback_processor
    if _feedback_processor is None:
        _feedback_processor = FeedbackProcessor(storage_dir, max_cache_size)
    return _feedback_processor 