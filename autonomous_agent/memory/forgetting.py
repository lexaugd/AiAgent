"""
Forgetting mechanisms for the Autonomous Coding Agent's memory system.

This module implements various forgetting curves and memory retention algorithms
to optimize memory usage and relevance over time.
"""

import math
import time
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from loguru import logger

from memory.types import MemoryPriority, MemoryMetadata, ExtendedMemoryItem


class ForgettingCurve:
    """
    Base class for forgetting curve algorithms.
    
    Forgetting curves calculate memory retention over time.
    """
    
    def calculate_retention(self, 
                           time_elapsed: float, 
                           strength: float, 
                           access_count: int) -> float:
        """
        Calculate the memory retention factor.
        
        Args:
            time_elapsed (float): Time elapsed since the memory was created or reinforced.
            strength (float): Strength factor of the memory.
            access_count (int): Number of times the memory has been accessed.
            
        Returns:
            float: Retention factor between 0 and 1.
        """
        raise NotImplementedError("Subclasses must implement calculate_retention")


class EbbinghausForgettingCurve(ForgettingCurve):
    """
    Implementation of the Ebbinghaus forgetting curve.
    
    R = e^(-t/S) where:
    - R is the retention factor (0 to 1)
    - t is the time elapsed
    - S is the strength of the memory
    """
    
    def calculate_retention(self, 
                           time_elapsed: float, 
                           strength: float, 
                           access_count: int) -> float:
        """
        Calculate the memory retention factor using the Ebbinghaus curve.
        
        Args:
            time_elapsed (float): Time elapsed in seconds since last access.
            strength (float): Strength factor of the memory.
            access_count (int): Number of times the memory has been accessed.
            
        Returns:
            float: Retention factor between 0 and 1.
        """
        # Convert time to days for a more human-like scale
        days_elapsed = time_elapsed / (24 * 60 * 60)
        
        # Apply the formula R = e^(-t/S)
        # The strength factor increases with access_count
        adjusted_strength = strength * (1 + 0.2 * access_count)
        
        # Calculate retention
        retention = math.exp(-days_elapsed / adjusted_strength)
        
        # Ensure retention is between 0 and 1
        return max(0.0, min(1.0, retention))


class PowerLawForgettingCurve(ForgettingCurve):
    """
    Implementation of a power law forgetting curve.
    
    R = 1 / (1 + a*t^b) where:
    - R is the retention factor (0 to 1)
    - t is the time elapsed
    - a and b are parameters that control the shape of the curve
    """
    
    def __init__(self, a: float = 0.1, b: float = 0.5):
        """
        Initialize the power law forgetting curve.
        
        Args:
            a (float): Parameter controlling the initial forgetting rate.
            b (float): Parameter controlling the shape of the curve.
        """
        self.a = a
        self.b = b
    
    def calculate_retention(self, 
                           time_elapsed: float, 
                           strength: float, 
                           access_count: int) -> float:
        """
        Calculate the memory retention factor using a power law curve.
        
        Args:
            time_elapsed (float): Time elapsed in seconds since last access.
            strength (float): Strength factor of the memory.
            access_count (int): Number of times the memory has been accessed.
            
        Returns:
            float: Retention factor between 0 and 1.
        """
        # Convert time to days and adjust by strength
        days_elapsed = time_elapsed / (24 * 60 * 60) / strength
        
        # Apply access count effect (decreases rate of forgetting)
        adjustment = 1.0 / (1.0 + 0.1 * access_count)
        adjusted_a = self.a * adjustment
        
        # Calculate retention using power law
        if days_elapsed <= 0:
            return 1.0
        else:
            retention = 1.0 / (1.0 + adjusted_a * (days_elapsed ** self.b))
            
        # Ensure retention is between 0 and 1
        return max(0.0, min(1.0, retention))


@dataclass
class ForgettingParams:
    """Parameters for the forgetting mechanism."""
    
    # Base retention thresholds for different priority levels
    critical_threshold: float = 0.0  # Never forget critical memories
    high_threshold: float = 0.2
    medium_threshold: float = 0.4
    low_threshold: float = 0.6
    
    # Retention boost factors based on memory attributes
    access_boost: float = 0.05  # Boost per access
    recency_boost_days: float = 7.0  # Recent memories get a boost
    recency_boost_factor: float = 0.2  # Amount of recency boost
    
    # Minimum age before forgetting can occur (in days)
    min_age_days: float = 1.0


class MemoryForgetting:
    """
    Implements forgetting mechanisms for memory optimization.
    """
    
    def __init__(self, curve: Optional[ForgettingCurve] = None, params: Optional[ForgettingParams] = None):
        """
        Initialize the memory forgetting system.
        
        Args:
            curve (ForgettingCurve, optional): The forgetting curve to use.
            params (ForgettingParams, optional): Parameters for the forgetting mechanism.
        """
        self.curve = curve or EbbinghausForgettingCurve()
        self.params = params or ForgettingParams()
    
    def should_forget(self, item: Union[MemoryMetadata, ExtendedMemoryItem], 
                     current_time: Optional[float] = None) -> Tuple[bool, float]:
        """
        Determine if a memory item should be forgotten.
        
        Args:
            item (Union[MemoryMetadata, ExtendedMemoryItem]): The memory item to evaluate.
            current_time (float, optional): Current time for calculation. Default is time.time().
            
        Returns:
            Tuple[bool, float]: (should_forget, retention_factor)
        """
        # Extract metadata if ExtendedMemoryItem
        metadata = item.metadata if hasattr(item, 'metadata') else item
        
        # Set current time if not provided
        if current_time is None:
            current_time = time.time()
        
        # Get the most recent access time
        last_accessed = metadata.last_accessed or metadata.created_at
        
        # Calculate time elapsed since creation and last access
        time_since_creation = current_time - metadata.created_at
        time_since_access = current_time - last_accessed
        
        # Check minimum age - don't forget very new memories
        min_age_seconds = self.params.min_age_days * 24 * 60 * 60
        if time_since_creation < min_age_seconds:
            return False, 1.0
        
        # Base strength based on priority
        priority_strength_map = {
            MemoryPriority.CRITICAL: 10.0,
            MemoryPriority.HIGH: 5.0,
            MemoryPriority.MEDIUM: 2.0,
            MemoryPriority.LOW: 1.0
        }
        
        # Get memory strength from priority
        base_strength = priority_strength_map.get(metadata.priority, 2.0)
        
        # Apply recency boost if the memory was accessed recently
        recency_boost = 0.0
        recency_window = self.params.recency_boost_days * 24 * 60 * 60
        if time_since_access < recency_window:
            recency_factor = 1.0 - (time_since_access / recency_window)
            recency_boost = self.params.recency_boost_factor * recency_factor
        
        # Calculate final strength
        strength = base_strength * (1.0 + recency_boost)
        
        # Calculate retention using the forgetting curve
        retention = self.curve.calculate_retention(
            time_elapsed=time_since_access,
            strength=strength,
            access_count=metadata.access_count
        )
        
        # Get threshold based on priority
        threshold_map = {
            MemoryPriority.CRITICAL: self.params.critical_threshold,
            MemoryPriority.HIGH: self.params.high_threshold,
            MemoryPriority.MEDIUM: self.params.medium_threshold,
            MemoryPriority.LOW: self.params.low_threshold
        }
        
        threshold = threshold_map.get(metadata.priority, self.params.medium_threshold)
        
        # Determine if the memory should be forgotten
        should_forget = retention < threshold
        
        return should_forget, retention
    
    def apply_forgetting(self, 
                        items: List[Union[MemoryMetadata, ExtendedMemoryItem]], 
                        max_items: Optional[int] = None) -> List[Union[MemoryMetadata, ExtendedMemoryItem]]:
        """
        Apply forgetting mechanism to a list of memory items.
        
        Args:
            items (List[Union[MemoryMetadata, ExtendedMemoryItem]]): The items to evaluate.
            max_items (int, optional): Maximum items to keep. If None, only applies threshold-based forgetting.
            
        Returns:
            List[Union[MemoryMetadata, ExtendedMemoryItem]]: The items that should be kept.
        """
        # Current time for consistent calculations
        current_time = time.time()
        
        # Calculate retention for all items
        items_with_retention = []
        for item in items:
            should_forget, retention = self.should_forget(item, current_time)
            
            # Skip items that should be forgotten by threshold
            if should_forget:
                continue
            
            items_with_retention.append((item, retention))
        
        # If max_items is specified and we have more items than allowed,
        # sort by retention and keep only the top max_items
        if max_items is not None and len(items_with_retention) > max_items:
            # Sort by retention (highest first)
            items_with_retention.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only the top max_items
            items_with_retention = items_with_retention[:max_items]
        
        # Extract and return only the items
        return [item for item, _ in items_with_retention]
    
    def reinforcement_effect(self, 
                           item: Union[MemoryMetadata, ExtendedMemoryItem], 
                           context: Optional[str] = None,
                           agent_id: Optional[str] = None,
                           relevance_score: Optional[float] = None) -> None:
        """
        Apply reinforcement to a memory item when it's accessed.
        
        Args:
            item (Union[MemoryMetadata, ExtendedMemoryItem]): The memory item to reinforce.
            context (str, optional): The context of the access.
            agent_id (str, optional): The ID of the agent accessing the memory.
            relevance_score (float, optional): The relevance score of the memory.
        """
        if hasattr(item, 'record_access'):
            item.record_access(context, agent_id, relevance_score)
        elif hasattr(item, 'metadata') and hasattr(item.metadata, 'record_access'):
            item.metadata.record_access(context, agent_id, relevance_score)


class MemoryConsolidation:
    """
    Implements memory consolidation strategies to optimize long-term storage.
    """
    
    def __init__(self, forgetting: Optional[MemoryForgetting] = None):
        """
        Initialize the memory consolidation system.
        
        Args:
            forgetting (MemoryForgetting, optional): Forgetting mechanism to use.
        """
        self.forgetting = forgetting or MemoryForgetting()
    
    def consolidate_memories(self, 
                           items: List[ExtendedMemoryItem], 
                           similarity_threshold: float = 0.85,
                           similarity_fn: Optional[Callable[[ExtendedMemoryItem, ExtendedMemoryItem], float]] = None) -> List[ExtendedMemoryItem]:
        """
        Consolidate similar memories to reduce redundancy.
        
        Args:
            items (List[ExtendedMemoryItem]): The items to consolidate.
            similarity_threshold (float): Threshold for considering items similar.
            similarity_fn (Callable, optional): Function to calculate similarity between items.
            
        Returns:
            List[ExtendedMemoryItem]: The consolidated items.
        """
        # If no items or only one item, no consolidation needed
        if len(items) <= 1:
            return items
        
        # Apply forgetting first to remove low-retention items
        kept_items = self.forgetting.apply_forgetting(items)
        
        # If we don't have a similarity function, we can't consolidate
        if similarity_fn is None:
            return kept_items
        
        # Find similar items and consolidate them
        consolidated = []
        processed_ids = set()
        
        for i, item1 in enumerate(kept_items):
            if item1.item_id in processed_ids:
                continue
            
            similar_items = []
            
            for j, item2 in enumerate(kept_items):
                if i == j or item2.item_id in processed_ids:
                    continue
                
                # Calculate similarity
                similarity = similarity_fn(item1, item2)
                
                # If similar enough, add to similar items
                if similarity >= similarity_threshold:
                    similar_items.append(item2)
            
            # If we found similar items, consolidate them
            if similar_items:
                consolidated_item = self._merge_items(item1, similar_items)
                consolidated.append(consolidated_item)
                
                # Mark all these items as processed
                processed_ids.add(item1.item_id)
                for item in similar_items:
                    processed_ids.add(item.item_id)
            else:
                # No similar items, just keep this one
                consolidated.append(item1)
                processed_ids.add(item1.item_id)
        
        return consolidated
    
    def _merge_items(self, 
                    primary: ExtendedMemoryItem, 
                    similar: List[ExtendedMemoryItem]) -> ExtendedMemoryItem:
        """
        Merge similar items into a consolidated item.
        
        Args:
            primary (ExtendedMemoryItem): The primary item to merge others into.
            similar (List[ExtendedMemoryItem]): Similar items to merge.
            
        Returns:
            ExtendedMemoryItem: The consolidated item.
        """
        # Create a new metadata with combined information
        combined_metadata = MemoryMetadata(
            item_type=primary.metadata.item_type,
            created_at=primary.metadata.created_at,
            source=primary.metadata.source,
            priority=max([primary.metadata.priority] + [item.metadata.priority for item in similar]),
            last_accessed=max([primary.metadata.last_accessed or 0] + 
                             [item.metadata.last_accessed or 0 for item in similar]),
            access_count=primary.metadata.access_count + sum(item.metadata.access_count for item in similar),
            access_history=primary.metadata.access_history.copy(),
            associations=primary.metadata.associations.copy(),
            custom=primary.metadata.custom.copy()
        )
        
        # Add access histories from similar items
        for item in similar:
            combined_metadata.access_history.extend(item.metadata.access_history)
        
        # Add unique associations from similar items
        existing_target_ids = {assoc.target_id for assoc in combined_metadata.associations}
        for item in similar:
            for assoc in item.metadata.associations:
                if assoc.target_id not in existing_target_ids:
                    combined_metadata.associations.append(assoc)
                    existing_target_ids.add(assoc.target_id)
        
        # Add references to the merged items in custom metadata
        merged_ids = [item.item_id for item in similar]
        if 'merged_items' not in combined_metadata.custom:
            combined_metadata.custom['merged_items'] = []
        combined_metadata.custom['merged_items'].extend(merged_ids)
        
        # Create the consolidated item
        return ExtendedMemoryItem(
            content=primary.content,  # Keep the primary content
            metadata=combined_metadata,
            item_id=primary.item_id,
            embedding=primary.embedding
        ) 