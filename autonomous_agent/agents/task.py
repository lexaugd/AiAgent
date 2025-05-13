"""
Task and Goal implementation for the Autonomous Coding Agent.

This module provides the Task and Goal classes that represent objectives for the autonomous agent.
"""

import time
import uuid
from typing import Dict, List, Any, Optional
from loguru import logger


class Goal:
    """
    Represents a high-level goal for the agent.
    """
    
    def __init__(self, description: str, success_criteria: List[str]):
        """
        Initialize a goal.
        
        Args:
            description (str): Description of the goal.
            success_criteria (List[str]): Criteria for determining if the goal is completed.
        """
        self.id = str(uuid.uuid4())
        self.description = description
        self.success_criteria = success_criteria
        self.completed = False
        self.created_at = time.time()
        self.completed_at = None
        self.tasks = []
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the goal to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: The goal as a dictionary.
        """
        return {
            "id": self.id,
            "description": self.description,
            "success_criteria": self.success_criteria,
            "completed": self.completed,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "tasks": [task.id for task in self.tasks]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Goal':
        """
        Create a goal from a dictionary.
        
        Args:
            data (Dict[str, Any]): The dictionary containing goal data.
            
        Returns:
            Goal: The created goal.
        """
        goal = cls(
            description=data.get("description", ""),
            success_criteria=data.get("success_criteria", [])
        )
        goal.id = data.get("id", goal.id)
        goal.completed = data.get("completed", False)
        goal.created_at = data.get("created_at", goal.created_at)
        goal.completed_at = data.get("completed_at")
        return goal
    
    def mark_completed(self) -> None:
        """Mark the goal as completed."""
        self.completed = True
        self.completed_at = time.time()
        logger.info(f"Goal completed: {self.description}")


class Task:
    """
    Represents a specific task to accomplish a goal.
    """
    
    def __init__(self, description: str, goal: Goal, priority: int = 1):
        """
        Initialize a task.
        
        Args:
            description (str): Description of the task.
            goal (Goal): The goal this task is associated with.
            priority (int, optional): The priority of the task (higher = more important). Defaults to 1.
        """
        self.id = str(uuid.uuid4())
        self.description = description
        self.goal = goal
        self.priority = priority
        self.completed = False
        self.created_at = time.time()
        self.completed_at = None
        self.steps = []
        self.results = []
        self.attempts = 0
        self.max_attempts = 3
        self.context = {}
        
        # Add this task to the goal
        if goal not in goal.tasks:
            goal.tasks.append(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: The task as a dictionary.
        """
        return {
            "id": self.id,
            "description": self.description,
            "goal_id": self.goal.id,
            "priority": self.priority,
            "completed": self.completed,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "steps": self.steps,
            "results": self.results,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "context": self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], goals: Dict[str, Goal]) -> Optional['Task']:
        """
        Create a task from a dictionary.
        
        Args:
            data (Dict[str, Any]): The dictionary containing task data.
            goals (Dict[str, Goal]): Dictionary of goals indexed by ID.
            
        Returns:
            Optional[Task]: The created task, or None if the goal is not found.
        """
        goal_id = data.get("goal_id")
        if not goal_id or goal_id not in goals:
            logger.error(f"Cannot create task: goal with ID {goal_id} not found")
            return None
        
        task = cls(
            description=data.get("description", ""),
            goal=goals[goal_id],
            priority=data.get("priority", 1)
        )
        task.id = data.get("id", task.id)
        task.completed = data.get("completed", False)
        task.created_at = data.get("created_at", task.created_at)
        task.completed_at = data.get("completed_at")
        task.steps = data.get("steps", [])
        task.results = data.get("results", [])
        task.attempts = data.get("attempts", 0)
        task.max_attempts = data.get("max_attempts", 3)
        task.context = data.get("context", {})
        return task
    
    def mark_completed(self) -> None:
        """Mark the task as completed."""
        self.completed = True
        self.completed_at = time.time()
        logger.info(f"Task completed: {self.description}")
        
        # Check if all tasks for the goal are completed
        if all(task.completed for task in self.goal.tasks):
            self.goal.mark_completed()
    
    def increment_attempt(self) -> bool:
        """
        Increment the attempt counter and check if max attempts reached.
        
        Returns:
            bool: True if max attempts not exceeded, False otherwise.
        """
        self.attempts += 1
        if self.attempts >= self.max_attempts:
            logger.warning(f"Max attempts reached for task: {self.description}")
            return False
        return True
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """
        Add a result to the task.
        
        Args:
            result (Dict[str, Any]): The result to add.
        """
        self.results.append({
            "timestamp": time.time(),
            "attempt": self.attempts,
            **result
        })
    
    def add_step(self, step: Dict[str, Any]) -> None:
        """
        Add a step to the task.
        
        Args:
            step (Dict[str, Any]): The step to add.
        """
        self.steps.append({
            "timestamp": time.time(),
            **step
        })
    
    def update_context(self, updates: Dict[str, Any]) -> None:
        """
        Update the task context.
        
        Args:
            updates (Dict[str, Any]): The updates to apply to the context.
        """
        self.context.update(updates) 