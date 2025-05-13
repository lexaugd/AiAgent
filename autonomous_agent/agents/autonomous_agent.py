"""
Autonomous Agent implementation for the Autonomous Coding Agent.

This module provides the base AutonomousAgent class that implements the autonomous execution loop.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from loguru import logger
import threading

from models.llm_interface import LocalLLM
from memory.short_term import ShortTermMemory, get_memory
from tools.base import Tool
from agents.task import Goal, Task


class AutonomousAgent:
    """
    Base class for autonomous agents that implement the observe-plan-act-reflect cycle.
    """
    
    def __init__(
        self,
        name: str,
        llm: LocalLLM,
        tools: List[Tool],
        memory_id: Optional[str] = None,
        workspace: str = "."
    ):
        """
        Initialize the autonomous agent.
        
        Args:
            name (str): The name of the agent.
            llm (LocalLLM): The LLM to use for planning and reasoning.
            tools (List[Tool]): The tools available to the agent.
            memory_id (str, optional): The memory ID to use. Defaults to None.
            workspace (str, optional): The workspace directory. Defaults to ".".
        """
        self.name = name
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.memory = get_memory(memory_id or f"autonomous_{name}")
        
        # Task management
        self.goals = {}
        self.task_queue = []
        self.current_task = None
        
        # Execution state
        self.running = False
        self.paused = False
        self._stop_requested = False
        self._interrupt_checks = []
        
        # Context
        self.context = {
            "workspace": os.path.abspath(workspace),
            "allowed_directories": [os.path.abspath(workspace)],
            "permissions": ["file_read", "file_write"],  # Default permissions
            "agent_name": name
        }
        
        # Statistics
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "start_time": None,
            "total_runtime": 0,
            "total_actions": 0
        }
        
        logger.info(f"Initialized autonomous agent: {self.name}")
    
    def add_goal(self, goal: Goal) -> None:
        """
        Add a goal for the agent to work on.
        
        Args:
            goal (Goal): The goal to add.
        """
        self.goals[goal.id] = goal
        logger.info(f"Added goal: {goal.description}")
    
    def add_task(self, task: Task) -> None:
        """
        Add a task to the agent's queue.
        
        Args:
            task (Task): The task to add.
        """
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        logger.info(f"Added task with priority {task.priority}: {task.description}")
    
    def add_interrupt_check(self, check_func: Callable[[], bool]) -> None:
        """
        Add a function to check for interruptions.
        
        Args:
            check_func (Callable[[], bool]): Function that returns True if execution should be interrupted.
        """
        self._interrupt_checks.append(check_func)
    
    def start(self, background: bool = False) -> Optional[threading.Thread]:
        """
        Start the autonomous execution loop.
        
        Args:
            background (bool, optional): Whether to run in the background. Defaults to False.
            
        Returns:
            Optional[threading.Thread]: The thread if running in background, None otherwise.
        """
        if background:
            thread = threading.Thread(target=self._execution_loop)
            thread.daemon = True
            thread.start()
            return thread
        else:
            self._execution_loop()
            return None
    
    def stop(self) -> None:
        """Request the agent to stop execution."""
        self._stop_requested = True
        logger.info(f"Stop requested for agent: {self.name}")
    
    def pause(self) -> None:
        """Pause the agent execution."""
        self.paused = True
        logger.info(f"Paused agent: {self.name}")
    
    def resume(self) -> None:
        """Resume the agent execution after pausing."""
        self.paused = False
        logger.info(f"Resumed agent: {self.name}")
    
    def _execution_loop(self) -> None:
        """The main execution loop implementing the observe-plan-act-reflect cycle."""
        try:
            self.running = True
            self._stop_requested = False
            self.stats["start_time"] = time.time()
            
            # Run initialization
            self._initialize()
            
            # Main loop
            while self.running and not self._stop_requested:
                try:
                    # Handle pausing
                    if self.paused:
                        time.sleep(0.5)
                        continue
                    
                    # 1. Observe: Gather current state and context
                    observation = self._observe()
                    
                    # 2. Plan: Determine next actions if needed
                    if not self.current_task:
                        self.current_task = self._select_next_task()
                        
                    if not self.current_task:
                        # No tasks to perform, idle behavior
                        self._handle_idle()
                        time.sleep(1)  # Don't busy-wait
                        continue
                    
                    # Create a plan for the task
                    plan = self._create_plan(self.current_task, observation)
                    
                    # 3. Act: Execute the plan
                    result = self._execute_plan(plan)
                    
                    # 4. Reflect: Evaluate results and update state
                    self._reflect(self.current_task, plan, result)
                    
                    # Check if current task is complete
                    if self._is_task_complete(self.current_task, result):
                        self._finalize_task(self.current_task, result)
                        self.stats["tasks_completed"] += 1
                        self.current_task = None
                    else:
                        # Update the task with new information
                        self._update_task(self.current_task, result)
                        
                        # If we've reached max attempts, mark as failed and move on
                        if not self.current_task.increment_attempt():
                            logger.warning(f"Task failed after {self.current_task.attempts} attempts: {self.current_task.description}")
                            self.stats["tasks_failed"] += 1
                            self.current_task = None
                        
                except Exception as e:
                    self._handle_exception(e)
                    
                # Check for interruption or pause requests
                for check in self._interrupt_checks:
                    if check():
                        logger.info("Interrupt triggered by check function")
                        self._stop_requested = True
                        break
                
                # Sleep to prevent CPU hogging
                time.sleep(0.1)
        
        finally:
            # Cleanup and save state
            self.running = False
            self.stats["total_runtime"] = time.time() - (self.stats["start_time"] or time.time())
            self._save_state()
            logger.info(f"Agent {self.name} stopped. Stats: {json.dumps(self.stats, indent=2)}")
    
    def _initialize(self) -> None:
        """Initialize the agent before starting the execution loop."""
        logger.info(f"Initializing agent: {self.name}")
        
        # Load state from disk if available
        self._load_state()
        
        # Add a simple initialization message to memory
        self.memory.add_system_message(f"You are {self.name}, an autonomous agent focused on accomplishing tasks efficiently.")
    
    def _observe(self) -> Dict[str, Any]:
        """
        Gather information about the current state and context.
        
        Returns:
            Dict[str, Any]: Observations about the current state.
        """
        # Basic implementation, subclasses should extend this
        observation = {
            "timestamp": time.time(),
            "workspace": self.context["workspace"],
            "current_task": self.current_task.to_dict() if self.current_task else None,
            "task_queue_size": len(self.task_queue),
            "goals_count": len(self.goals),
            "goals_completed": sum(1 for goal in self.goals.values() if goal.completed),
            "available_tools": list(self.tools.keys())
        }
        
        return observation
    
    def _select_next_task(self) -> Optional[Task]:
        """
        Select the next task to work on from the queue.
        
        Returns:
            Optional[Task]: The next task to work on, or None if no tasks are available.
        """
        if not self.task_queue:
            return None
        
        # Sort by priority and select the highest
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        return self.task_queue.pop(0)
    
    def _create_plan(self, task: Task, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a plan for accomplishing the given task.
        
        Args:
            task (Task): The task to plan for.
            observation (Dict[str, Any]): Observations about the current state.
            
        Returns:
            Dict[str, Any]: The plan for accomplishing the task.
        """
        # This is a placeholder. Subclasses should implement this based on their specific needs.
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}" for name, tool in self.tools.items()
        ])
        
        prompt = f"""
You are {self.name}, an autonomous agent tasked with: {task.description}

Your task is part of the goal: {task.goal.description}

Currently available tools:
{tool_descriptions}

Based on the task and available tools, create a step-by-step plan to accomplish this task.
Each step should include:
1. The tool to use (if any)
2. The parameters to pass to the tool
3. The expected outcome of the step

The plan should be in JSON format with the following structure:
{{
    "reasoning": "Your step-by-step reasoning about how to approach this task",
    "steps": [
        {{
            "id": "step-1",
            "description": "Description of what this step accomplishes",
            "tool": "tool_name", 
            "args": {{"arg1": "value1", "arg2": "value2"}},
            "expected_outcome": "What you expect to happen from this step"
        }},
        ...
    ]
}}

If you need information not currently available, include steps to gather that information first.
"""
        
        # Add the current task context to the prompt if available
        if task.context:
            context_str = json.dumps(task.context, indent=2)
            prompt += f"\nAdditional context for this task:\n{context_str}\n"
        
        # Generate the plan using the LLM
        try:
            response = self.llm.generate(prompt=prompt)
            
            # Extract JSON from the response
            plan_json = self._extract_json(response)
            if not plan_json:
                logger.error(f"Failed to extract plan JSON from response: {response}")
                return {"error": "Failed to create plan", "steps": []}
            
            return plan_json
        except Exception as e:
            logger.exception(f"Error creating plan: {e}")
            return {"error": str(e), "steps": []}
    
    def _execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the given plan.
        
        Args:
            plan (Dict[str, Any]): The plan to execute.
            
        Returns:
            Dict[str, Any]: The results of executing the plan.
        """
        results = {
            "success": True,
            "steps": {},
            "error": None
        }
        
        # Check if the plan has steps
        if "error" in plan:
            results["success"] = False
            results["error"] = plan["error"]
            return results
        
        if "steps" not in plan or not plan["steps"]:
            results["success"] = False
            results["error"] = "Plan contains no steps"
            return results
        
        # Execute each step of the plan
        for step in plan["steps"]:
            step_id = step.get("id", f"step-{len(results['steps']) + 1}")
            tool_name = step.get("tool")
            args = step.get("args", {})
            
            logger.info(f"Executing step {step_id}: {step.get('description')} using tool {tool_name}")
            
            # Record the start of this step
            if self.current_task:
                self.current_task.add_step({
                    "id": step_id,
                    "description": step.get("description"),
                    "tool": tool_name,
                    "args": args,
                    "status": "started"
                })
            
            # Execute the step
            step_result = self._execute_step(tool_name, args)
            results["steps"][step_id] = step_result
            
            # Update the task with the step result
            if self.current_task:
                self.current_task.add_step({
                    "id": step_id,
                    "description": step.get("description"),
                    "tool": tool_name,
                    "args": args,
                    "status": "completed" if step_result.get("success", False) else "failed",
                    "result": step_result
                })
            
            # If the step failed, stop execution of the plan
            if not step_result.get("success", False):
                results["success"] = False
                results["error"] = f"Step {step_id} failed: {step_result.get('error', 'Unknown error')}"
                break
            
            # Increment the action counter
            self.stats["total_actions"] += 1
        
        return results
    
    def _execute_step(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step using the specified tool.
        
        Args:
            tool_name (str): The name of the tool to use.
            args (Dict[str, Any]): Arguments to pass to the tool.
            
        Returns:
            Dict[str, Any]: The result of executing the step.
        """
        # Check if the tool exists
        if not tool_name or tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }
        
        # Get the tool
        tool = self.tools[tool_name]
        
        # Execute the tool with the provided arguments and current context
        try:
            result = tool.execute(context=self.context, **args)
            return result
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}: {e}")
            return {
                "success": False,
                "error": f"Error executing tool: {str(e)}"
            }
    
    def _reflect(self, task: Task, plan: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Reflect on the execution results and update the agent's understanding.
        
        Args:
            task (Task): The task that was executed.
            plan (Dict[str, Any]): The plan that was executed.
            result (Dict[str, Any]): The results of the execution.
        """
        # Add the results to the task
        if task:
            task.add_result(result)
            
            # Update the task context with any new information
            new_context = {}
            
            # Extract useful information from the results
            for step_id, step_result in result.get("steps", {}).items():
                if step_result.get("success", False):
                    # Store successful results in the context for future reference
                    new_context[f"step_{step_id}"] = step_result.get("result", step_result)
            
            task.update_context(new_context)
        
        # Log the reflection
        if result.get("success", False):
            logger.info(f"Plan executed successfully for task: {task.description if task else 'Unknown'}")
        else:
            logger.warning(f"Plan execution failed for task: {task.description if task else 'Unknown'}, error: {result.get('error', 'Unknown')}")
    
    def _is_task_complete(self, task: Task, result: Dict[str, Any]) -> bool:
        """
        Determine if the task is complete based on the execution results.
        
        Args:
            task (Task): The task to check.
            result (Dict[str, Any]): The results of the execution.
            
        Returns:
            bool: True if the task is complete, False otherwise.
        """
        # If the plan failed, the task is not complete
        if not result.get("success", False):
            return False
        
        # This is a simple implementation. Subclasses should refine this logic.
        # For now, if the plan executed successfully, we consider the task complete.
        return result.get("success", False)
    
    def _finalize_task(self, task: Task, result: Dict[str, Any]) -> None:
        """
        Finalize a completed task.
        
        Args:
            task (Task): The task to finalize.
            result (Dict[str, Any]): The results of the execution.
        """
        task.mark_completed()
        logger.info(f"Task completed: {task.description}")
        
        # Add the completion to memory
        self.memory.add_system_message(f"Completed task: {task.description}")
    
    def _update_task(self, task: Task, result: Dict[str, Any]) -> None:
        """
        Update a task based on execution results.
        
        Args:
            task (Task): The task to update.
            result (Dict[str, Any]): The results of the execution.
        """
        # This is a placeholder. Subclasses might implement this.
        pass
    
    def _handle_idle(self) -> None:
        """Handle idle time when no tasks are available."""
        logger.debug(f"Agent {self.name} is idle (no tasks available)")
    
    def _handle_exception(self, exception: Exception) -> None:
        """
        Handle an exception during execution.
        
        Args:
            exception (Exception): The exception that occurred.
        """
        logger.exception(f"Exception in execution loop: {exception}")
        
        # If we have a current task, update it with the error
        if self.current_task:
            self.current_task.add_result({
                "success": False,
                "error": f"Exception: {str(exception)}"
            })
    
    def _save_state(self) -> None:
        """Save the agent's state to disk."""
        try:
            state_dir = Path(self.context["workspace"]) / ".agent_state"
            os.makedirs(state_dir, exist_ok=True)
            
            # Save goals
            goals_data = {goal_id: goal.to_dict() for goal_id, goal in self.goals.items()}
            with open(state_dir / "goals.json", "w") as f:
                json.dump(goals_data, f, indent=2)
            
            # Save task queue
            task_queue_data = [task.to_dict() for task in self.task_queue]
            with open(state_dir / "task_queue.json", "w") as f:
                json.dump(task_queue_data, f, indent=2)
            
            # Save current task if exists
            if self.current_task:
                with open(state_dir / "current_task.json", "w") as f:
                    json.dump(self.current_task.to_dict(), f, indent=2)
            
            # Save statistics
            with open(state_dir / "stats.json", "w") as f:
                json.dump(self.stats, f, indent=2)
            
            logger.info(f"Saved agent state to {state_dir}")
        except Exception as e:
            logger.exception(f"Error saving agent state: {e}")
    
    def _load_state(self) -> None:
        """Load the agent's state from disk."""
        try:
            state_dir = Path(self.context["workspace"]) / ".agent_state"
            if not state_dir.exists():
                logger.info(f"No saved state found at {state_dir}")
                return
            
            # Load goals
            goals_path = state_dir / "goals.json"
            if goals_path.exists():
                with open(goals_path, "r") as f:
                    goals_data = json.load(f)
                    for goal_id, goal_data in goals_data.items():
                        self.goals[goal_id] = Goal.from_dict(goal_data)
            
            # Load task queue
            task_queue_path = state_dir / "task_queue.json"
            if task_queue_path.exists():
                with open(task_queue_path, "r") as f:
                    task_queue_data = json.load(f)
                    for task_data in task_queue_data:
                        task = Task.from_dict(task_data, self.goals)
                        if task:
                            self.task_queue.append(task)
            
            # Load current task if exists
            current_task_path = state_dir / "current_task.json"
            if current_task_path.exists():
                with open(current_task_path, "r") as f:
                    current_task_data = json.load(f)
                    self.current_task = Task.from_dict(current_task_data, self.goals)
            
            # Load statistics
            stats_path = state_dir / "stats.json"
            if stats_path.exists():
                with open(stats_path, "r") as f:
                    self.stats.update(json.load(f))
            
            logger.info(f"Loaded agent state from {state_dir}")
        except Exception as e:
            logger.exception(f"Error loading agent state: {e}")
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text that might contain other content.
        
        Args:
            text (str): Text that might contain JSON.
            
        Returns:
            Optional[Dict[str, Any]]: The extracted JSON, or None if no valid JSON found.
        """
        # Look for code blocks that might contain JSON
        import re
        json_pattern = r"```(?:json)?\s*(.*?)```"
        match = re.search(json_pattern, text, re.DOTALL)
        
        if match:
            # Try to parse the JSON from the code block
            try:
                return json.loads(match.group(1))
            except:
                pass
        
        # Try to find JSON-like content directly
        try:
            # Look for content between curly braces
            curly_pattern = r"(\{.*\})"
            match = re.search(curly_pattern, text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except:
            pass
        
        # As a last resort, try to parse the entire text as JSON
        try:
            return json.loads(text)
        except:
            return None 