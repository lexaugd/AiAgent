#!/usr/bin/env python3
"""
Test script for the learning system components.

This script runs tests to verify the correct functioning of the learning components:
- Experience tracking
- Feedback processing
- Knowledge extraction
- Self-reflection
- Learning manager
"""

import sys
import os
from pathlib import Path
import unittest
import tempfile
import shutil
import time
import json
from typing import Dict, List, Any

# Add paths to make imports work
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

from learning.types import Experience, Feedback, KnowledgeItem, ReflectionResult
from learning.types import ExperienceType, FeedbackType, KnowledgeType
from learning.experience import ExperienceTracker
from learning.feedback import FeedbackProcessor
from learning.extraction import KnowledgeExtractor
from learning.reflection import Reflector
from learning.manager import LearningManager


class TestExperienceTracker(unittest.TestCase):
    """Tests for the ExperienceTracker class."""

    def setUp(self):
        """Set up a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.experience_tracker = ExperienceTracker(storage_dir=self.temp_dir)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_record_and_get_experience(self):
        """Test recording and retrieving experiences."""
        # Create and record an experience
        experience = Experience(
            context="Test context",
            query="Test query",
            response="Test response",
            experience_type=ExperienceType.CODE_EXPLANATION
        )
        
        experience_id = self.experience_tracker.record_experience(experience)
        
        # Retrieve the experience
        retrieved_experience = self.experience_tracker.get_experience(experience_id)
        
        # Verify the retrieved experience
        self.assertIsNotNone(retrieved_experience)
        self.assertEqual(retrieved_experience.context, "Test context")
        self.assertEqual(retrieved_experience.query, "Test query")
        self.assertEqual(retrieved_experience.response, "Test response")
        self.assertEqual(retrieved_experience.experience_type, ExperienceType.CODE_EXPLANATION)

    def test_update_experience(self):
        """Test updating an experience."""
        # Create and record an experience
        experience = Experience(
            context="Test context",
            query="Test query",
            response="Test response",
            experience_type=ExperienceType.CODE_EXPLANATION
        )
        
        experience_id = self.experience_tracker.record_experience(experience)
        
        # Update the experience
        feedback = {"feedback_id_1": {"content": "Great explanation", "type": "confirmation"}}
        self.experience_tracker.update_experience(
            experience_id=experience_id,
            outcome="success",
            feedback=feedback,
            metadata_updates={"tag": "updated"}
        )
        
        # Retrieve the updated experience
        updated_experience = self.experience_tracker.get_experience(experience_id)
        
        # Verify the updates
        self.assertEqual(updated_experience.outcome, "success")
        self.assertEqual(updated_experience.feedback, feedback)
        self.assertEqual(updated_experience.metadata["tag"], "updated")

    def test_list_experiences(self):
        """Test listing experiences with filters."""
        # Create and record experiences with different types
        for exp_type in [ExperienceType.CODE_EXPLANATION, ExperienceType.CODE_GENERATION, ExperienceType.ERROR_RESOLUTION]:
            experience = Experience(
                context=f"Context for {exp_type.value}",
                query=f"Query for {exp_type.value}",
                response=f"Response for {exp_type.value}",
                experience_type=exp_type
            )
            self.experience_tracker.record_experience(experience)
            
        # List experiences by type
        code_explanations = self.experience_tracker.list_experiences(
            experience_type=ExperienceType.CODE_EXPLANATION
        )
        
        # Verify the filtered results
        self.assertEqual(len(code_explanations), 1)
        self.assertEqual(code_explanations[0].experience_type, ExperienceType.CODE_EXPLANATION)
        
        # List all experiences
        all_experiences = self.experience_tracker.list_experiences()
        self.assertEqual(len(all_experiences), 3)


class TestFeedbackProcessor(unittest.TestCase):
    """Tests for the FeedbackProcessor class."""

    def setUp(self):
        """Set up a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.feedback_processor = FeedbackProcessor(storage_dir=self.temp_dir)
        
        # Also create an experience tracker for testing integration
        self.experience_dir = tempfile.mkdtemp()
        self.experience_tracker = ExperienceTracker(storage_dir=self.experience_dir)

    def tearDown(self):
        """Clean up the temporary directories."""
        shutil.rmtree(self.temp_dir)
        shutil.rmtree(self.experience_dir)

    def test_process_and_get_feedback(self):
        """Test processing and retrieving feedback."""
        # Create and process feedback
        feedback = Feedback(
            content="This was very helpful",
            feedback_type=FeedbackType.CONFIRMATION,
            rating=5.0
        )
        
        feedback_id = self.feedback_processor.process_feedback(feedback)
        
        # Retrieve the feedback
        retrieved_feedback = self.feedback_processor.get_feedback(feedback_id)
        
        # Verify the retrieved feedback
        self.assertIsNotNone(retrieved_feedback)
        self.assertEqual(retrieved_feedback.content, "This was very helpful")
        self.assertEqual(retrieved_feedback.feedback_type, FeedbackType.CONFIRMATION)
        self.assertEqual(retrieved_feedback.rating, 5.0)

    def test_update_associated_experience(self):
        """Test updating an associated experience when processing feedback."""
        # Create and record an experience
        experience = Experience(
            context="Test context",
            query="Test query",
            response="Test response",
            experience_type=ExperienceType.CODE_EXPLANATION
        )
        
        experience_id = self.experience_tracker.record_experience(experience)
        
        # Create and process feedback with the experience as target
        feedback = Feedback(
            content="This was very helpful",
            feedback_type=FeedbackType.CONFIRMATION,
            rating=5.0,
            target_response_id=experience_id
        )
        
        # Override the experience tracker
        self.feedback_processor.experience_tracker = self.experience_tracker
        
        # Process the feedback
        feedback_id = self.feedback_processor.process_feedback(feedback)
        
        # Retrieve the updated experience
        updated_experience = self.experience_tracker.get_experience(experience_id)
        
        # Verify the feedback was added to the experience
        self.assertIn(feedback_id, updated_experience.feedback)
        self.assertEqual(updated_experience.feedback[feedback_id]["content"], "This was very helpful")


class TestKnowledgeExtractor(unittest.TestCase):
    """Tests for the KnowledgeExtractor class."""

    def setUp(self):
        """Set up for testing."""
        self.knowledge_extractor = KnowledgeExtractor(extraction_threshold=0.5)

    def test_extract_code_blocks_from_text(self):
        """Test extracting code blocks from text."""
        # Text with code blocks
        text = """
        Here's an example of a Python function:
        
        ```python
        def add(a, b):
            return a + b
        ```
        
        And another example:
        
        ```javascript
        function multiply(a, b) {
            return a * b;
        }
        ```
        """
        
        code_blocks = self.knowledge_extractor._extract_code_blocks_from_text(text)
        
        # Verify the extracted code blocks
        self.assertEqual(len(code_blocks), 2)
        self.assertIn("def add(a, b):", code_blocks[0])
        self.assertIn("function multiply(a, b) {", code_blocks[1])

    def test_extract_knowledge_from_text(self):
        """Test extracting knowledge from text."""
        # Sample text
        text = """
        # Python Dictionary Example
        
        Python dictionaries are key-value stores. Here's an example:
        
        ```python
        user = {
            "name": "Alice",
            "age": 30,
            "is_admin": True
        }
        
        # Accessing values
        print(user["name"])  # Output: Alice
        ```
        
        Always remember to check if a key exists before accessing it to avoid KeyError exceptions.
        """
        
        knowledge_items = self.knowledge_extractor.extract_knowledge_from_text(
            text=text,
            source="test_source"
        )
        
        # Verify the extracted knowledge
        self.assertGreater(len(knowledge_items), 0)
        
        # At least one code snippet should be extracted
        code_snippets = [item for item in knowledge_items if item.knowledge_type == KnowledgeType.CODE_SNIPPET]
        self.assertGreater(len(code_snippets), 0)
        
        # The code snippet should contain the dictionary example
        self.assertTrue(any("user = {" in item.content for item in code_snippets))


class TestReflector(unittest.TestCase):
    """Tests for the Reflector class."""

    def setUp(self):
        """Set up temporary directories for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.reflector = Reflector(storage_dir=self.temp_dir, reflection_period=3)
        
        # Create experience tracker for test data
        self.experience_dir = tempfile.mkdtemp()
        self.experience_tracker = ExperienceTracker(storage_dir=self.experience_dir)
        
        # Override the experience tracker in the reflector
        self.reflector.experience_tracker = self.experience_tracker
        
        # Create test experiences
        self.create_test_experiences()

    def tearDown(self):
        """Clean up the temporary directories."""
        shutil.rmtree(self.temp_dir)
        shutil.rmtree(self.experience_dir)

    def create_test_experiences(self):
        """Create test experiences for reflection."""
        # Create experiences with different outcomes and types
        experience_types = [
            ExperienceType.CODE_EXPLANATION,
            ExperienceType.CODE_GENERATION,
            ExperienceType.ERROR_RESOLUTION
        ]
        
        outcomes = ["success", "failure", "unknown"]
        
        for i, (exp_type, outcome) in enumerate(zip(experience_types, outcomes)):
            experience = Experience(
                context=f"Context {i}",
                query=f"Query {i}",
                response=f"Response {i}",
                experience_type=exp_type
            )
            
            experience_id = self.experience_tracker.record_experience(experience)
            
            # Update the outcome
            self.experience_tracker.update_experience(
                experience_id=experience_id,
                outcome=outcome
            )
            
            # Add some feedback
            if outcome == "success":
                feedback = {"feedback_id_1": {"content": "Great explanation", "type": "confirmation", "rating": 5.0}}
            elif outcome == "failure":
                feedback = {"feedback_id_2": {"content": "This didn't work", "type": "rejection", "rating": 2.0}}
            else:
                feedback = {}
                
            if feedback:
                self.experience_tracker.update_experience(
                    experience_id=experience_id,
                    feedback=feedback
                )

    def test_reflect_on_experiences(self):
        """Test reflecting on a set of experiences."""
        # Get all experiences
        experiences = self.experience_tracker.list_experiences()
        
        # Perform reflection
        reflection_result = self.reflector.reflect_on_experiences(experiences)
        
        # Verify the reflection result
        self.assertIsNotNone(reflection_result)
        self.assertGreater(len(reflection_result.insights), 0)
        self.assertGreater(len(reflection_result.improvement_areas), 0)
        self.assertGreater(len(reflection_result.action_plan), 0)
        
        # Verify the reflection was saved
        saved_reflections = list(self.reflector.reflection_results.values())
        self.assertEqual(len(saved_reflections), 1)
        self.assertEqual(saved_reflections[0].reflection_id, reflection_result.reflection_id)

    def test_notify_new_experience(self):
        """Test that notification of new experiences triggers reflection at the threshold."""
        # Should be no reflections initially
        self.assertEqual(len(self.reflector.reflection_results), 0)
        
        # Create a new experience and notify (should not trigger reflection yet)
        experience = Experience(
            context="New context",
            query="New query",
            response="New response",
            experience_type=ExperienceType.QUESTION_ANSWERING
        )
        
        # The reflector has a period of 3, and we already have 3 experiences,
        # so adding 1 more should trigger reflection
        triggered = self.reflector.notify_new_experience(experience)
        
        # Verify reflection was triggered
        self.assertTrue(triggered)
        self.assertEqual(len(self.reflector.reflection_results), 1)
        
        # Verify the experience counter was reset
        self.assertEqual(self.reflector.experience_count_since_reflection, 0)


class TestLearningManager(unittest.TestCase):
    """Tests for the LearningManager class."""

    def setUp(self):
        """Set up temporary directories for testing."""
        # Create a temporary config with directories
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "experience_storage_dir": Path(self.temp_dir) / "experiences",
            "feedback_storage_dir": Path(self.temp_dir) / "feedback",
            "reflection_storage_dir": Path(self.temp_dir) / "reflections",
            "reflection_period": 5,
            "knowledge_extraction_threshold": 0.5
        }
        
        # Create directories
        for dir_path in [
            self.config["experience_storage_dir"],
            self.config["feedback_storage_dir"],
            self.config["reflection_storage_dir"]
        ]:
            os.makedirs(dir_path, exist_ok=True)
            
        self.learning_manager = LearningManager(config=self.config)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_record_experience(self):
        """Test recording an experience through the manager."""
        experience_id = self.learning_manager.record_experience(
            context="Test context",
            query="Test query",
            response="Test response",
            experience_type=ExperienceType.CODE_EXPLANATION,
            metadata={"test": True},
            extract_knowledge=True
        )
        
        # Verify the experience was recorded
        experience = self.learning_manager.experience_tracker.get_experience(experience_id)
        self.assertIsNotNone(experience)
        self.assertEqual(experience.context, "Test context")
        self.assertEqual(experience.query, "Test query")
        self.assertEqual(experience.metadata["test"], True)

    def test_process_feedback(self):
        """Test processing feedback through the manager."""
        # First record an experience
        experience_id = self.learning_manager.record_experience(
            context="Test context",
            query="Test query",
            response="Test response",
            experience_type=ExperienceType.CODE_EXPLANATION,
            extract_knowledge=False
        )
        
        # Process feedback for the experience
        feedback_id = self.learning_manager.process_feedback(
            content="This was helpful",
            feedback_type=FeedbackType.CONFIRMATION,
            target_response_id=experience_id,
            rating=5.0,
            update_experience=True
        )
        
        # Verify the feedback was processed
        feedback = self.learning_manager.feedback_processor.get_feedback(feedback_id)
        self.assertIsNotNone(feedback)
        self.assertEqual(feedback.content, "This was helpful")
        self.assertEqual(feedback.target_response_id, experience_id)
        
        # Verify the experience was updated
        experience = self.learning_manager.experience_tracker.get_experience(experience_id)
        self.assertEqual(experience.outcome, "success")  # Set based on CONFIRMATION type

    def test_learn_from_conversation(self):
        """Test learning from a conversation."""
        # Sample conversation
        conversation = [
            {"role": "user", "content": "How do I use list comprehensions in Python?"},
            {"role": "assistant", "content": """
                List comprehensions are a concise way to create lists in Python. Here's an example:
                
                ```python
                # Create a list of squares
                squares = [x**2 for x in range(10)]
                print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
                ```
                
                You can also add conditions:
                
                ```python
                # Only even numbers
                even_squares = [x**2 for x in range(10) if x % 2 == 0]
                print(even_squares)  # [0, 4, 16, 36, 64]
                ```
            """},
            {"role": "user", "content": "Thanks, that's very helpful!"},
            {"role": "assistant", "content": "You're welcome! Let me know if you have any other questions."}
        ]
        
        # Process the conversation
        results = self.learning_manager.learn_from_conversation(
            messages=conversation,
            conversation_id="test_conversation"
        )
        
        # Verify experiences and feedback were extracted
        self.assertEqual(len(results["experiences"]), 2)  # 2 assistant responses
        self.assertEqual(len(results["feedback"]), 1)  # 1 feedback ("Thanks...")
        self.assertGreater(len(results["knowledge_items"]), 0)  # At least some knowledge extracted


def run_tests():
    """Run all tests."""
    unittest.main()


if __name__ == "__main__":
    run_tests() 