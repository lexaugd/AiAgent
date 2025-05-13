#!/usr/bin/env python3
"""
Learning system demo for the Autonomous Coding Agent.

This script demonstrates the capabilities of the learning system, including:
1. Experience tracking
2. Feedback processing
3. Knowledge extraction
4. Self-reflection
"""

import sys
import os
from pathlib import Path
import time
import json
import random
from typing import Dict, List, Any

# Add paths to make imports work
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))

from learning.manager import get_learning_manager
from learning.types import ExperienceType, FeedbackType, KnowledgeType

# Constants
DEMO_STORAGE_DIR = Path.home() / ".autonomous_agent" / "learning_demo"
DEMO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def convert_paths_to_strings(obj):
    """Convert Path objects to strings for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths_to_strings(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def print_section(title):
    """Print a formatted section title."""
    print("\n" + "-" * 80)
    print(f" {title} ".center(80, "-"))
    print("-" * 80 + "\n")


def setup_demo_environment():
    """Set up the demo environment with directories and configurations."""
    print_section("Setting up demo environment")
    
    config = {
        "experience_storage_dir": DEMO_STORAGE_DIR / "experiences",
        "feedback_storage_dir": DEMO_STORAGE_DIR / "feedback",
        "reflection_storage_dir": DEMO_STORAGE_DIR / "reflections",
        "reflection_period": 5,  # Trigger reflection after 5 experiences
        "knowledge_extraction_threshold": 0.5  # Lower threshold for demo purposes
    }
    
    print(f"Demo storage directory: {DEMO_STORAGE_DIR}")
    
    # Convert Path objects to strings for JSON serialization
    json_safe_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
    }
    print(f"Configuration: {json.dumps(json_safe_config, indent=2)}")
    
    return config


def demo_experience_tracking(learning_manager):
    """Demonstrate experience tracking capabilities."""
    print_section("Experience Tracking Demo")
    
    # Sample experiences
    sample_experiences = [
        {
            "context": "User is working on a Python web application",
            "query": "How do I handle form submissions in Flask?",
            "response": """
            In Flask, you can handle form submissions using the request object:

            ```python
            from flask import Flask, request, render_template

            app = Flask(__name__)

            @app.route('/form', methods=['GET', 'POST'])
            def handle_form():
                if request.method == 'POST':
                    name = request.form.get('name')
                    email = request.form.get('email')
                    # Process the form data
                    return f'Form submitted: {name}, {email}'
                return render_template('form.html')
            ```

            Make sure to set the methods parameter to include 'POST' for routes that accept form submissions.
            """,
            "experience_type": ExperienceType.CODE_EXPLANATION,
            "metadata": {"tags": ["flask", "web", "forms"]}
        },
        {
            "context": "User is debugging a Python script",
            "query": "I'm getting this error: TypeError: 'int' object is not iterable. What's wrong?",
            "response": """
            The error "TypeError: 'int' object is not iterable" occurs when you try to iterate over an integer, which isn't possible.

            This typically happens in situations like:

            ```python
            # Wrong:
            number = 5
            for i in number:  # This will raise the error
                print(i)
            ```

            To fix this, make sure you're iterating over an iterable object like a list, tuple, or range:

            ```python
            # Correct:
            number = 5
            for i in range(number):  # This works
                print(i)
            ```

            Check your code for places where you might be trying to iterate over an integer variable.
            """,
            "experience_type": ExperienceType.ERROR_RESOLUTION,
            "metadata": {"error_type": "TypeError", "language": "python"}
        },
        {
            "context": "User is building a new application",
            "query": "Write a function to calculate Fibonacci numbers recursively",
            "response": """
            Here's a recursive function to calculate Fibonacci numbers:

            ```python
            def fibonacci(n):
                if n <= 0:
                    return 0
                elif n == 1:
                    return 1
                else:
                    return fibonacci(n-1) + fibonacci(n-2)
                    
            # Example usage:
            for i in range(10):
                print(f"fibonacci({i}) = {fibonacci(i)}")
            ```

            Note that this recursive implementation is not efficient for large values of n due to repeated calculations. Consider using memoization or an iterative approach for better performance.
            """,
            "experience_type": ExperienceType.CODE_GENERATION,
            "metadata": {"algorithm": "fibonacci", "approach": "recursive"}
        }
    ]
    
    # Record experiences
    experience_ids = []
    for idx, exp in enumerate(sample_experiences):
        print(f"Recording experience {idx+1}/{len(sample_experiences)}...")
        exp_id = learning_manager.record_experience(
            context=exp["context"],
            query=exp["query"],
            response=exp["response"],
            experience_type=exp["experience_type"],
            metadata=exp["metadata"],
            extract_knowledge=True
        )
        experience_ids.append(exp_id)
        print(f"  Recorded experience ID: {exp_id}")
        
    print("\nExperience statistics:")
    stats = learning_manager.experience_tracker.get_statistics()
    # Convert Path objects to strings for JSON serialization
    json_safe_stats = convert_paths_to_strings(stats)
    print(json.dumps(json_safe_stats, indent=2))
    
    return experience_ids


def demo_feedback_processing(learning_manager, experience_ids):
    """Demonstrate feedback processing capabilities."""
    print_section("Feedback Processing Demo")
    
    if not experience_ids:
        print("No experiences available to provide feedback on.")
        return []
        
    # Sample feedback
    sample_feedback = [
        {
            "content": "Thanks, this worked perfectly for my Flask application!",
            "feedback_type": FeedbackType.CONFIRMATION,
            "target_response_id": experience_ids[0],
            "rating": 5.0
        },
        {
            "content": "The solution didn't completely fix my issue. I still get the error sometimes.",
            "feedback_type": FeedbackType.CORRECTION,
            "target_response_id": experience_ids[1],
            "rating": 3.0
        },
        {
            "content": "This works, but can you also explain how to make it more efficient with memoization?",
            "feedback_type": FeedbackType.CLARIFICATION,
            "target_response_id": experience_ids[2],
            "rating": 4.0
        }
    ]
    
    # Process feedback
    feedback_ids = []
    for idx, fb in enumerate(sample_feedback):
        print(f"Processing feedback {idx+1}/{len(sample_feedback)}...")
        fb_id = learning_manager.process_feedback(
            content=fb["content"],
            feedback_type=fb["feedback_type"],
            target_response_id=fb["target_response_id"],
            rating=fb["rating"],
            metadata={"source": "demo"}
        )
        feedback_ids.append(fb_id)
        print(f"  Processed feedback ID: {fb_id}")
        
    print("\nFeedback statistics:")
    stats = learning_manager.feedback_processor.get_statistics()
    json_safe_stats = convert_paths_to_strings(stats)
    print(json.dumps(json_safe_stats, indent=2))
    
    print("\nFeedback trends:")
    trends = learning_manager.feedback_processor.analyze_feedback_trends()
    json_safe_trends = convert_paths_to_strings(trends)
    print(json.dumps(json_safe_trends, indent=2))
    
    return feedback_ids


def demo_knowledge_extraction(learning_manager):
    """Demonstrate knowledge extraction capabilities."""
    print_section("Knowledge Extraction Demo")
    
    # Sample text for knowledge extraction
    sample_text = """
    # Python Best Practices

    Here are some important best practices for Python development:

    1. Follow PEP 8 style guidelines for consistent code formatting.
    
    2. Use virtual environments to manage dependencies for each project.
    
    3. Write unit tests for your code using frameworks like pytest.
    
    4. Use list comprehensions for cleaner, more readable code:
    ```python
    # Instead of:
    squares = []
    for i in range(10):
        squares.append(i**2)
        
    # Use:
    squares = [i**2 for i in range(10)]
    ```
    
    5. Handle exceptions properly:
    ```python
    try:
        # Risky operation
        file = open('data.txt', 'r')
        content = file.read()
    except FileNotFoundError:
        print("File not found.")
    finally:
        file.close()
    ```
    
    6. Use context managers for resource management:
    ```python
    with open('data.txt', 'r') as file:
        content = file.read()
    # File is automatically closed after the block
    ```
    
    Always remember that readability counts! Clear and maintainable code is more important than clever code.
    """
    
    print("Extracting knowledge from sample text...")
    knowledge_items = learning_manager.extract_knowledge(
        text=sample_text,
        source="demo_best_practices"
    )
    
    print(f"\nExtracted {len(knowledge_items)} knowledge items:")
    for idx, item in enumerate(knowledge_items):
        print(f"\n{idx+1}. {item.knowledge_type.value.upper()}: {item.knowledge_id}")
        print(f"   Confidence: {item.confidence:.2f}")
        print(f"   Content snippet: {item.content[:100]}...")
        
    return [item.knowledge_id for item in knowledge_items]


def demo_reflection(learning_manager):
    """Demonstrate reflection capabilities."""
    print_section("Reflection Demo")
    
    # Add a few more experiences to trigger reflection
    print("Adding additional experiences to trigger reflection...")
    
    experience_templates = [
        {
            "context_template": "User is working on {project_type}",
            "query_templates": [
                "How do I implement {feature} in {language}?",
                "What's the best way to handle {problem} in {language}?",
                "Can you help me debug this {language} code that's causing {error}?"
            ],
            "response_template": "Here's how you can implement {feature} in {language}:\n\n```{language}\n{code}\n```\n\nThis approach {advantage}.",
            "experience_types": [ExperienceType.CODE_GENERATION, ExperienceType.CODE_EXPLANATION, ExperienceType.ERROR_RESOLUTION],
            "possible_values": {
                "project_type": ["a web application", "a data analysis project", "a mobile app", "a game"],
                "feature": ["authentication", "data validation", "error handling", "caching", "API integration"],
                "language": ["Python", "JavaScript", "TypeScript", "Java", "C++"],
                "problem": ["memory leaks", "performance issues", "concurrency", "security vulnerabilities"],
                "error": ["syntax error", "runtime exception", "type mismatch", "null reference"],
                "code": ["def example():\n    print('Example code')\n    return True", "function example() {\n    console.log('Example code');\n    return true;\n}", "class Example {\n    public static void main() {\n        System.out.println(\"Example code\");\n    }\n}"],
                "advantage": ["is efficient", "is easy to maintain", "follows best practices", "is well-documented"]
            }
        }
    ]
    
    # Generate and record random experiences
    for i in range(3):  # Add 3 more experiences
        template = random.choice(experience_templates)
        
        # Randomly select values
        project_type = random.choice(template["possible_values"]["project_type"])
        feature = random.choice(template["possible_values"]["feature"])
        language = random.choice(template["possible_values"]["language"])
        problem = random.choice(template["possible_values"]["problem"])
        error = random.choice(template["possible_values"]["error"])
        code = random.choice(template["possible_values"]["code"])
        advantage = random.choice(template["possible_values"]["advantage"])
        
        # Create the experience
        context = template["context_template"].format(project_type=project_type)
        query = random.choice(template["query_templates"]).format(
            feature=feature, language=language, problem=problem, error=error
        )
        response = template["response_template"].format(
            feature=feature, language=language, code=code, advantage=advantage
        )
        experience_type = random.choice(template["experience_types"])
        
        # Record the experience
        exp_id = learning_manager.record_experience(
            context=context,
            query=query,
            response=response,
            experience_type=experience_type,
            metadata={"generated": True, "language": language},
            extract_knowledge=True
        )
        print(f"  Recorded additional experience ID: {exp_id}")
        
        # Add feedback for the experience
        feedback_types = [FeedbackType.CONFIRMATION, FeedbackType.CLARIFICATION, FeedbackType.CORRECTION]
        ratings = [3.0, 4.0, 5.0]
        
        fb_type = random.choice(feedback_types)
        rating = random.choice(ratings)
        
        if fb_type == FeedbackType.CONFIRMATION:
            content = "Thanks, this is exactly what I needed!"
        elif fb_type == FeedbackType.CLARIFICATION:
            content = f"This looks good, but can you explain more about how {feature} works?"
        else:
            content = f"This doesn't quite work for my {project_type} project. Can you revise it?"
            
        fb_id = learning_manager.process_feedback(
            content=content,
            feedback_type=fb_type,
            target_response_id=exp_id,
            rating=rating,
            metadata={"generated": True}
        )
        print(f"  Processed feedback ID: {fb_id}")
        
        # Small delay to make timestamps different
        time.sleep(0.5)
    
    # Trigger reflection manually
    print("\nTriggering reflection on all experiences...")
    reflection_result = learning_manager.trigger_reflection()
    
    print("\nReflection insights:")
    for idx, insight in enumerate(reflection_result.insights):
        print(f"{idx+1}. {insight}")
        
    print("\nIdentified improvement areas:")
    for idx, area in enumerate(reflection_result.improvement_areas):
        print(f"{idx+1}. {area}")
        
    print("\nAction plan:")
    for idx, action in enumerate(reflection_result.action_plan):
        print(f"{idx+1}. {action}")
        
    return reflection_result.reflection_id


def demo_learning_from_conversation(learning_manager):
    """Demonstrate learning from a complete conversation."""
    print_section("Learning from Conversation Demo")
    
    # Sample conversation (in chat message format)
    conversation = [
        {"role": "user", "content": "I need to implement a binary search tree in Python. Can you help?"},
        {"role": "assistant", "content": """
            Sure, here's a basic implementation of a Binary Search Tree in Python:

            ```python
            class Node:
                def __init__(self, key):
                    self.key = key
                    self.left = None
                    self.right = None

            class BinarySearchTree:
                def __init__(self):
                    self.root = None
                
                def insert(self, key):
                    self.root = self._insert_recursive(self.root, key)
                
                def _insert_recursive(self, root, key):
                    # If tree is empty, create a new node
                    if root is None:
                        return Node(key)
                    
                    # Traverse the tree
                    if key < root.key:
                        root.left = self._insert_recursive(root.left, key)
                    else:
                        root.right = self._insert_recursive(root.right, key)
                    
                    return root
                
                def search(self, key):
                    return self._search_recursive(self.root, key)
                
                def _search_recursive(self, root, key):
                    # Base case: root is None or key is at root
                    if root is None or root.key == key:
                        return root
                    
                    # Key is greater than root's key
                    if key < root.key:
                        return self._search_recursive(root.left, key)
                    
                    # Key is less than root's key
                    return self._search_recursive(root.right, key)
                
                def inorder_traversal(self):
                    result = []
                    self._inorder_recursive(self.root, result)
                    return result
                
                def _inorder_recursive(self, root, result):
                    if root:
                        self._inorder_recursive(root.left, result)
                        result.append(root.key)
                        self._inorder_recursive(root.right, result)
            ```

            This implementation includes the basic operations: insertion, search, and inorder traversal. You can use it like this:

            ```python
            # Create a BST
            bst = BinarySearchTree()

            # Insert values
            bst.insert(50)
            bst.insert(30)
            bst.insert(70)
            bst.insert(20)
            bst.insert(40)

            # Search
            found_node = bst.search(40)
            print("Found:", found_node.key if found_node else "Not found")

            # Traverse
            print("Inorder traversal:", bst.inorder_traversal())
            ```

            This will output:
            ```
            Found: 40
            Inorder traversal: [20, 30, 40, 50, 70]
            ```

            Is there any specific part of the BST you'd like me to explain in more detail?
        """},
        {"role": "user", "content": "Thanks! This is very helpful. Could you also add a delete method to the BST?"},
        {"role": "assistant", "content": """
            Yes, I'd be happy to add a delete method to the Binary Search Tree. Deletion in a BST is a bit more complex because there are three cases to handle:

            1. Node with no children: Simply remove the node.
            2. Node with one child: Replace the node with its child.
            3. Node with two children: Find the inorder successor (smallest value in the right subtree), replace the node's value with the successor's value, then delete the successor.

            Here's the updated code with the delete method:

            ```python
            class Node:
                def __init__(self, key):
                    self.key = key
                    self.left = None
                    self.right = None

            class BinarySearchTree:
                def __init__(self):
                    self.root = None
                
                # ... (insert, search, and traversal methods remain the same)
                
                def delete(self, key):
                    self.root = self._delete_recursive(self.root, key)
                
                def _delete_recursive(self, root, key):
                    # Base case: If tree is empty
                    if root is None:
                        return root
                    
                    # Traverse the tree
                    if key < root.key:
                        root.left = self._delete_recursive(root.left, key)
                    elif key > root.key:
                        root.right = self._delete_recursive(root.right, key)
                    else:
                        # Node with only one child or no child
                        if root.left is None:
                            return root.right
                        elif root.right is None:
                            return root.left
                        
                        # Node with two children
                        # Get the inorder successor (smallest in the right subtree)
                        root.key = self._min_value(root.right)
                        
                        # Delete the inorder successor
                        root.right = self._delete_recursive(root.right, root.key)
                    
                    return root
                
                def _min_value(self, node):
                    current = node
                    # Find the leftmost leaf
                    while current.left is not None:
                        current = current.left
                    return current.key
            ```

            Usage example with deletion:

            ```python
            # Create a BST
            bst = BinarySearchTree()

            # Insert values
            bst.insert(50)
            bst.insert(30)
            bst.insert(70)
            bst.insert(20)
            bst.insert(40)
            bst.insert(60)
            bst.insert(80)

            print("Inorder traversal before deletion:", bst.inorder_traversal())

            # Delete a leaf node (20)
            bst.delete(20)
            print("Inorder traversal after deleting 20:", bst.inorder_traversal())

            # Delete a node with one child (30)
            bst.delete(30)
            print("Inorder traversal after deleting 30:", bst.inorder_traversal())

            # Delete a node with two children (70)
            bst.delete(70)
            print("Inorder traversal after deleting 70:", bst.inorder_traversal())
            ```

            This will output:
            ```
            Inorder traversal before deletion: [20, 30, 40, 50, 60, 70, 80]
            Inorder traversal after deleting 20: [30, 40, 50, 60, 70, 80]
            Inorder traversal after deleting 30: [40, 50, 60, 70, 80]
            Inorder traversal after deleting 70: [40, 50, 60, 80]
            ```

            The delete operation ensures that the BST property is maintained after removing a node.
        """},
        {"role": "user", "content": "Perfect! This is exactly what I needed. The implementation is very clear."},
        {"role": "assistant", "content": "I'm glad I could help! If you have any more questions about binary search trees or need assistance with other data structures or algorithms, feel free to ask."}
    ]
    
    print("Processing conversation to extract experiences, feedback, and knowledge...")
    results = learning_manager.learn_from_conversation(
        messages=conversation,
        conversation_id="demo_bst_conversation",
        metadata={"topic": "data structures", "subtopic": "binary search tree"}
    )
    
    print("\nLearning results from conversation:")
    print(f"Extracted experiences: {len(results['experiences'])}")
    print(f"Detected feedback: {len(results['feedback'])}")
    print(f"Extracted knowledge items: {len(results['knowledge_items'])}")
    
    return results


def demo_integration_with_memory(learning_manager):
    """Demonstrate integration with the memory system."""
    print_section("Integration with Memory Demo")
    
    print("Retrieving knowledge from memory based on queries...")
    
    # Sample queries to test retrieval
    queries = [
        "How to implement a binary search tree",
        "Python best practices",
        "Handle form submissions in Flask",
        "Fix TypeError in Python",
        "Fibonacci implementation"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Retrieve from memory using existing methods
        results = learning_manager.memory_manager.retrieve_relevant(
            query=query,
            n_results=3
        )
        
        # Also check for code examples
        code_results = learning_manager.memory_manager.retrieve_code_examples(
            query=query,
            n_results=2
        )
        
        # Combine results
        all_results = results + code_results
        
        if all_results:
            print(f"Found {len(all_results)} relevant items:")
            for idx, item in enumerate(all_results):
                print(f"  {idx+1}. Type: {item.item_type}")
                print(f"     Content snippet: {item.content[:100]}..." if len(item.content) > 100 else f"     Content: {item.content}")
                if hasattr(item, 'metadata') and item.metadata:
                    # Print some key metadata if available
                    print(f"     Source: {item.metadata.get('source', 'unknown')}")
                    print(f"     Confidence: {item.metadata.get('confidence', 'N/A')}")
        else:
            print("No relevant items found in memory.")
    
    print("\nRetrieving improvement suggestions...")
    suggestions = learning_manager.get_improvement_suggestions()
    
    print(f"Found {len(suggestions)} improvement suggestions:")
    for idx, suggestion in enumerate(suggestions):
        print(f"{idx+1}. {suggestion}")
        
    return suggestions


def main():
    """Run the learning system demo."""
    print_header("AUTONOMOUS CODING AGENT - LEARNING SYSTEM DEMO")
    
    # Setup
    config = setup_demo_environment()
    learning_manager = get_learning_manager(config)
    
    # Run demos
    experience_ids = demo_experience_tracking(learning_manager)
    feedback_ids = demo_feedback_processing(learning_manager, experience_ids)
    knowledge_ids = demo_knowledge_extraction(learning_manager)
    reflection_id = demo_reflection(learning_manager)
    conversation_results = demo_learning_from_conversation(learning_manager)
    memory_results = demo_integration_with_memory(learning_manager)
    
    # Show overall learning statistics
    print_section("Overall Learning Statistics")
    
    stats = learning_manager.get_learning_statistics()
    
    # Convert Path objects to strings for JSON serialization
    json_safe_stats = convert_paths_to_strings(stats)
    print(json.dumps(json_safe_stats, indent=2))
    
    print_header("DEMO COMPLETED")
    print(f"All demo data stored in: {DEMO_STORAGE_DIR}")
    print("\nComponents demonstrated:")
    print("- Experience tracking")
    print("- Feedback processing")
    print("- Knowledge extraction")
    print("- Self-reflection")
    print("- Learning from conversations")
    print("- Integration with memory system")


if __name__ == "__main__":
    main() 