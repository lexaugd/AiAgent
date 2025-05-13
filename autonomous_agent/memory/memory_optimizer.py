"""
Memory optimization module for improving context window utilization.

This module implements strategies to optimize memory usage and context composition
to improve the quality and relevance of context provided to the LLM, addressing
the hallucination issues identified in our investigation.
"""

import re
import time
from typing import Dict, List, Any, Tuple, Optional, Union
from enum import Enum
from loguru import logger

from .short_term import Message
from .long_term import MemoryItem
from .embeddings import EmbeddingGenerator


class QueryComplexity(Enum):
    """Enum representing different query complexity levels."""
    SIMPLE = 1       # Simple, factual queries
    MODERATE = 2     # Queries requiring some context and reasoning
    COMPLEX = 3      # Complex questions requiring significant context
    AMBIGUOUS = 4    # Ambiguous queries needing clarification


class QueryType(Enum):
    """Enum representing different query types identified in our analysis."""
    SPECIFIC_TECHNICAL = 1  # Specific technical queries about concrete topics
    NOVEL_CONCEPT = 2       # Queries about novel or specialized concepts
    CONTEXT_DEPENDENT = 3   # Queries that depend heavily on conversation context
    AMBIGUOUS = 4           # Vague or ambiguous queries


class MemoryOptimizer:
    """
    Memory optimizer that implements strategies to improve context window utilization
    and relevance of retrieved information.
    """
    
    def __init__(self, 
                 model_context_size: int = 4096,
                 embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize the memory optimizer.
        
        Args:
            model_context_size (int): Maximum context window size for the model in tokens
            embedding_generator (EmbeddingGenerator, optional): Embedding generator for relevance scoring
        """
        self.model_context_size = model_context_size
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        
        # Default token allocation percentages by content type
        self.default_allocations = {
            "system_prompt": 0.10,       # 10% for system prompt
            "recent_messages": 0.40,     # 40% for recent conversation
            "relevant_knowledge": 0.45,  # 45% for relevant knowledge
            "working_memory": 0.05       # 5% for working memory
        }
        
        # Token allocation adjustments based on query type
        self.query_type_allocations = {
            QueryType.SPECIFIC_TECHNICAL: {
                "system_prompt": 0.10,
                "recent_messages": 0.30,
                "relevant_knowledge": 0.55,
                "working_memory": 0.05
            },
            QueryType.NOVEL_CONCEPT: {
                "system_prompt": 0.10,
                "recent_messages": 0.25,
                "relevant_knowledge": 0.60,
                "working_memory": 0.05
            },
            QueryType.CONTEXT_DEPENDENT: {
                "system_prompt": 0.10,
                "recent_messages": 0.60,
                "relevant_knowledge": 0.25,
                "working_memory": 0.05
            },
            QueryType.AMBIGUOUS: {
                "system_prompt": 0.15,  # More guidance needed
                "recent_messages": 0.50,
                "relevant_knowledge": 0.30,
                "working_memory": 0.05
            }
        }
        
        # Complexity adjustments (multipliers applied to knowledge allocation)
        self.complexity_adjustments = {
            QueryComplexity.SIMPLE: 0.8,     # Reduce knowledge for simple queries
            QueryComplexity.MODERATE: 1.0,   # Standard allocation
            QueryComplexity.COMPLEX: 1.2,    # More knowledge for complex queries
            QueryComplexity.AMBIGUOUS: 0.7   # Less knowledge, more system guidance
        }
        
        logger.info(f"Initialized MemoryOptimizer with context size: {model_context_size} tokens")

    def classify_query(self, query: str, conversation_history: List[Message]) -> Tuple[QueryType, QueryComplexity]:
        """
        Classify the query type and complexity to determine optimal allocation strategy.
        
        Args:
            query (str): The current user query
            conversation_history (List[Message]): Recent conversation history
            
        Returns:
            Tuple[QueryType, QueryComplexity]: Classified query type and complexity
        """
        # Classify query type
        query_lower = query.lower()
        
        # Check for context-dependent queries
        context_phrases = ["that", "this", "those", "it", "they", "the code", "the problem", 
                           "the issue", "the function", "the result", "explain", "why"]
        
        # If short query with context-dependent phrases, likely context-dependent
        if len(query.split()) < 8 and any(phrase in query_lower for phrase in context_phrases):
            query_type = QueryType.CONTEXT_DEPENDENT
        # Check for specific technical queries
        elif re.search(r'(how|what|why).+(implement|create|build|use|fix|debug|optimize|improve)', query_lower):
            query_type = QueryType.SPECIFIC_TECHNICAL
        # Check for vague/ambiguous queries
        elif len(query.split()) < 4 or "better" in query_lower or "improve" in query_lower:
            query_type = QueryType.AMBIGUOUS
        # Check for novel concept queries
        elif "quantum" in query_lower or "neural" in query_lower or "ai" in query_lower or "blockchain" in query_lower:
            query_type = QueryType.NOVEL_CONCEPT
        else:
            # Default to specific technical if we can't clearly classify
            query_type = QueryType.SPECIFIC_TECHNICAL
        
        # Classify complexity
        if len(query.split()) < 5:
            complexity = QueryComplexity.SIMPLE
        elif len(query.split()) > 15 or ";" in query or "and" in query_lower and "or" in query_lower:
            complexity = QueryComplexity.COMPLEX
        elif "how" in query_lower and any(word in query_lower for word in ["implement", "create", "design", "architect"]):
            complexity = QueryComplexity.COMPLEX
        else:
            complexity = QueryComplexity.MODERATE
            
        logger.debug(f"Query classified as {query_type.name}, {complexity.name}")
        return query_type, complexity

    def get_token_allocations(self, query_type: QueryType, complexity: QueryComplexity) -> Dict[str, float]:
        """
        Get token allocations based on query type and complexity.
        
        Args:
            query_type (QueryType): The type of query
            complexity (QueryComplexity): The complexity of the query
            
        Returns:
            Dict[str, float]: Token allocation percentages for each content type
        """
        # Start with the base allocation for this query type
        allocations = self.query_type_allocations[query_type].copy()
        
        # Apply complexity adjustments to relevant_knowledge
        knowledge_allocation = allocations["relevant_knowledge"]
        adjusted_knowledge = knowledge_allocation * self.complexity_adjustments[complexity]
        
        # Calculate the difference and redistribute to recent_messages
        difference = knowledge_allocation - adjusted_knowledge
        allocations["relevant_knowledge"] = adjusted_knowledge
        allocations["recent_messages"] += difference
        
        logger.debug(f"Token allocations for {query_type.name}, {complexity.name}: {allocations}")
        return allocations

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a string.
        This is a more sophisticated approximation than simple word count.
        
        Args:
            text (str): Text to estimate tokens for
            
        Returns:
            int: Estimated token count
        """
        # Improved token estimation based on GPT tokenization patterns
        # This is still an approximation, but better than word count
        
        # Count regular words
        words = len(text.split())
        
        # Count special tokens (punctuation, etc.)
        punctuation_count = len(re.findall(r'[,.;:!?()[\]{}"`\'#$%^&*=+~_|<>/\\-]', text))
        
        # Count numbers as special tokens
        number_count = len(re.findall(r'\d+', text))
        
        # Count code-related tokens
        code_tokens = len(re.findall(r'[=\+\-\*/:<>{}[\]()]', text))
        
        # Estimate final count: most words are 1-2 tokens, punctuation is usually a single token
        return int(words * 1.3 + punctuation_count * 0.5 + number_count * 0.5 + code_tokens * 0.5)

    def optimize_conversation_history(self, messages: List[Message], 
                                     allocated_tokens: int,
                                     query_type: QueryType) -> List[Message]:
        """
        Optimize the conversation history to fit within the allocated token budget.
        
        Args:
            messages (List[Message]): The complete conversation history
            allocated_tokens (int): Allocated token budget
            query_type (QueryType): The query type for optimization strategy
            
        Returns:
            List[Message]: Optimized conversation history
        """
        if not messages:
            return []
            
        # Different strategies based on query type
        if query_type == QueryType.CONTEXT_DEPENDENT:
            # For context-dependent queries, recent messages are critical
            # Prioritize the most recent messages but keep user-assistant pairs intact
            
            # Ensure we have pairs of messages (user followed by assistant)
            optimized = []
            remaining_tokens = allocated_tokens
            
            # Always include the most recent message
            last_message = messages[-1]
            last_tokens = self.estimate_tokens(last_message.content)
            optimized.append(last_message)
            remaining_tokens -= last_tokens
            
            # Process the rest of the messages in reverse order (most recent first)
            # Keep user-assistant pairs together
            i = len(messages) - 2
            while i >= 0 and remaining_tokens > 0:
                message = messages[i]
                tokens = self.estimate_tokens(message.content)
                
                # If we can't fit this message, try to summarize it
                if tokens > remaining_tokens and tokens > 100:
                    # Create a truncated version that fits
                    truncated = self._truncate_message(message, remaining_tokens)
                    optimized.insert(0, truncated)
                    remaining_tokens = 0
                else:
                    optimized.insert(0, message)
                    remaining_tokens -= tokens
                
                i -= 1
                
        else:
            # For other query types, we want a mix of recent and earlier messages
            
            # Step 1: Always include the most recent messages
            optimized = []
            remaining_tokens = allocated_tokens
            
            # Include the last 2-3 messages always (most recent context)
            recent_count = min(3, len(messages))
            recent_messages = messages[-recent_count:]
            
            for msg in recent_messages:
                tokens = self.estimate_tokens(msg.content)
                optimized.append(msg)
                remaining_tokens -= tokens
            
            # If we have tokens left, add important earlier messages
            if remaining_tokens > 100 and len(messages) > recent_count:
                # Include some messages from the beginning (initial context)
                early_messages = []
                for msg in messages[:3]:  # First 3 messages
                    tokens = self.estimate_tokens(msg.content)
                    if tokens <= remaining_tokens:
                        early_messages.append(msg)
                        remaining_tokens -= tokens
                    else:
                        # Try to include a truncated version
                        if tokens > 100:  # Only truncate substantial messages
                            truncated = self._truncate_message(msg, remaining_tokens)
                            early_messages.append(truncated)
                            remaining_tokens = 0
                        break
                
                # Add early messages at the beginning
                optimized = early_messages + optimized
            
            # If we still have tokens left and enough messages, include some middle context
            if remaining_tokens > 200 and len(messages) > recent_count + 3:
                middle_start = 3
                middle_end = len(messages) - recent_count
                middle_messages = []
                
                # Take messages from the middle, prioritizing user messages
                for i in range(middle_start, middle_end):
                    msg = messages[i]
                    
                    # Prioritize user messages as they contain the questions/requests
                    if msg.role == "user":
                        tokens = self.estimate_tokens(msg.content)
                        if tokens <= remaining_tokens:
                            middle_messages.append(msg)
                            remaining_tokens -= tokens
                        
                        # Also try to include the response to this user message
                        if i + 1 < middle_end and messages[i + 1].role == "assistant":
                            assistant_msg = messages[i + 1]
                            asst_tokens = self.estimate_tokens(assistant_msg.content)
                            
                            if asst_tokens <= remaining_tokens:
                                middle_messages.append(assistant_msg)
                                remaining_tokens -= asst_tokens
                
                # Insert middle messages between early and recent
                if middle_messages:
                    early_count = len(optimized) - recent_count
                    optimized = optimized[:early_count] + middle_messages + optimized[early_count:]
        
        return optimized

    def _truncate_message(self, message: Message, max_tokens: int) -> Message:
        """
        Truncate a message to fit within the specified token limit.
        
        Args:
            message (Message): Message to truncate
            max_tokens (int): Maximum tokens allowed
            
        Returns:
            Message: Truncated message
        """
        content = message.content
        
        # If it's a code-heavy message, preserve code blocks first
        if "```" in content:
            # Try to preserve code blocks
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', content, re.DOTALL)
            if code_blocks:
                # Extract the largest code block
                largest_block = max(code_blocks, key=len)
                
                # If the largest block fits, keep it and truncate the rest
                block_tokens = self.estimate_tokens(largest_block)
                if block_tokens <= max_tokens - 20:  # Allow some tokens for wrapper
                    wrapper_text = "Message contained code (truncated):\n\n```\n"
                    wrapper_end = "\n```"
                    remaining = max_tokens - block_tokens - self.estimate_tokens(wrapper_text + wrapper_end)
                    
                    # If we have tokens left, add some context before the code
                    if remaining > 20:
                        # Find the code block position
                        block_pos = content.find(largest_block)
                        prefix = content[:block_pos].strip()
                        if prefix:
                            # Add as much prefix as we can
                            if self.estimate_tokens(prefix) <= remaining:
                                truncated_content = prefix + "\n\n" + wrapper_text + largest_block + wrapper_end
                            else:
                                # Truncate the prefix
                                words = prefix.split()
                                truncated_prefix = ""
                                for word in words:
                                    if self.estimate_tokens(truncated_prefix + " " + word) > remaining:
                                        break
                                    truncated_prefix += " " + word
                                truncated_content = truncated_prefix.strip() + "...\n\n" + wrapper_text + largest_block + wrapper_end
                        else:
                            truncated_content = wrapper_text + largest_block + wrapper_end
                    else:
                        truncated_content = wrapper_text + largest_block + wrapper_end
                    
                    return Message(role=message.role, content=truncated_content)
        
        # For regular messages, truncate by reducing content
        words = content.split()
        truncated_content = ""
        for word in words:
            if self.estimate_tokens(truncated_content + " " + word) > max_tokens - 3:  # Reserve tokens for ellipsis
                break
            truncated_content += " " + word
        
        truncated_content = truncated_content.strip() + "..."
        return Message(role=message.role, content=truncated_content)

    def optimize_relevant_knowledge(self, items: List[MemoryItem], 
                                   allocated_tokens: int,
                                   query: str) -> List[MemoryItem]:
        """
        Optimize relevant knowledge items to fit within allocated token budget,
        prioritizing most relevant and diverse items.
        
        Args:
            items (List[MemoryItem]): Knowledge items retrieved from long-term memory
            allocated_tokens (int): Allocated token budget
            query (str): The query for relevance scoring
            
        Returns:
            List[MemoryItem]: Optimized knowledge items
        """
        if not items:
            return []
            
        # Step 1: Calculate token counts for each item
        items_with_tokens = []
        for item in items:
            tokens = self.estimate_tokens(item.content)
            items_with_tokens.append((item, tokens))
        
        # Step 2: Score items by relevance to query (if embedding generator available)
        if self.embedding_generator:
            query_embedding = self.embedding_generator.generate(query)
            
            # Calculate relevance scores based on cosine similarity
            items_with_scores = []
            for item, tokens in items_with_tokens:
                if hasattr(item, 'embedding') and item.embedding is not None:
                    similarity = self.embedding_generator.calculate_similarity(
                        query_embedding, item.embedding
                    )
                    items_with_scores.append((item, tokens, similarity))
                else:
                    # Generate embedding if not available
                    embedding = self.embedding_generator.generate(item.content)
                    similarity = self.embedding_generator.calculate_similarity(
                        query_embedding, embedding
                    )
                    items_with_scores.append((item, tokens, similarity))
            
            # Sort by relevance score (descending)
            items_with_scores.sort(key=lambda x: x[2], reverse=True)
        else:
            # If no embedding generator, use the order provided (assumed to be relevance-sorted)
            items_with_scores = [(item, tokens, 1.0) for item, tokens in items_with_tokens]
        
        # Step 3: Select items to include, maintaining diversity and relevance
        selected_items = []
        remaining_tokens = allocated_tokens
        
        # Track already included item types to ensure diversity
        included_types = set()
        
        # First pass: Include highest-scored items of different types
        for item, tokens, score in items_with_scores:
            item_type = item.metadata.get('item_type', 'unknown')
            
            # If we've seen this type before, only include if highly relevant
            if item_type in included_types and score < 0.7:
                continue
                
            if tokens <= remaining_tokens:
                selected_items.append(item)
                remaining_tokens -= tokens
                included_types.add(item_type)
            elif tokens > 100:  # Only try to truncate longer items
                # Try to include a truncated version
                truncated_content = self._truncate_content(item.content, remaining_tokens)
                if truncated_content:
                    truncated_item = MemoryItem(
                        id=item.id,
                        content=truncated_content,
                        metadata=item.metadata.copy(),
                        embedding=item.embedding
                    )
                    selected_items.append(truncated_item)
                    remaining_tokens = 0
                    included_types.add(item_type)
            
            # Stop if we're out of tokens
            if remaining_tokens < 50:
                break
                
        # Second pass: If we still have tokens, include more items
        if remaining_tokens >= 100:
            for item, tokens, score in items_with_scores:
                # Skip already included items
                if item in selected_items:
                    continue
                    
                if tokens <= remaining_tokens:
                    selected_items.append(item)
                    remaining_tokens -= tokens
                
                # Stop if we're out of tokens
                if remaining_tokens < 50:
                    break
        
        return selected_items

    def _truncate_content(self, content: str, max_tokens: int) -> Optional[str]:
        """
        Truncate content to fit within the specified token limit.
        
        Args:
            content (str): Content to truncate
            max_tokens (int): Maximum tokens allowed
            
        Returns:
            Optional[str]: Truncated content or None if can't be meaningfully truncated
        """
        # If the content is too short to meaningfully truncate, return None
        if len(content) < 100 or max_tokens < 50:
            return None
            
        # If it's a code block, try to preserve important parts
        if content.strip().startswith("```") or "```" in content:
            # For code, prioritize function/class definitions and imports
            important_lines = []
            
            # Extract lines that contain key code elements
            for line in content.split("\n"):
                if re.search(r'^(import|from|def|class|function|const|let|var|export|public|private|@)', line.strip()):
                    important_lines.append(line)
            
            # If we have important lines, create a summary with them
            if important_lines:
                summary = "Code summary (truncated):\n\n```\n"
                for line in important_lines:
                    if self.estimate_tokens(summary + line + "\n") <= max_tokens - 5:
                        summary += line + "\n"
                    else:
                        break
                summary += "# ... truncated ...\n```"
                
                if self.estimate_tokens(summary) <= max_tokens:
                    return summary
        
        # For regular text, truncate by preserving the beginning
        words = content.split()
        truncated = ""
        for word in words:
            if self.estimate_tokens(truncated + " " + word) > max_tokens - 3:  # Reserve tokens for ellipsis
                break
            truncated += " " + word
        
        truncated = truncated.strip() + "..."
        return truncated if truncated != "..." else None

    def build_optimized_context(self, 
                              query: str,
                              system_prompt: str,
                              messages: List[Message],
                              knowledge_items: List[MemoryItem],
                              working_memory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build an optimized context for the LLM using the specified components and dynamic allocation.
        
        Args:
            query (str): The current query
            system_prompt (str): The system prompt
            messages (List[Message]): Conversation history
            knowledge_items (List[MemoryItem]): Relevant knowledge items
            working_memory (Dict[str, Any]): Working memory items
            
        Returns:
            Dict[str, Any]: Optimized context with components fitting within the context window
        """
        # Step 1: Classify the query to determine optimal allocation strategy
        query_type, complexity = self.classify_query(query, messages)
        
        # Step 2: Get token allocations based on query type and complexity
        allocations = self.get_token_allocations(query_type, complexity)
        
        # Step 3: Calculate token budgets for each component
        system_tokens = int(self.model_context_size * allocations["system_prompt"])
        message_tokens = int(self.model_context_size * allocations["recent_messages"])
        knowledge_tokens = int(self.model_context_size * allocations["relevant_knowledge"])
        working_memory_tokens = int(self.model_context_size * allocations["working_memory"])
        
        # Step 4: Check system prompt fits in budget, truncate if needed
        system_prompt_tokens = self.estimate_tokens(system_prompt)
        if system_prompt_tokens > system_tokens:
            logger.warning(f"System prompt exceeds allocated tokens: {system_prompt_tokens} > {system_tokens}")
            # We don't truncate system prompt here, but we log the issue
        
        # Step 5: Optimize conversation history
        optimized_messages = self.optimize_conversation_history(
            messages, message_tokens, query_type
        )
        
        # Step 6: Optimize relevant knowledge
        optimized_knowledge = self.optimize_relevant_knowledge(
            knowledge_items, knowledge_tokens, query
        )
        
        # Step 7: Prepare working memory (simple serialization with token limit)
        serialized_working_memory = {}
        remaining_tokens = working_memory_tokens
        
        for key, value in working_memory.items():
            serialized = str(value)
            tokens = self.estimate_tokens(serialized)
            
            if tokens <= remaining_tokens:
                serialized_working_memory[key] = value
                remaining_tokens -= tokens
            else:
                # If value is too large, truncate it
                if tokens > 100:  # Only truncate substantial values
                    truncated = serialized[:int(len(serialized) * (remaining_tokens / tokens))] + "..."
                    serialized_working_memory[key] = truncated
                    remaining_tokens = 0
                break
        
        # Step 8: Calculate total tokens used and context utilization
        message_content = "\n".join([m.content for m in optimized_messages])
        knowledge_content = "\n".join([k.content for k in optimized_knowledge])
        working_memory_content = str(serialized_working_memory)
        
        total_tokens = (
            system_prompt_tokens +
            self.estimate_tokens(message_content) +
            self.estimate_tokens(knowledge_content) +
            self.estimate_tokens(working_memory_content)
        )
        
        utilization = total_tokens / self.model_context_size
        
        # Log the utilization metrics
        logger.info(f"Context utilization: {utilization:.1%} ({total_tokens}/{self.model_context_size} tokens)")
        logger.info(f"Query type: {query_type.name}, Complexity: {complexity.name}")
        logger.info(f"System prompt: {system_prompt_tokens} tokens")
        logger.info(f"Messages: {self.estimate_tokens(message_content)} tokens ({len(optimized_messages)} messages)")
        logger.info(f"Knowledge: {self.estimate_tokens(knowledge_content)} tokens ({len(optimized_knowledge)} items)")
        logger.info(f"Working memory: {self.estimate_tokens(working_memory_content)} tokens ({len(serialized_working_memory)} items)")
        
        # Return the optimized context
        return {
            "system_prompt": system_prompt,
            "messages": optimized_messages,
            "knowledge_items": optimized_knowledge,
            "working_memory": serialized_working_memory,
            "metrics": {
                "total_tokens": total_tokens,
                "utilization": utilization,
                "query_type": query_type.name,
                "complexity": complexity.name
            }
        } 