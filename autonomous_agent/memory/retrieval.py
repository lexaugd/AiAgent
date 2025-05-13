"""
Advanced memory retrieval mechanisms for the Autonomous Coding Agent.

This module provides specialized query mechanisms, ranking algorithms,
and context-aware retrieval techniques for optimizing memory access.
"""

import re
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from loguru import logger

import numpy as np

from memory.types import MemoryType, MemoryMetadata, ExtendedMemoryItem
from memory.long_term import LongTermMemory, MemoryItem
from memory.embeddings import EmbeddingGenerator, EmbeddingUtils


@dataclass
class RetrievalResult:
    """
    Result of a memory retrieval operation.
    """
    items: List[Union[MemoryItem, ExtendedMemoryItem]]
    query: str
    total_found: int
    retrieval_time: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}


class QueryExpansion:
    """
    Techniques for expanding queries to improve memory retrieval.
    """
    
    @staticmethod
    def expand_coding_query(query: str) -> List[str]:
        """
        Expand a coding-related query with related terms.
        
        Args:
            query (str): The original query.
            
        Returns:
            List[str]: Expanded queries.
        """
        expansions = [query]
        
        # Look for programming language mentions
        language_patterns = {
            "python": ["python", "py", "pytest", "django", "flask", "numpy", "pandas"],
            "javascript": ["javascript", "js", "node", "nodejs", "react", "vue", "angular"],
            "typescript": ["typescript", "ts", "tsx", "angular", "nestjs"],
            "java": ["java", "spring", "maven", "gradle", "junit"],
            "c#": ["c#", "csharp", ".net", "asp.net", "dotnet"],
            "go": ["golang", "go lang", "go programming"],
            "rust": ["rust", "cargo", "rustc"],
            "sql": ["sql", "mysql", "postgresql", "sqlite", "database query"]
        }
        
        # Extract any language mentions
        mentioned_languages = []
        for lang, terms in language_patterns.items():
            if any(term in query.lower() for term in terms):
                mentioned_languages.append(lang)
        
        # Add language-specific expansions
        for lang in mentioned_languages:
            # Add common problems/patterns for that language
            if lang == "python":
                expansions.append(f"{query} python function")
                expansions.append(f"{query} python class")
                if "error" in query.lower() or "exception" in query.lower():
                    expansions.append(f"{query} python traceback")
            elif lang == "javascript":
                expansions.append(f"{query} javascript function")
                expansions.append(f"{query} javascript async")
                if "error" in query.lower():
                    expansions.append(f"{query} javascript exception")
            # Add more language-specific expansions as needed
        
        # Look for common programming concepts and expand
        concepts = {
            "function": ["function", "method", "def ", "function(", "()"],
            "class": ["class", "class ", "object", "instance"],
            "error": ["error", "exception", "traceback", "stack trace", "fail", "bug"],
            "api": ["api", "endpoint", "rest", "http", "request"],
            "data": ["data", "json", "dict", "list", "array", "object"],
            "database": ["database", "db", "sql", "query", "table"]
        }
        
        # Extract mentioned concepts
        mentioned_concepts = []
        for concept, terms in concepts.items():
            if any(term in query.lower() for term in terms):
                mentioned_concepts.append(concept)
        
        # Add concept-specific expansions
        for concept in mentioned_concepts:
            if concept == "function":
                expansions.append(f"{query} implementation")
                expansions.append(f"{query} parameters")
            elif concept == "error":
                expansions.append(f"{query} solution")
                expansions.append(f"{query} fix")
            elif concept == "api":
                expansions.append(f"{query} endpoint")
                expansions.append(f"{query} request response")
        
        return expansions
    
    @staticmethod
    def expand_with_code_patterns(query: str) -> List[str]:
        """
        Expand a query by identifying potential code patterns.
        
        Args:
            query (str): The original query.
            
        Returns:
            List[str]: Expanded queries.
        """
        expansions = [query]
        
        # Extract potential code snippets in backticks or code blocks
        code_snippets = re.findall(r'`(.*?)`|```(.*?)```', query, re.DOTALL)
        
        # Flatten the list of tuples and remove empty strings
        code_snippets = [snippet for group in code_snippets for snippet in group if snippet]
        
        if code_snippets:
            # Extract identifiers from code snippets
            for snippet in code_snippets:
                # Extract function names, variable names, class names, etc.
                identifiers = re.findall(r'\b([a-zA-Z_]\w*)\b', snippet)
                
                # Add queries focusing on specific identifiers
                for identifier in identifiers:
                    if len(identifier) > 2 and identifier not in ['if', 'for', 'def', 'class', 'var', 'let', 'const', 'function']:
                        expansions.append(f"{query} {identifier}")
        
        # Extract potential API patterns (e.g., HTTP methods, endpoints)
        api_patterns = re.findall(r'\b(GET|POST|PUT|DELETE|PATCH)\b\s+([/\w]+)', query, re.IGNORECASE)
        for method, endpoint in api_patterns:
            expansions.append(f"{method} {endpoint}")
            expansions.append(f"API {endpoint}")
        
        return expansions
    
    @staticmethod
    def expand_documentation_query(query: str) -> List[str]:
        """
        Expand a documentation-related query.
        
        Args:
            query (str): The original query.
            
        Returns:
            List[str]: Expanded queries.
        """
        expansions = [query]
        
        # Add documentation-specific terms
        for prefix in ["how to", "guide", "documentation", "example", "tutorial"]:
            if prefix not in query.lower():
                expansions.append(f"{prefix} {query}")
        
        # If asking about usage, add examples
        if "how" in query.lower() or "use" in query.lower():
            expansions.append(f"{query} example")
            expansions.append(f"{query} usage")
        
        return expansions


class ContextAwareRetrieval:
    """
    Implements context-aware retrieval strategies for more relevant results.
    """
    
    def __init__(
        self,
        long_term_memory: LongTermMemory,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        """
        Initialize the context-aware retrieval system.
        
        Args:
            long_term_memory (LongTermMemory): The long-term memory to query.
            embedding_generator (EmbeddingGenerator, optional): The embedding generator to use.
        """
        self.long_term_memory = long_term_memory
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
    
    def retrieve(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        item_type: Optional[str] = None,
        n_results: int = 10,
        expand_query: bool = True,
        rerank: bool = True
    ) -> RetrievalResult:
        """
        Retrieve items from memory with context-awareness.
        
        Args:
            query (str): The query to search for.
            context (Dict[str, Any], optional): Additional context for the query.
            item_type (str, optional): Type of items to retrieve.
            n_results (int): Number of results to return.
            expand_query (bool): Whether to expand the query.
            rerank (bool): Whether to rerank results by relevance.
            
        Returns:
            RetrievalResult: The retrieval results.
        """
        start_time = time.time()
        
        # Process and prepare the query
        processed_query, expanded_queries = self._prepare_query(query, context, expand_query)
        
        # Prepare filter for item_type if provided
        filters = None
        if item_type:
            filters = {"item_type": {"$eq": item_type}}
        
        # Collect results from all queries
        all_results = []
        
        # First query with the processed query
        primary_results = self.long_term_memory.query(
            processed_query, 
            filters=filters, 
            n_results=n_results
        )
        all_results.extend(primary_results)
        
        # Use expanded queries if enabled and get additional results
        if expand_query and expanded_queries:
            for expanded_query in expanded_queries[:2]:  # Limit to top 2 expansions
                # Get fewer results for expanded queries
                expanded_results = self.long_term_memory.query(
                    expanded_query,
                    filters=filters,
                    n_results=max(3, n_results // 2)
                )
                all_results.extend(expanded_results)
        
        # Remove duplicates by item_id
        unique_results = {}
        for item in all_results:
            if item.item_id not in unique_results:
                unique_results[item.item_id] = item
        
        all_results = list(unique_results.values())
        
        # Rerank results if enabled
        if rerank and len(all_results) > 1:
            all_results = self._rerank_results(processed_query, all_results, context)
        
        # Limit to n_results
        final_results = all_results[:n_results]
        
        # Record the retrieval
        retrieval_time = time.time() - start_time
        
        return RetrievalResult(
            items=final_results,
            query=processed_query,
            total_found=len(all_results),
            retrieval_time=retrieval_time,
            metadata={
                "expanded_queries": expanded_queries,
                "context": context,
                "item_type": item_type
            }
        )
    
    def retrieve_by_type(
        self,
        query: str,
        memory_types: List[str],
        context: Optional[Dict[str, Any]] = None,
        n_results: int = 10
    ) -> Dict[str, RetrievalResult]:
        """
        Retrieve items from memory by different memory types.
        
        Args:
            query (str): The query to search for.
            memory_types (List[str]): Types of memory to retrieve.
            context (Dict[str, Any], optional): Additional context for the query.
            n_results (int): Number of results to return per type.
            
        Returns:
            Dict[str, RetrievalResult]: Results for each memory type.
        """
        results = {}
        
        for memory_type in memory_types:
            # Get results for this type
            type_results = self.retrieve(
                query=query,
                context=context,
                item_type=memory_type,
                n_results=n_results
            )
            
            results[memory_type] = type_results
        
        return results
    
    def retrieve_code_examples(
        self,
        query: str,
        language: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        n_results: int = 5
    ) -> RetrievalResult:
        """
        Retrieve code examples with specialized processing.
        
        Args:
            query (str): The query to search for.
            language (str, optional): The programming language.
            context (Dict[str, Any], optional): Additional context for the query.
            n_results (int): Number of results to return.
            
        Returns:
            RetrievalResult: The retrieval results.
        """
        # Prepare context for code queries
        code_context = context or {}
        if language:
            code_context["language"] = language
        
        # Perform specialized query expansion for code
        expanded_queries = QueryExpansion.expand_coding_query(query)
        expanded_queries.extend(QueryExpansion.expand_with_code_patterns(query))
        
        # Process the query for code retrieval
        processed_query = self.embedding_generator._preprocess_query(query)
        
        # Prepare filters for the query - use the proper ChromaDB operator format
        chroma_filter = {"$and": [{"item_type": {"$eq": "code"}}]}
        if language:
            chroma_filter["$and"].append({"language": {"$eq": language}})
        
        start_time = time.time()
        
        # Collect results from all queries
        all_results = []
        
        # First query with the processed query
        primary_results = self.long_term_memory.query(
            processed_query, 
            filters=chroma_filter, 
            n_results=n_results
        )
        all_results.extend(primary_results)
        
        # Use expanded queries to get additional results
        for expanded_query in expanded_queries[:3]:  # Limit to top 3 expansions
            # Get fewer results for expanded queries
            expanded_results = self.long_term_memory.query(
                expanded_query,
                filters=chroma_filter,
                n_results=max(2, n_results // 2)
            )
            all_results.extend(expanded_results)
        
        # Remove duplicates
        unique_results = {}
        for item in all_results:
            if item.item_id not in unique_results:
                unique_results[item.item_id] = item
        
        all_results = list(unique_results.values())
        
        # Rerank using code-specific relevance
        reranked_results = self._rerank_code_results(processed_query, all_results, language)
        
        # Limit to n_results
        final_results = reranked_results[:n_results]
        
        # Record the retrieval
        retrieval_time = time.time() - start_time
        
        return RetrievalResult(
            items=final_results,
            query=processed_query,
            total_found=len(all_results),
            retrieval_time=retrieval_time,
            metadata={
                "expanded_queries": expanded_queries,
                "language": language,
                "context": code_context
            }
        )
    
    def _prepare_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        expand: bool = True
    ) -> Tuple[str, List[str]]:
        """
        Prepare a query with context information and optional expansion.
        
        Args:
            query (str): The original query.
            context (Dict[str, Any], optional): Additional context.
            expand (bool): Whether to expand the query.
            
        Returns:
            Tuple[str, List[str]]: (processed_query, expanded_queries)
        """
        # Process the query
        processed_query = query.strip()
        
        # Incorporate context if available
        if context:
            # Extract relevant context elements
            if "current_task" in context:
                task_desc = context["current_task"]
                if not any(term in processed_query.lower() for term in task_desc.lower().split()):
                    processed_query = f"{processed_query} {task_desc}"
            
            if "language" in context:
                language = context["language"]
                if language and language.lower() not in processed_query.lower():
                    processed_query = f"{processed_query} {language}"
        
        # Generate query expansions if enabled
        expanded_queries = []
        if expand:
            # General expansion
            expanded_queries.extend(QueryExpansion.expand_coding_query(query))
            expanded_queries.extend(QueryExpansion.expand_with_code_patterns(query))
            expanded_queries.extend(QueryExpansion.expand_documentation_query(query))
            
            # Remove duplicates and the original query
            expanded_queries = [q for q in expanded_queries if q != query]
            expanded_queries = list(dict.fromkeys(expanded_queries))  # Remove duplicates while preserving order
        
        return processed_query, expanded_queries
    
    def _rerank_results(
        self, 
        query: str, 
        results: List[Union[MemoryItem, ExtendedMemoryItem]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Union[MemoryItem, ExtendedMemoryItem]]:
        """
        Rerank results by their relevance to the query and context.
        
        Args:
            query (str): The processed query.
            results (List[Union[MemoryItem, ExtendedMemoryItem]]): The results to rerank.
            context (Dict[str, Any], optional): Additional context.
            
        Returns:
            List[Union[MemoryItem, ExtendedMemoryItem]]: Reranked results.
        """
        if not results:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate(query, content_type="query")
        
        # Calculate relevance scores
        scored_results = []
        for item in results:
            # Get or generate item embedding
            item_embedding = getattr(item, "embedding", None)
            if item_embedding is None:
                # Generate embedding if not available
                item_embedding = self.embedding_generator.generate(item.content, content_type="default")
            
            # Calculate semantic similarity
            similarity = EmbeddingUtils.cosine_similarity(query_embedding, item_embedding)
            
            # Apply context-based boosts
            boost = 0.0
            
            # Boost for recency
            if hasattr(item, "metadata") and hasattr(item.metadata, "last_accessed"):
                # Recency boost - fresher memories get higher scores
                last_access = item.metadata.last_accessed or 0
                age_days = (time.time() - last_access) / (24 * 60 * 60)
                if age_days < 7:  # Boost for items accessed in the last week
                    recency_boost = max(0.0, 0.1 * (1.0 - age_days / 7.0))
                    boost += recency_boost
            
            # Boost for popularity
            if hasattr(item, "metadata") and hasattr(item.metadata, "access_count"):
                # Popularity boost - frequently accessed memories get higher scores
                access_count = item.metadata.access_count or 0
                popularity_boost = min(0.1, 0.01 * access_count)
                boost += popularity_boost
            
            # Apply context-specific boosts
            if context:
                # Language match boost for code
                if (
                    "language" in context and 
                    hasattr(item, "metadata") and 
                    hasattr(item.metadata, "custom") and
                    item.metadata.custom
                ):
                    if "language" in item.metadata.custom and item.metadata.custom["language"] == context["language"]:
                        boost += 0.15
            
            # Final score with boost
            final_score = similarity + boost
            
            scored_results.append((item, final_score))
        
        # Sort by score (highest first)
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return reranked items
        return [item for item, _ in scored_results]
    
    def _rerank_code_results(
        self, 
        query: str, 
        results: List[Union[MemoryItem, ExtendedMemoryItem]],
        language: Optional[str] = None
    ) -> List[Union[MemoryItem, ExtendedMemoryItem]]:
        """
        Rerank code results using specialized ranking for code.
        
        Args:
            query (str): The processed query.
            results (List[Union[MemoryItem, ExtendedMemoryItem]]): The results to rerank.
            language (str, optional): The programming language.
            
        Returns:
            List[Union[MemoryItem, ExtendedMemoryItem]]: Reranked results.
        """
        if not results:
            return []
        
        # Generate query embedding with code preprocessing
        query_embedding = self.embedding_generator.generate(query, content_type="query")
        
        # Extract key terms from the query
        query_terms = set(re.findall(r'\b([a-zA-Z_]\w{2,})\b', query.lower()))
        
        # Calculate relevance scores with code-specific factors
        scored_results = []
        for item in results:
            # Get or generate item embedding
            item_embedding = getattr(item, "embedding", None)
            if item_embedding is None:
                # Generate embedding with code preprocessing
                item_embedding = self.embedding_generator.generate(item.content, content_type="code")
            
            # Calculate semantic similarity
            similarity = EmbeddingUtils.cosine_similarity(query_embedding, item_embedding)
            
            # Code-specific relevance factors
            boost = 0.0
            
            # Term matching boost
            content_lower = item.content.lower()
            term_matches = sum(1 for term in query_terms if term in content_lower)
            term_match_boost = min(0.2, 0.03 * term_matches)
            boost += term_match_boost
            
            # Language match boost
            if language and hasattr(item, "metadata") and item.metadata:
                meta = item.metadata
                if hasattr(meta, "custom") and meta.custom and "language" in meta.custom:
                    if meta.custom["language"] == language:
                        boost += 0.2
                        
            # Function/class definition boost for relevant queries
            if "function" in query.lower() or "class" in query.lower() or "implement" in query.lower():
                if (
                    re.search(r'(def|function|class)\s+\w+', content_lower) or 
                    re.search(r'(\w+)\s+=\s+function', content_lower)
                ):
                    boost += 0.15
            
            # Example/sample boost for relevant queries
            if "example" in query.lower() or "sample" in query.lower() or "how to" in query.lower():
                if "example" in content_lower or "sample" in content_lower or "usage" in content_lower:
                    boost += 0.1
            
            # Final score with boost
            final_score = similarity + boost
            
            scored_results.append((item, final_score))
        
        # Sort by score (highest first)
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return reranked items
        return [item for item, _ in scored_results]


class MultiSourceRetrieval:
    """
    Integrates memory retrieval from multiple sources for comprehensive results.
    """
    
    def __init__(
        self,
        context_retrieval: ContextAwareRetrieval,
        memory_manager: Any  # Actual type is MemoryManager, but avoiding circular imports
    ):
        """
        Initialize multi-source retrieval.
        
        Args:
            context_retrieval (ContextAwareRetrieval): Context-aware retrieval system.
            memory_manager: Memory manager that coordinates different memory systems.
        """
        self.context_retrieval = context_retrieval
        self.memory_manager = memory_manager
    
    def retrieve_comprehensive(
        self,
        query: str,
        max_items: int = 10,
        include_types: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve comprehensive information from multiple memory sources.
        
        Args:
            query (str): The query to search for.
            max_items (int): Maximum total items to retrieve.
            include_types (List[str], optional): Memory types to include.
            context (Dict[str, Any], optional): Additional context for retrieval.
            
        Returns:
            Dict[str, Any]: Comprehensive results from different sources.
        """
        types_to_include = include_types or [
            "code", "documentation", "concept", 
            "error", "task", "conversation"
        ]
        
        # Get working memory relevant to the query
        working_memory = {
            k: v for k, v in self.memory_manager.working_memory.items()
            if any(term in k.lower() for term in query.lower().split())
        }
        
        # Allocate items per type based on available types
        items_per_type = max(2, max_items // len(types_to_include))
        
        # Retrieve from long-term memory by types
        long_term_results = self.context_retrieval.retrieve_by_type(
            query=query,
            memory_types=types_to_include,
            context=context,
            n_results=items_per_type
        )
        
        # Get conversation context from short-term memory
        recent_messages = self.memory_manager.get_conversation_history(5)
        
        # Combine all results
        comprehensive_results = {
            "query": query,
            "working_memory": working_memory,
            "long_term_memory": {
                memory_type: {
                    "items": [item.to_dict() for item in result.items],
                    "total_found": result.total_found,
                    "retrieval_time": result.retrieval_time
                }
                for memory_type, result in long_term_results.items()
            },
            "conversation_context": recent_messages,
            "total_items": sum(len(result.items) for result in long_term_results.values())
        }
        
        return comprehensive_results 