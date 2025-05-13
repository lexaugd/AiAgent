"""
Embedding utilities for the Autonomous Coding Agent.

This module provides utilities for generating and manipulating embeddings,
with special focus on code-specific embeddings.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from loguru import logger

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Class for generating embeddings with specialized preprocessing for different content types.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name (str): The name of the SentenceTransformer model to use.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        # Register content type handlers
        self.content_handlers = {
            "code": self._preprocess_code,
            "documentation": self._preprocess_documentation,
            "concept": self._preprocess_concept,
            "error": self._preprocess_error,
            "query": self._preprocess_query,
            "default": lambda x: x  # Identity function for default
        }
        
        logger.info(f"Initialized EmbeddingGenerator with model: {model_name}")
    
    def generate(self, content: str, content_type: str = "default") -> List[float]:
        """
        Generate an embedding for the given content.
        
        Args:
            content (str): The content to embed.
            content_type (str): The type of content for specialized preprocessing.
            
        Returns:
            List[float]: The embedding vector.
        """
        # Apply content-specific preprocessing
        handler = self.content_handlers.get(content_type, self.content_handlers["default"])
        processed_content = handler(content)
        
        # Generate embedding
        embedding = self.model.encode(processed_content)
        
        # Convert to list and return
        return embedding.tolist()
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate the cosine similarity between two embeddings.
        
        Args:
            embedding1 (List[float]): The first embedding.
            embedding2 (List[float]): The second embedding.
            
        Returns:
            float: The cosine similarity between the embeddings.
        """
        return EmbeddingUtils.cosine_similarity(embedding1, embedding2)
    
    def generate_batch(self, contents: List[str], content_types: Optional[List[str]] = None) -> List[List[float]]:
        """
        Generate embeddings for a batch of content.
        
        Args:
            contents (List[str]): The contents to embed.
            content_types (List[str], optional): The types of content for specialized preprocessing.
            
        Returns:
            List[List[float]]: The embedding vectors.
        """
        if content_types is None:
            content_types = ["default"] * len(contents)
        
        # Apply content-specific preprocessing
        processed_contents = []
        for content, content_type in zip(contents, content_types):
            handler = self.content_handlers.get(content_type, self.content_handlers["default"])
            processed_content = handler(content)
            processed_contents.append(processed_content)
        
        # Generate embeddings
        embeddings = self.model.encode(processed_contents)
        
        # Convert to list and return
        return [embedding.tolist() for embedding in embeddings]
    
    def _preprocess_code(self, code: str) -> str:
        """
        Preprocess code for better embedding quality.
        
        Args:
            code (str): The code to preprocess.
            
        Returns:
            str: The preprocessed code.
        """
        # Remove comments
        # Simple regex for C-style comments, Python comments, and others
        code = re.sub(r'//.*?$|/\*.*?\*/|#.*?$', '', code, flags=re.MULTILINE | re.DOTALL)
        
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        
        # Extract function/class definitions to emphasize them
        definitions = []
        
        # Python function/class definitions
        python_defs = re.findall(r'(def|class)\s+([a-zA-Z0-9_]+)', code)
        definitions.extend([f"{d[0]} {d[1]}" for d in python_defs])
        
        # JavaScript/TypeScript function/class definitions
        js_defs = re.findall(r'(function|class)\s+([a-zA-Z0-9_]+)', code)
        definitions.extend([f"{d[0]} {d[1]}" for d in js_defs])
        
        # Join with the original code (giving extra weight to definitions)
        if definitions:
            preprocessed = " ".join(definitions) + " " + code
        else:
            preprocessed = code
        
        return preprocessed
    
    def _preprocess_documentation(self, doc: str) -> str:
        """
        Preprocess documentation for better embedding quality.
        
        Args:
            doc (str): The documentation to preprocess.
            
        Returns:
            str: The preprocessed documentation.
        """
        # Remove markdown formatting
        doc = re.sub(r'##+', '', doc)  # Remove headers
        doc = re.sub(r'\*\*|__', '', doc)  # Remove bold
        doc = re.sub(r'\*|_', '', doc)  # Remove italics
        doc = re.sub(r'```.*?```', '', doc, flags=re.DOTALL)  # Remove code blocks
        
        # Normalize whitespace
        doc = re.sub(r'\s+', ' ', doc)
        
        return doc
    
    def _preprocess_concept(self, concept: str) -> str:
        """
        Preprocess concept descriptions for better embedding quality.
        
        Args:
            concept (str): The concept description to preprocess.
            
        Returns:
            str: The preprocessed concept description.
        """
        # For concepts, we want to emphasize keywords
        # Simple heuristic: capitalize words that might be important
        words = concept.split()
        for i, word in enumerate(words):
            if len(word) > 4 and word[0].isalpha():  # Potential important term
                words[i] = word.capitalize()
        
        return " ".join(words)
    
    def _preprocess_error(self, error: str) -> str:
        """
        Preprocess error messages for better embedding quality.
        
        Args:
            error (str): The error message to preprocess.
            
        Returns:
            str: The preprocessed error message.
        """
        # Extract error type and message
        error_type_match = re.search(r'([A-Za-z]+Error|Exception):', error)
        error_type = error_type_match.group(1) if error_type_match else ""
        
        # Remove line numbers and file paths
        error = re.sub(r'File ".*?", line \d+', '', error)
        
        # Emphasize error type if found
        if error_type:
            preprocessed = f"{error_type} {error_type} {error}"
        else:
            preprocessed = error
        
        return preprocessed
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess queries for better embedding quality.
        
        Args:
            query (str): The query to preprocess.
            
        Returns:
            str: The preprocessed query.
        """
        # Extract potential code snippets
        code_snippets = re.findall(r'`(.*?)`', query)
        
        # Process the query without the code snippets
        clean_query = re.sub(r'`.*?`', '', query)
        
        # Process any code snippets separately
        processed_snippets = [self._preprocess_code(snippet) for snippet in code_snippets]
        
        # Combine and return
        return clean_query + " " + " ".join(processed_snippets)


class CodeChunker:
    """
    Class for chunking code into semantically meaningful units for embedding.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the code chunker.
        
        Args:
            chunk_size (int): Maximum size of each chunk in characters.
            overlap (int): Overlap between chunks in characters.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Register language-specific chunkers
        self.language_chunkers = {
            "python": self._chunk_python,
            "javascript": self._chunk_javascript,
            "typescript": self._chunk_javascript,  # Use the same chunker for TS
            "java": self._chunk_java,
            "c": self._chunk_c_family,
            "cpp": self._chunk_c_family,
            "csharp": self._chunk_c_family,
            "default": self._chunk_default
        }
        
        logger.info(f"Initialized CodeChunker with chunk_size={chunk_size}, overlap={overlap}")
    
    def chunk(self, code: str, language: str = "default") -> List[Dict[str, Any]]:
        """
        Chunk the given code into semantically meaningful units.
        
        Args:
            code (str): The code to chunk.
            language (str): The programming language of the code.
            
        Returns:
            List[Dict[str, Any]]: The code chunks with metadata.
        """
        chunker = self.language_chunkers.get(language, self.language_chunkers["default"])
        return chunker(code)
    
    def _chunk_default(self, code: str) -> List[Dict[str, Any]]:
        """
        Default chunking strategy for unknown languages.
        
        Args:
            code (str): The code to chunk.
            
        Returns:
            List[Dict[str, Any]]: The code chunks with metadata.
        """
        chunks = []
        code_lines = code.split('\n')
        
        chunk_start = 0
        current_chunk = []
        current_size = 0
        
        for i, line in enumerate(code_lines):
            line_size = len(line)
            
            # If adding this line would exceed the chunk size, create a new chunk
            if current_size + line_size > self.chunk_size and current_chunk:
                chunks.append({
                    "content": '\n'.join(current_chunk),
                    "metadata": {
                        "start_line": chunk_start,
                        "end_line": i - 1,
                        "line_count": i - chunk_start
                    }
                })
                
                # Start a new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.overlap // len(line))
                current_chunk = current_chunk[overlap_start:]
                chunk_start = i - len(current_chunk)
                current_size = sum(len(l) for l in current_chunk)
            
            current_chunk.append(line)
            current_size += line_size
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append({
                "content": '\n'.join(current_chunk),
                "metadata": {
                    "start_line": chunk_start,
                    "end_line": len(code_lines) - 1,
                    "line_count": len(code_lines) - chunk_start
                }
            })
        
        return chunks
    
    def _chunk_python(self, code: str) -> List[Dict[str, Any]]:
        """
        Python-specific chunking strategy that respects function and class boundaries.
        
        Args:
            code (str): The Python code to chunk.
            
        Returns:
            List[Dict[str, Any]]: The code chunks with metadata.
        """
        chunks = []
        code_lines = code.split('\n')
        
        # Identify function and class definitions
        boundaries = []
        in_func_or_class = False
        current_indent = 0
        start_line = 0
        
        for i, line in enumerate(code_lines):
            stripped = line.lstrip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue
            
            # Calculate current indentation
            indent = len(line) - len(stripped)
            
            # Check for function or class definition
            if re.match(r'^(def|class)\s+', stripped):
                if not in_func_or_class:
                    # Start of a new top-level function or class
                    if current_indent == 0:
                        # If we were tracking a previous section, end it
                        if start_line < i:
                            boundaries.append((start_line, i - 1))
                        start_line = i
                    
                    in_func_or_class = True
                    current_indent = indent
            
            # Check for end of function or class based on indentation
            elif in_func_or_class and indent <= current_indent and stripped:
                in_func_or_class = False
                boundaries.append((start_line, i - 1))
                start_line = i
                current_indent = indent
        
        # Add the last section
        if start_line < len(code_lines):
            boundaries.append((start_line, len(code_lines) - 1))
        
        # Create chunks based on boundaries
        for start, end in boundaries:
            chunk_content = '\n'.join(code_lines[start:end+1])
            
            # If the chunk is too large, use the default chunker
            if len(chunk_content) > self.chunk_size * 1.5:
                sub_chunks = self._chunk_default(chunk_content)
                for sub_chunk in sub_chunks:
                    sub_chunk["metadata"]["start_line"] += start
                    sub_chunk["metadata"]["end_line"] += start
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    "content": chunk_content,
                    "metadata": {
                        "start_line": start,
                        "end_line": end,
                        "line_count": end - start + 1
                    }
                })
        
        return chunks
    
    def _chunk_javascript(self, code: str) -> List[Dict[str, Any]]:
        """
        JavaScript/TypeScript-specific chunking strategy.
        
        Args:
            code (str): The JavaScript code to chunk.
            
        Returns:
            List[Dict[str, Any]]: The code chunks with metadata.
        """
        chunks = []
        code_lines = code.split('\n')
        
        # Identify function, class, and object definitions
        boundaries = []
        bracket_stack = []
        start_line = 0
        
        for i, line in enumerate(code_lines):
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('//'):
                continue
            
            # Count opening and closing brackets to track scope
            opening_brackets = stripped.count('{')
            closing_brackets = stripped.count('}')
            
            # Track bracket balance
            for _ in range(opening_brackets):
                bracket_stack.append(i)
            
            for _ in range(closing_brackets):
                if bracket_stack:
                    bracket_stack.pop()
                
                # If bracket stack is empty, it might be the end of a function or class
                if not bracket_stack and closing_brackets > 0:
                    # If we have tracked a section, end it
                    if start_line < i:
                        boundaries.append((start_line, i))
                    start_line = i + 1
        
        # Add the last section if needed
        if start_line < len(code_lines):
            boundaries.append((start_line, len(code_lines) - 1))
        
        # Create chunks based on boundaries
        for start, end in boundaries:
            chunk_content = '\n'.join(code_lines[start:end+1])
            
            # If the chunk is too large, use the default chunker
            if len(chunk_content) > self.chunk_size * 1.5:
                sub_chunks = self._chunk_default(chunk_content)
                for sub_chunk in sub_chunks:
                    sub_chunk["metadata"]["start_line"] += start
                    sub_chunk["metadata"]["end_line"] += start
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    "content": chunk_content,
                    "metadata": {
                        "start_line": start,
                        "end_line": end,
                        "line_count": end - start + 1
                    }
                })
        
        return chunks
    
    def _chunk_java(self, code: str) -> List[Dict[str, Any]]:
        """
        Java-specific chunking strategy.
        
        Args:
            code (str): The Java code to chunk.
            
        Returns:
            List[Dict[str, Any]]: The code chunks with metadata.
        """
        # Java chunking is similar to C-family languages
        return self._chunk_c_family(code)
    
    def _chunk_c_family(self, code: str) -> List[Dict[str, Any]]:
        """
        C-family language chunking strategy (C, C++, C#).
        
        Args:
            code (str): The code to chunk.
            
        Returns:
            List[Dict[str, Any]]: The code chunks with metadata.
        """
        chunks = []
        code_lines = code.split('\n')
        
        # Identify function and class definitions
        boundaries = []
        bracket_stack = []
        start_line = 0
        in_function = False
        
        for i, line in enumerate(code_lines):
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
                continue
            
            # Look for function or class definitions
            if not in_function and re.search(r'(\w+\s+\w+\s*\([^)]*\)\s*{)|(\w+\s+\w+\s*:)', stripped):
                in_function = True
                if start_line < i:
                    boundaries.append((start_line, i - 1))
                start_line = i
            
            # Count opening and closing brackets to track scope
            opening_brackets = stripped.count('{')
            closing_brackets = stripped.count('}')
            
            # Track bracket balance
            for _ in range(opening_brackets):
                bracket_stack.append(i)
            
            for _ in range(closing_brackets):
                if bracket_stack:
                    bracket_stack.pop()
                
                # If bracket stack is empty, it might be the end of a function or class
                if not bracket_stack and closing_brackets > 0 and in_function:
                    in_function = False
                    boundaries.append((start_line, i))
                    start_line = i + 1
        
        # Add the last section if needed
        if start_line < len(code_lines):
            boundaries.append((start_line, len(code_lines) - 1))
        
        # Create chunks based on boundaries
        for start, end in boundaries:
            chunk_content = '\n'.join(code_lines[start:end+1])
            
            # If the chunk is too large, use the default chunker
            if len(chunk_content) > self.chunk_size * 1.5:
                sub_chunks = self._chunk_default(chunk_content)
                for sub_chunk in sub_chunks:
                    sub_chunk["metadata"]["start_line"] += start
                    sub_chunk["metadata"]["end_line"] += start
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    "content": chunk_content,
                    "metadata": {
                        "start_line": start,
                        "end_line": end,
                        "line_count": end - start + 1
                    }
                })
        
        return chunks


class EmbeddingUtils:
    """
    Utility functions for working with embeddings.
    """
    
    @staticmethod
    def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate the cosine similarity between two embeddings.
        
        Args:
            embedding1 (List[float]): The first embedding.
            embedding2 (List[float]): The second embedding.
            
        Returns:
            float: The cosine similarity between the embeddings.
        """
        # Convert to numpy arrays
        a = np.array(embedding1)
        b = np.array(embedding2)
        
        # Calculate cosine similarity
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    @staticmethod
    def average_embeddings(embeddings: List[List[float]], weights: Optional[List[float]] = None) -> List[float]:
        """
        Calculate the weighted average of multiple embeddings.
        
        Args:
            embeddings (List[List[float]]): The embeddings to average.
            weights (List[float], optional): The weights for each embedding.
            
        Returns:
            List[float]: The weighted average embedding.
        """
        if not embeddings:
            return []
        
        # Convert to numpy arrays
        vectors = np.array(embeddings)
        
        # Apply weights if provided
        if weights:
            if len(weights) != len(embeddings):
                raise ValueError("Number of weights must match number of embeddings")
            
            weights = np.array(weights).reshape(-1, 1)
            result = np.average(vectors, axis=0, weights=weights.flatten())
        else:
            result = np.mean(vectors, axis=0)
        
        # Normalize the result
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        
        return result.tolist()
    
    @staticmethod
    def rank_by_relevance(query_embedding: List[float], candidate_embeddings: List[List[float]]) -> List[int]:
        """
        Rank candidates by relevance to a query.
        
        Args:
            query_embedding (List[float]): The query embedding.
            candidate_embeddings (List[List[float]]): The candidate embeddings.
            
        Returns:
            List[int]: Indices of candidates in descending order of relevance.
        """
        similarities = []
        
        # Calculate similarities
        for candidate in candidate_embeddings:
            similarity = EmbeddingUtils.cosine_similarity(query_embedding, candidate)
            similarities.append(similarity)
        
        # Return indices sorted by similarity (highest first)
        return [i for i, _ in sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)] 