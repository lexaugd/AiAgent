"""
Knowledge extraction for the Autonomous Coding Agent.

This module provides functionality to extract knowledge from experiences and conversations.
"""

import time
import json
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import concurrent.futures
from loguru import logger

import sys
import os
sys.path.append("../..")
try:
    # Try direct import first (when run as a module)
    from memory.embeddings import EmbeddingGenerator
    from memory.long_term import get_long_term_memory
except ImportError:
    # Try relative import (when run from the autonomous_agent directory)
    from ..memory.embeddings import EmbeddingGenerator
    from ..memory.long_term import get_long_term_memory

from .types import KnowledgeItem, KnowledgeType, Experience
from .experience import get_experience_tracker

# Singleton instance
_knowledge_extractor = None

class KnowledgeExtractor:
    """Class to extract knowledge from experiences and conversations."""
    
    def __init__(
        self, 
        extraction_threshold: float = 0.6,
        embedding_batch_size: int = 5
    ):
        """
        Initialize the knowledge extractor.
        
        Args:
            extraction_threshold (float, optional): Confidence threshold for extraction
            embedding_batch_size (int, optional): Batch size for embedding generation
        """
        self.extraction_threshold = extraction_threshold
        self.embedding_batch_size = embedding_batch_size
        
        # Get instances of related components
        self.experience_tracker = get_experience_tracker()
        self.long_term_memory = get_long_term_memory()
        self.embedding_generator = EmbeddingGenerator()
        
        # Patterns for code extraction
        self.code_patterns = {
            # Function definition
            "function_def": re.compile(r"def\s+(\w+)\s*\([^)]*\)\s*:"),
            # Class definition
            "class_def": re.compile(r"class\s+(\w+)"),
            # Code block
            "code_block": re.compile(r"```(?:python|javascript|typescript|java|cpp|ruby|go|rust|php)?\n(.*?)```", re.DOTALL),
            # Error patterns
            "error_pattern": re.compile(r"(?:error|exception|traceback).*?:.*?(?:\n\s+.*?)+", re.IGNORECASE | re.DOTALL),
        }
        
        logger.info(f"Initialized KnowledgeExtractor with threshold: {extraction_threshold}")
        
    def extract_from_experience(self, experience: Experience) -> List[KnowledgeItem]:
        """
        Extract knowledge from a single experience.
        
        Args:
            experience (Experience): The experience to extract knowledge from
            
        Returns:
            List[KnowledgeItem]: Extracted knowledge items
        """
        knowledge_items = []
        
        # Extract based on experience type
        if experience.experience_type.value == "code_generation":
            code_items = self._extract_code_snippets(experience)
            knowledge_items.extend(code_items)
            
        elif experience.experience_type.value == "error_resolution":
            error_items = self._extract_error_solutions(experience)
            knowledge_items.extend(error_items)
            
        elif experience.experience_type.value == "code_explanation":
            concept_items = self._extract_concepts(experience)
            knowledge_items.extend(concept_items)
            
        # Extract generic knowledge applicable to all experience types
        fact_items = self._extract_facts(experience)
        knowledge_items.extend(fact_items)
        
        # Filter out low-confidence items
        knowledge_items = [item for item in knowledge_items if item.confidence >= self.extraction_threshold]
        
        # Generate embeddings for the knowledge items
        self._generate_embeddings(knowledge_items)
        
        # Store in long-term memory
        self._store_knowledge_items(knowledge_items)
        
        return knowledge_items
        
    def batch_extract_from_experiences(
        self, 
        experience_ids: Optional[List[str]] = None,
        experience_types: Optional[List[str]] = None,
        limit: int = 20
    ) -> Dict[str, List[KnowledgeItem]]:
        """
        Extract knowledge from multiple experiences.
        
        Args:
            experience_ids (List[str], optional): IDs of specific experiences to process
            experience_types (List[str], optional): Types of experiences to process
            limit (int, optional): Maximum number of experiences to process
            
        Returns:
            Dict[str, List[KnowledgeItem]]: Mapping of experience IDs to extracted knowledge
        """
        # Get experiences to process
        if experience_ids:
            experiences = [self.experience_tracker.get_experience(exp_id) for exp_id in experience_ids]
            experiences = [exp for exp in experiences if exp is not None]
        else:
            experiences = self.experience_tracker.list_experiences(
                experience_type=experience_types[0] if experience_types else None,
                limit=limit
            )
            
            if experience_types and len(experience_types) > 1:
                # Filter by multiple experience types
                experiences = [
                    exp for exp in experiences 
                    if exp.experience_type.value in experience_types
                ]
        
        # Process experiences in parallel
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_exp = {
                executor.submit(self.extract_from_experience, exp): exp 
                for exp in experiences
            }
            
            for future in concurrent.futures.as_completed(future_to_exp):
                exp = future_to_exp[future]
                try:
                    knowledge_items = future.result()
                    results[exp.experience_id] = knowledge_items
                except Exception as e:
                    logger.error(f"Error extracting knowledge from experience {exp.experience_id}: {e}")
                    results[exp.experience_id] = []
        
        return results
        
    def extract_knowledge_from_text(
        self, 
        text: str,
        source: str,
        knowledge_types: Optional[List[KnowledgeType]] = None
    ) -> List[KnowledgeItem]:
        """
        Extract knowledge directly from text.
        
        Args:
            text (str): The text to extract knowledge from
            source (str): The source of the text
            knowledge_types (List[KnowledgeType], optional): Types of knowledge to extract
            
        Returns:
            List[KnowledgeItem]: Extracted knowledge items
        """
        knowledge_items = []
        
        # Set default knowledge types if not specified
        if not knowledge_types:
            knowledge_types = [
                KnowledgeType.CODE_SNIPPET,
                KnowledgeType.FACT,
                KnowledgeType.CONCEPT
            ]
        
        # Extract code snippets
        if KnowledgeType.CODE_SNIPPET in knowledge_types:
            code_blocks = self._extract_code_blocks_from_text(text)
            for code in code_blocks:
                item = KnowledgeItem(
                    content=code.strip(),
                    knowledge_type=KnowledgeType.CODE_SNIPPET,
                    source=source,
                    confidence=0.9,  # High confidence for explicitly delimited code
                    metadata={"extraction_method": "code_block_regex"}
                )
                knowledge_items.append(item)
                
        # Extract facts
        if KnowledgeType.FACT in knowledge_types:
            facts = self._extract_facts_from_text(text)
            for fact, confidence in facts:
                item = KnowledgeItem(
                    content=fact.strip(),
                    knowledge_type=KnowledgeType.FACT,
                    source=source,
                    confidence=confidence,
                    metadata={"extraction_method": "fact_extraction"}
                )
                knowledge_items.append(item)
                
        # Extract concepts
        if KnowledgeType.CONCEPT in knowledge_types:
            concepts = self._extract_concepts_from_text(text)
            for concept, confidence in concepts:
                item = KnowledgeItem(
                    content=concept.strip(),
                    knowledge_type=KnowledgeType.CONCEPT,
                    source=source,
                    confidence=confidence,
                    metadata={"extraction_method": "concept_extraction"}
                )
                knowledge_items.append(item)
                
        # Extract error solutions
        if KnowledgeType.ERROR_SOLUTION in knowledge_types:
            error_solutions = self._extract_error_patterns_from_text(text)
            for solution, confidence in error_solutions:
                item = KnowledgeItem(
                    content=solution.strip(),
                    knowledge_type=KnowledgeType.ERROR_SOLUTION,
                    source=source,
                    confidence=confidence,
                    metadata={"extraction_method": "error_pattern_extraction"}
                )
                knowledge_items.append(item)
        
        # Filter out low-confidence items
        knowledge_items = [item for item in knowledge_items if item.confidence >= self.extraction_threshold]
        
        # Generate embeddings for the knowledge items
        self._generate_embeddings(knowledge_items)
        
        # Store in long-term memory
        self._store_knowledge_items(knowledge_items)
        
        return knowledge_items
        
    def _extract_code_snippets(self, experience: Experience) -> List[KnowledgeItem]:
        """Extract code snippets from an experience."""
        code_items = []
        
        # Extract code blocks from the response
        code_blocks = self._extract_code_blocks_from_text(experience.response)
        
        for code_block in code_blocks:
            if len(code_block.strip()) > 10:  # Ignore very short snippets
                item = KnowledgeItem(
                    content=code_block.strip(),
                    knowledge_type=KnowledgeType.CODE_SNIPPET,
                    source=experience.experience_id,
                    confidence=0.9,  # High confidence for code blocks
                    metadata={
                        "experience_type": experience.experience_type.value,
                        "extraction_method": "code_block_pattern"
                    }
                )
                code_items.append(item)
                
        # Look for code patterns outside code blocks
        code_patterns = self._find_code_patterns(experience.response)
        
        for pattern_type, code_pattern in code_patterns:
            # Avoid duplicating code already found in code blocks
            if not any(code_pattern in block for block in code_blocks):
                item = KnowledgeItem(
                    content=code_pattern.strip(),
                    knowledge_type=KnowledgeType.CODE_PATTERN,
                    source=experience.experience_id,
                    confidence=0.7,  # Lower confidence for code patterns
                    metadata={
                        "pattern_type": pattern_type,
                        "experience_type": experience.experience_type.value,
                        "extraction_method": "code_pattern_match"
                    }
                )
                code_items.append(item)
                
        return code_items
        
    def _extract_error_solutions(self, experience: Experience) -> List[KnowledgeItem]:
        """Extract error solutions from an experience."""
        error_items = []
        
        # Look for error patterns in the query (user's question)
        error_patterns = self.code_patterns["error_pattern"].findall(experience.query)
        
        # Extract the solution from the response for each error pattern
        for error_pattern in error_patterns:
            # Simplified: Assume the entire response is the solution
            solution = experience.response
            
            item = KnowledgeItem(
                content=f"Error: {error_pattern.strip()}\nSolution: {solution.strip()}",
                knowledge_type=KnowledgeType.ERROR_SOLUTION,
                source=experience.experience_id,
                confidence=0.8,
                metadata={
                    "error_text": error_pattern.strip(),
                    "experience_type": experience.experience_type.value,
                    "extraction_method": "error_pattern_match"
                }
            )
            error_items.append(item)
            
        return error_items
        
    def _extract_concepts(self, experience: Experience) -> List[KnowledgeItem]:
        """Extract concepts from an explanation experience."""
        concept_items = []
        
        # Very simple concept extraction - treat paragraphs as potential concepts
        paragraphs = experience.response.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) > 50 and len(paragraph) < 500:  # Reasonable length for a concept
                # Simple heuristic: concepts likely contain explanatory language
                explanatory_markers = ['is', 'means', 'refers to', 'represents', 'consists of']
                explanation_score = sum(1 for marker in explanatory_markers if marker in paragraph.lower())
                confidence = min(0.5 + (explanation_score * 0.1), 0.9)  # Scale confidence
                
                if confidence >= self.extraction_threshold:
                    item = KnowledgeItem(
                        content=paragraph,
                        knowledge_type=KnowledgeType.CONCEPT,
                        source=experience.experience_id,
                        confidence=confidence,
                        metadata={
                            "explanation_score": explanation_score,
                            "experience_type": experience.experience_type.value,
                            "extraction_method": "paragraph_analysis"
                        }
                    )
                    concept_items.append(item)
                    
        return concept_items
        
    def _extract_facts(self, experience: Experience) -> List[KnowledgeItem]:
        """Extract factual statements from an experience."""
        fact_items = []
        
        # Simple fact extraction - look for sentences that sound factual
        sentences = re.split(r'[.!?]', experience.response)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:  # Reasonable length for a fact
                # Simple heuristic: facts often use certain phrases or patterns
                fact_markers = ['always', 'never', 'all', 'none', 'must', 'should', 'is a', 'are the']
                factual_score = sum(1 for marker in fact_markers if f" {marker} " in f" {sentence.lower()} ")
                confidence = min(0.4 + (factual_score * 0.1), 0.8)  # Scale confidence
                
                if confidence >= self.extraction_threshold:
                    item = KnowledgeItem(
                        content=sentence,
                        knowledge_type=KnowledgeType.FACT,
                        source=experience.experience_id,
                        confidence=confidence,
                        metadata={
                            "factual_score": factual_score,
                            "experience_type": experience.experience_type.value,
                            "extraction_method": "sentence_analysis"
                        }
                    )
                    fact_items.append(item)
                    
        return fact_items
        
    def _extract_code_blocks_from_text(self, text: str) -> List[str]:
        """Extract code blocks from text using regex."""
        # Extract code within markdown code blocks
        code_blocks = []
        
        # Find code blocks with language specification
        matches = self.code_patterns["code_block"].findall(text)
        code_blocks.extend(matches)
        
        # Also look for indented code blocks (4 spaces or tabs)
        lines = text.split("\n")
        current_block = []
        in_block = False
        
        for line in lines:
            if line.startswith("    ") or line.startswith("\t"):
                current_block.append(line.lstrip())
                in_block = True
            elif line.strip() == "" and in_block:
                # Empty line within a block, keep the block going
                current_block.append("")
            elif in_block:
                # End of block
                if len(current_block) > 2:  # Avoid single-line indents
                    code_blocks.append("\n".join(current_block))
                current_block = []
                in_block = False
                
        # Add the last block if we were in one
        if in_block and len(current_block) > 2:
            code_blocks.append("\n".join(current_block))
            
        return code_blocks
        
    def _find_code_patterns(self, text: str) -> List[Tuple[str, str]]:
        """Find code patterns in text."""
        patterns = []
        
        # Find function definitions
        for match in self.code_patterns["function_def"].finditer(text):
            # Extract the whole line plus the next few lines for context
            start = match.start()
            end = text.find("\n\n", start)
            if end == -1:
                end = len(text)
            pattern = text[start:end].strip()
            patterns.append(("function_definition", pattern))
            
        # Find class definitions
        for match in self.code_patterns["class_def"].finditer(text):
            # Extract the whole line plus the next few lines for context
            start = match.start()
            end = text.find("\n\n", start)
            if end == -1:
                end = len(text)
            pattern = text[start:end].strip()
            patterns.append(("class_definition", pattern))
            
        return patterns
        
    def _extract_facts_from_text(self, text: str) -> List[Tuple[str, float]]:
        """Extract factual statements from text."""
        facts = []
        
        # Split text into sentences
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:  # Reasonable length for a fact
                # Simple heuristic: facts often use certain phrases or patterns
                fact_markers = ['always', 'never', 'all', 'none', 'must', 'should', 'is a', 'are the']
                factual_score = sum(1 for marker in fact_markers if f" {marker} " in f" {sentence.lower()} ")
                confidence = min(0.4 + (factual_score * 0.1), 0.8)  # Scale confidence
                
                if confidence >= self.extraction_threshold:
                    facts.append((sentence, confidence))
                    
        return facts
        
    def _extract_concepts_from_text(self, text: str) -> List[Tuple[str, float]]:
        """Extract concepts from text."""
        concepts = []
        
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) > 50 and len(paragraph) < 500:  # Reasonable length for a concept
                # Simple heuristic: concepts likely contain explanatory language
                explanatory_markers = ['is', 'means', 'refers to', 'represents', 'consists of']
                explanation_score = sum(1 for marker in explanatory_markers if marker in paragraph.lower())
                confidence = min(0.5 + (explanation_score * 0.1), 0.9)  # Scale confidence
                
                if confidence >= self.extraction_threshold:
                    concepts.append((paragraph, confidence))
                    
        return concepts
        
    def _extract_error_patterns_from_text(self, text: str) -> List[Tuple[str, float]]:
        """Extract error patterns and solutions from text."""
        error_solutions = []
        
        # Look for error patterns
        error_patterns = self.code_patterns["error_pattern"].findall(text)
        
        for error_pattern in error_patterns:
            # Look for solutions after the error pattern
            start_idx = text.find(error_pattern) + len(error_pattern)
            end_idx = text.find("\n\n", start_idx)
            if end_idx == -1:
                end_idx = len(text)
                
            solution = text[start_idx:end_idx].strip()
            
            # Skip if solution is too short
            if len(solution) < 20:
                continue
                
            error_solutions.append((f"Error: {error_pattern}\nSolution: {solution}", 0.8))
            
        return error_solutions
        
    def _generate_embeddings(self, knowledge_items: List[KnowledgeItem]):
        """Generate embeddings for knowledge items in batches."""
        items_without_embedding = [item for item in knowledge_items if item.embedding is None]
        
        if not items_without_embedding:
            return
            
        # Process in batches
        for i in range(0, len(items_without_embedding), self.embedding_batch_size):
            batch = items_without_embedding[i:i+self.embedding_batch_size]
            texts = [item.content for item in batch]
            
            try:
                embeddings = self.embedding_generator.generate_batch(texts)
                
                # Assign embeddings to items
                for j, item in enumerate(batch):
                    item.embedding = embeddings[j]
            except Exception as e:
                logger.error(f"Error generating embeddings for knowledge batch: {e}")
                
    def _store_knowledge_items(self, knowledge_items: List[KnowledgeItem]):
        """Store knowledge items in long-term memory."""
        for item in knowledge_items:
            if not item.embedding:
                continue  # Skip items without embeddings
                
            try:
                # Store in long-term memory
                self.long_term_memory.add_item(
                    item=item.content,
                    item_type=item.knowledge_type.value,
                    metadata={
                        **item.metadata,
                        "confidence": item.confidence,
                        "knowledge_id": item.knowledge_id,
                        "timestamp": item.timestamp,
                        "source": item.source
                    }
                )
                logger.debug(f"Stored knowledge item {item.knowledge_id} in long-term memory")
            except Exception as e:
                logger.error(f"Error storing knowledge item {item.knowledge_id} in long-term memory: {e}")


def get_knowledge_extractor(
    extraction_threshold: float = 0.6,
    embedding_batch_size: int = 5
) -> KnowledgeExtractor:
    """
    Get or create the singleton KnowledgeExtractor instance.
    
    Args:
        extraction_threshold (float, optional): Confidence threshold for extraction
        embedding_batch_size (int, optional): Batch size for embedding generation
        
    Returns:
        KnowledgeExtractor: The singleton KnowledgeExtractor instance
    """
    global _knowledge_extractor
    if _knowledge_extractor is None:
        _knowledge_extractor = KnowledgeExtractor(extraction_threshold, embedding_batch_size)
    return _knowledge_extractor 