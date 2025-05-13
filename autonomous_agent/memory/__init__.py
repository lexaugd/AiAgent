"""
Memory-related modules for the Autonomous Coding Agent.
"""

from .short_term import ShortTermMemory, Message, get_memory as get_short_term_memory
from .long_term import LongTermMemory, MemoryItem, get_long_term_memory
from .manager import MemoryManager, get_memory_manager
from .embeddings import EmbeddingGenerator, CodeChunker, EmbeddingUtils
from .types import (
    MemoryType, MemoryPriority, MemoryAccess, 
    MemoryAssociation, MemoryMetadata, ExtendedMemoryItem
)
from .forgetting import (
    ForgettingCurve, EbbinghausForgettingCurve, PowerLawForgettingCurve,
    ForgettingParams, MemoryForgetting, MemoryConsolidation
)
from .retrieval import (
    RetrievalResult, QueryExpansion, ContextAwareRetrieval, MultiSourceRetrieval
)

# Export the memory components for easy access
__all__ = [
    # Short-term memory
    'ShortTermMemory',
    'Message',
    'get_short_term_memory',
    
    # Long-term memory
    'LongTermMemory',
    'MemoryItem',
    'get_long_term_memory',
    
    # Memory manager
    'MemoryManager',
    'get_memory_manager',
    
    # Embeddings
    'EmbeddingGenerator',
    'CodeChunker',
    'EmbeddingUtils',
    
    # Memory types
    'MemoryType',
    'MemoryPriority',
    'MemoryAccess',
    'MemoryAssociation',
    'MemoryMetadata',
    'ExtendedMemoryItem',
    
    # Forgetting mechanisms
    'ForgettingCurve',
    'EbbinghausForgettingCurve',
    'PowerLawForgettingCurve',
    'ForgettingParams',
    'MemoryForgetting',
    'MemoryConsolidation',
    
    # Retrieval mechanisms
    'RetrievalResult',
    'QueryExpansion',
    'ContextAwareRetrieval',
    'MultiSourceRetrieval',
] 