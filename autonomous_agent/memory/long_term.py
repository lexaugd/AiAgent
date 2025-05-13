"""
Long-term memory implementation for the Autonomous Coding Agent.

This module provides a long-term memory implementation using ChromaDB for vector storage.
"""

import os
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from loguru import logger

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError  # Import the specific error
import numpy as np
from sentence_transformers import SentenceTransformer

from config import MEMORY_CONFIG, VECTOR_DB_DIR


class MemoryItem:
    """Class to represent an item in the long-term memory."""
    
    def __init__(
        self,
        content: str,
        item_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        item_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        embedding: Optional[List[float]] = None
    ):
        """
        Initialize a memory item.
        
        Args:
            content (str): The content of the memory item.
            item_type (str): The type of the memory item (code, concept, procedure, etc.).
            metadata (Dict[str, Any], optional): Additional metadata for the item.
            item_id (str, optional): Unique identifier for the item.
            timestamp (float, optional): The timestamp of when the item was created.
            embedding (List[float], optional): Pre-computed embedding vector.
        """
        self.content = content
        self.item_type = item_type
        self.metadata = metadata or {}
        self.item_id = item_id or f"{item_type}_{int(time.time())}_{id(self)}"
        self.timestamp = timestamp or time.time()
        self.embedding = embedding
        
        # Add timestamp to metadata
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = self.timestamp
        
        # Add item type to metadata
        if "item_type" not in self.metadata:
            self.metadata["item_type"] = self.item_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the memory item to a dictionary."""
        return {
            "item_id": self.item_id,
            "content": self.content,
            "item_type": self.item_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create a memory item from a dictionary."""
        return cls(
            content=data["content"],
            item_type=data["item_type"],
            metadata=data.get("metadata", {}),
            item_id=data.get("item_id"),
            timestamp=data.get("timestamp"),
            embedding=data.get("embedding")
        )


class LongTermMemory:
    """
    Long-term memory implementation using ChromaDB for vector storage.
    """
    
    def __init__(
        self,
        collection_name: str = "code_knowledge",
        embedding_model: Optional[str] = None,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the long-term memory.
        
        Args:
            collection_name (str): The name of the vector collection.
            embedding_model (str, optional): The name of the embedding model to use.
            persist_directory (str, optional): Directory for ChromaDB persistence.
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model or "all-MiniLM-L6-v2"
        self.persist_directory = persist_directory or VECTOR_DB_DIR
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Cache for frequently accessed items
        self.cache = {}
        self.cache_size = 100  # Maximum number of items to cache
        
        logger.info(f"Initialized LongTermMemory with collection: {collection_name}")
        logger.info(f"Using embedding model: {self.embedding_model_name}")
        logger.info(f"Persistence directory: {self.persist_directory}")
    
    def _get_or_create_collection(self):
        """Get an existing collection or create a new one."""
        try:
            return self.client.get_collection(name=self.collection_name)
        except NotFoundError:  # Explicitly catch the specific NotFoundError
            logger.info(f"Collection {self.collection_name} not found, creating it.")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"timestamp": time.time()}
            )
    
    def add_item(self, item: Union[MemoryItem, str], item_type: Optional[str] = None, 
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add an item to the long-term memory.
        
        Args:
            item (Union[MemoryItem, str]): The memory item or content string to add.
            item_type (str, optional): Required if item is a string, the type of memory.
            metadata (Dict[str, Any], optional): Additional metadata if item is a string.
            
        Returns:
            str: The ID of the added item.
        """
        # Convert string to MemoryItem if necessary
        if isinstance(item, str):
            if not item_type:
                raise ValueError("item_type must be provided when adding a string item")
            item = MemoryItem(content=item, item_type=item_type, metadata=metadata)
        
        # Generate embedding if not already present
        if item.embedding is None:
            item.embedding = self._generate_embedding(item.content)
        
        # Process metadata to ensure all values are primitive types
        processed_metadata = {}
        for key, value in item.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                processed_metadata[key] = value
            else:
                # Convert complex types to JSON strings
                processed_metadata[key] = json.dumps(value)
        
        # Add to ChromaDB collection
        self.collection.add(
            ids=[item.item_id],
            embeddings=[item.embedding],
            metadatas=[processed_metadata],
            documents=[item.content]
        )
        
        # Add to cache
        self._add_to_cache(item)
        
        logger.debug(f"Added item to long-term memory: {item.item_id} ({item.item_type})")
        return item.item_id
    
    def query(self, query_text: str, item_type: Optional[str] = None, 
              n_results: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[MemoryItem]:
        """
        Query the long-term memory for relevant items.
        
        Args:
            query_text (str): The query text.
            item_type (str, optional): Filter by item type.
            n_results (int): Maximum number of results to return.
            filters (Dict[str, Any], optional): Additional filters for the query.
            
        Returns:
            List[MemoryItem]: The retrieved memory items.
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query_text)
        
        # Prepare filter
        where_filter = filters or {}
        if item_type and not filters:
            # Convert item_type to ChromaDB filter format
            where_filter = {"item_type": {"$eq": item_type}}
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter if where_filter else None,
            include=["metadatas", "documents", "embeddings"]
        )
        
        # Convert results to MemoryItems
        memory_items = []
        for i in range(len(results["ids"][0])):
            item_id = results["ids"][0][i]
            content = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            
            # Process metadata to convert JSON strings back to complex types
            processed_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, str) and value.startswith(('[', '{', '"')) and (value.endswith(']') or value.endswith('}') or value.endswith('"')):
                    try:
                        processed_metadata[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        processed_metadata[key] = value
                else:
                    processed_metadata[key] = value
            
            # Get the embedding if available
            embedding = None
            if "embeddings" in results and results["embeddings"] is not None:
                try:
                    embedding = results["embeddings"][0][i]
                except (IndexError, TypeError):
                    # Handle the case when embeddings are not available
                    pass
            
            item = MemoryItem(
                content=content,
                item_type=processed_metadata.get("item_type", "unknown"),
                metadata=processed_metadata,
                item_id=item_id,
                timestamp=processed_metadata.get("timestamp"),
                embedding=embedding
            )
            memory_items.append(item)
        
        logger.debug(f"Query returned {len(memory_items)} results")
        return memory_items
    
    def get_item(self, item_id: str) -> Optional[MemoryItem]:
        """
        Get a specific item by ID.
        
        Args:
            item_id (str): The ID of the item to retrieve.
            
        Returns:
            Optional[MemoryItem]: The memory item if found, None otherwise.
        """
        # Check cache first
        if item_id in self.cache:
            logger.debug(f"Cache hit for item: {item_id}")
            return self.cache[item_id]
        
        # Query ChromaDB
        try:
            results = self.collection.get(
                ids=[item_id],
                include=["embeddings", "metadatas", "documents"]
            )
            
            if not results["ids"]:
                return None
            
            content = results["documents"][0]
            metadata = results["metadatas"][0]
            embedding = results["embeddings"][0] if "embeddings" in results else None
            
            # Process metadata to convert JSON strings back to complex types
            processed_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, str) and value.startswith(('[', '{', '"')) and (value.endswith(']') or value.endswith('}') or value.endswith('"')):
                    try:
                        processed_metadata[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        processed_metadata[key] = value
                else:
                    processed_metadata[key] = value
            
            item = MemoryItem(
                content=content,
                item_type=processed_metadata.get("item_type", "unknown"),
                metadata=processed_metadata,
                item_id=item_id,
                timestamp=processed_metadata.get("timestamp"),
                embedding=embedding
            )
            
            # Add to cache
            self._add_to_cache(item)
            
            return item
        except Exception as e:
            logger.error(f"Error retrieving item {item_id}: {e}")
            return None
    
    def update_item(self, item: MemoryItem) -> bool:
        """
        Update an existing item in the memory.
        
        Args:
            item (MemoryItem): The memory item to update.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Generate new embedding if needed
            if item.embedding is None:
                item.embedding = self._generate_embedding(item.content)
            
            # Process metadata to ensure all values are primitive types
            processed_metadata = {}
            for key, value in item.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    processed_metadata[key] = value
                else:
                    # Convert complex types to JSON strings
                    processed_metadata[key] = json.dumps(value)
            
            # Update in ChromaDB
            self.collection.update(
                ids=[item.item_id],
                embeddings=[item.embedding],
                metadatas=[processed_metadata],
                documents=[item.content]
            )
            
            # Update cache
            self._add_to_cache(item)
            
            logger.debug(f"Updated item in long-term memory: {item.item_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating item {item.item_id}: {e}")
            return False
    
    def delete_item(self, item_id: str) -> bool:
        """
        Delete an item from the memory.
        
        Args:
            item_id (str): The ID of the item to delete.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Delete from ChromaDB
            self.collection.delete(ids=[item_id])
            
            # Remove from cache
            if item_id in self.cache:
                del self.cache[item_id]
            
            logger.debug(f"Deleted item from long-term memory: {item_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting item {item_id}: {e}")
            return False
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], 
                        n_results: int = 10) -> List[MemoryItem]:
        """
        Search for items by metadata.
        
        Args:
            metadata_filter (Dict[str, Any]): Metadata to filter by.
            n_results (int): Maximum number of results to return.
            
        Returns:
            List[MemoryItem]: The matching memory items.
        """
        try:
            # Convert metadata filter to ChromaDB where format
            chroma_filter = {}
            if metadata_filter:
                # If there are multiple conditions, use $and operator
                if len(metadata_filter) > 1:
                    chroma_filter["$and"] = []
                    for key, value in metadata_filter.items():
                        if isinstance(value, (str, int, float, bool)):
                            chroma_filter["$and"].append({key: {"$eq": value}})
                        else:
                            # Convert complex types to JSON strings
                            chroma_filter["$and"].append({key: {"$eq": json.dumps(value)}})
                else:
                    # Single condition
                    key, value = next(iter(metadata_filter.items()))
                    if isinstance(value, (str, int, float, bool)):
                        chroma_filter[key] = {"$eq": value}
                    else:
                        chroma_filter[key] = {"$eq": json.dumps(value)}
            
            results = self.collection.get(
                where=chroma_filter if chroma_filter else None,
                limit=n_results
            )
            
            memory_items = []
            for i in range(len(results["ids"])):
                item_id = results["ids"][i]
                content = results["documents"][i]
                metadata = results["metadatas"][i]
                
                # Process metadata to convert JSON strings back to complex types
                processed_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, str) and value.startswith(('[', '{', '"')) and (value.endswith(']') or value.endswith('}') or value.endswith('"')):
                        try:
                            processed_metadata[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            processed_metadata[key] = value
                    else:
                        processed_metadata[key] = value
                
                item = MemoryItem(
                    content=content,
                    item_type=processed_metadata.get("item_type", "unknown"),
                    metadata=processed_metadata,
                    item_id=item_id,
                    timestamp=processed_metadata.get("timestamp")
                )
                memory_items.append(item)
            
            return memory_items
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Args:
            text (str): The text to embed.
            
        Returns:
            List[float]: The embedding vector.
        """
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(text)
            
            # Convert to list and return
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return a zero vector as fallback
            dimension = MEMORY_CONFIG["long_term"]["embedding_dimension"]
            return [0.0] * dimension
    
    def _add_to_cache(self, item: MemoryItem) -> None:
        """
        Add an item to the cache.
        
        Args:
            item (MemoryItem): The item to add to the cache.
        """
        # Add to cache
        self.cache[item.item_id] = item
        
        # If cache is too large, remove oldest items
        if len(self.cache) > self.cache_size:
            # Sort items by timestamp and remove oldest
            sorted_items = sorted(
                self.cache.items(), 
                key=lambda x: x[1].timestamp
            )
            
            # Remove oldest items
            for i in range(len(self.cache) - self.cache_size):
                del self.cache[sorted_items[i][0]]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dict[str, Any]: Statistics about the collection.
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "item_count": count,
                "embedding_model": self.embedding_model_name,
                "cache_size": len(self.cache),
                "max_cache_size": self.cache_size
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "collection_name": self.collection_name,
                "error": str(e)
            }


def get_long_term_memory(collection_name: str = "code_knowledge") -> LongTermMemory:
    """
    Get a long-term memory instance.
    
    Args:
        collection_name (str): The name of the collection to use.
        
    Returns:
        LongTermMemory: The long-term memory instance.
    """
    return LongTermMemory(collection_name=collection_name) 