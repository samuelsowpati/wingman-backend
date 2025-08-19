"""
Embedding Service Module

This module handles text-to-vector conversion using Sentence Transformers.
It converts text into numerical representations (embeddings) that capture semantic meaning.

Key Concepts:
- Embeddings: Numerical vectors that represent text meaning
- Sentence Transformers: Pre-trained models that create good embeddings
- Async processing: Non-blocking operations for better performance
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import asyncio
import functools
import logging

# Set up logging to track what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for converting text into vector embeddings.
    
    This class wraps the Sentence Transformers model and provides
    async methods for creating embeddings from text.
    """
    
    def __init__(self, model_name: str = "multi-qa-MiniLM-L6-cos-v1"):
        """
        Initialize the embedding service with a specific model.
        
        Args:
            model_name: The Sentence Transformers model to use
                       'multi-qa-MiniLM-L6-cos-v1' is a good balance of speed and quality
                       - Creates 384-dimensional vectors
                       - Fast inference
                       - Good for general text understanding
        """
        logger.info(f"Loading embedding model: {model_name}")
        logger.info("This may take a few minutes on first run...")
        
        # Load the pre-trained model
        # This downloads the model files if not already cached
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"✅ Embedding model loaded successfully!")
        logger.info(f"Model: {model_name}")
        logger.info(f"Embedding dimension: {self.dimension}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Convert a single text string into an embedding vector.
        
        Args:
            text: The text to convert (e.g., "What does SAF/AQ do?")
            
        Returns:
            A list of floating point numbers representing the text meaning
            
        Example:
            embedding = await service.get_embedding("Air Force responsibilities")
            # Returns something like: [0.1, -0.3, 0.7, 0.2, ...]
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to get_embedding")
            return [0.0] * self.dimension
        
        # Run the embedding creation in a thread pool to avoid blocking
        # the main async event loop (this keeps the API responsive)
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,  # Use default thread pool
            functools.partial(self.model.encode, [text.strip()])
        )
        
        # Convert numpy array to regular Python list
        return embedding[0].tolist()
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts into embeddings efficiently.
        
        This is more efficient than calling get_embedding multiple times
        because the model can process them together.
        
        Args:
            texts: List of text strings to convert
            
        Returns:
            List of embedding vectors, one for each input text
            
        Example:
            texts = ["SAF/AQ duties", "ACC responsibilities", "Air Force roles"]
            embeddings = await service.get_embeddings_batch(texts)
            # Returns: [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]
        """
        if not texts:
            return []
        
        # Filter out empty texts and keep track of original positions
        clean_texts = []
        text_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                clean_texts.append(text.strip())
                text_indices.append(i)
        
        if not clean_texts:
            logger.warning("No valid texts provided to get_embeddings_batch")
            return [[0.0] * self.dimension] * len(texts)
        
        # Process all texts at once (much faster than one by one)
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            functools.partial(self.model.encode, clean_texts)
        )
        
        # Create result list with embeddings in correct positions
        result = []
        clean_idx = 0
        
        for i in range(len(texts)):
            if i in text_indices:
                result.append(embeddings[clean_idx].tolist())
                clean_idx += 1
            else:
                # Return zero vector for empty texts
                result.append([0.0] * self.dimension)
        
        return result
    
    def get_model_info(self) -> dict:
        """
        Get information about the current embedding model.
        
        Returns:
            Dictionary with model details
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "description": "Sentence Transformer model for creating text embeddings"
        }


# Global instance that will be used throughout the application
# This is created once when the module is imported and reused
embedding_service = EmbeddingService()

# Alternative models you could use instead:
# embedding_service = EmbeddingService("all-mpnet-base-v2")  # Higher quality, slower
# embedding_service = EmbeddingService("all-distilroberta-v1")  # Different architecture