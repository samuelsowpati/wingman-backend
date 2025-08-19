"""
Vector Store Module

This module handles all interactions with Pinecone vector database.
It stores document embeddings and performs similarity searches.

Key Concepts:
- Vector database: Stores numerical representations of text for fast similarity search
- Pinecone: Managed vector database service (no infrastructure to manage)
- Upsert: Insert or update operation in vector databases
- Similarity search: Finding vectors closest to a query vector
- Metadata: Additional information stored with each vector
"""

from pinecone import Pinecone, ServerlessSpec
import os
from typing import List, Dict, Any, Optional
import uuid
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PineconeService:
    """
    Service for managing vector storage and search using Pinecone.
    
    This class handles:
    - Creating and managing Pinecone indexes
    - Storing document embeddings with metadata
    - Performing similarity searches
    - Managing vector database operations
    """
    
    def __init__(self, index_name: str = "air-force-docs", dimension: int = 384):
        """
        Initialize the Pinecone service.
        
        Args:
            index_name: Name of the Pinecone index to use
            dimension: Size of embedding vectors (384 for all-MiniLM-L6-v2)
        """
        # Allow overriding via environment for deployment flexibility
        self.index_name = os.getenv("PINECONE_INDEX", index_name)
        # Ensure dimension matches your embedding model
        self.dimension = int(os.getenv("PINECONE_DIMENSION", str(dimension)))
        
        # Get API key from environment variable
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        logger.info("🔌 Connecting to Pinecone...")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        
        # Create or connect to index
        self._setup_index()
        
        logger.info(f"✅ Connected to Pinecone index: {self.index_name}")
    
    def _setup_index(self):
        """
        Create the Pinecone index if it doesn't exist, or connect to existing one.
        
        Pinecone indexes are like databases - they store vectors and allow search.
        We use serverless spec which automatically scales and is cost-effective.
        """
        try:
            # Check if index already exists
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                logger.info(f"📊 Creating new Pinecone index: {self.index_name}")
                
                # Create serverless index
                # Serverless is cheaper and scales automatically
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",  # Best for text similarity
                    spec=ServerlessSpec(
                        cloud=os.getenv("PINECONE_CLOUD", "aws"),
                        region=os.getenv("PINECONE_REGION", "us-east-1")
                    )
                )
                
                # Wait for index to be ready
                logger.info("⏳ Waiting for index to be ready...")
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
                
                logger.info("✅ Index created and ready!")
            else:
                logger.info(f"📊 Using existing index: {self.index_name}")
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f"📈 Index stats: {stats['total_vector_count']} vectors stored")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Pinecone index: {str(e)}")
            raise
    
    async def upsert_document(
        self, 
        text: str, 
        embedding: List[float], 
        metadata: Dict[str, Any]
    ) -> str:
        """
        Store a document chunk in the vector database.
        
        "Upsert" means "insert or update" - if the ID exists, it updates;
        if not, it inserts a new record.
        
        Args:
            text: The actual text content
            embedding: Vector representation of the text (from embeddings.py)
            metadata: Additional information about the document
            
        Returns:
            Unique ID assigned to this vector
            
        Example:
            vector_id = await store.upsert_document(
                text="SAF/AQ maintains responsibility for acquisition...",
                embedding=[0.1, 0.2, 0.3, ...],  # 384 numbers
                metadata={
                    "source": "afi10-2402.pdf",
                    "doc_type": "AFI",
                    "chunk_id": 0
                }
            )
        """
        try:
            # Generate unique ID for this vector
            vector_id = str(uuid.uuid4())
            
            # Prepare metadata (Pinecone requires specific format)
            # We store the text content in metadata so we can retrieve it later
            vector_metadata = {
                "text": text[:40000],  # Pinecone has metadata size limits
                **metadata  # Include all provided metadata
            }
            
            # Upsert the vector
            self.index.upsert(vectors=[{
                "id": vector_id,
                "values": embedding,
                "metadata": vector_metadata
            }])
            
            logger.debug(f"📝 Stored vector {vector_id[:8]}... from {metadata.get('source', 'unknown')}")
            return vector_id
            
        except Exception as e:
            logger.error(f"❌ Failed to upsert document: {str(e)}")
            raise
    
    async def upsert_documents_batch(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Store multiple documents efficiently in a single operation.
        
        Batch operations are much faster than individual upserts when
        processing many documents at once.
        
        Args:
            documents: List of documents, each containing:
                - text: Document text
                - embedding: Vector representation  
                - metadata: Additional info
                
        Returns:
            List of vector IDs for stored documents
        """
        try:
            if not documents:
                return []
            
            # Prepare vectors for batch upsert
            vectors = []
            vector_ids = []
            
            for doc in documents:
                vector_id = str(uuid.uuid4())
                vector_ids.append(vector_id)
                
                # Prepare metadata
                metadata = doc.get('metadata', {})
                metadata['text'] = doc['text'][:40000]
                
                vectors.append({
                    "id": vector_id,
                    "values": doc['embedding'],
                    "metadata": metadata
                })
            
            # Batch upsert (much faster than individual operations)
            self.index.upsert(vectors=vectors)
            
            logger.info(f"✅ Batch stored {len(vectors)} documents")
            return vector_ids
            
        except Exception as e:
            logger.error(f"❌ Failed to batch upsert documents: {str(e)}")
            raise
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 20,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        Search for documents similar to a query.
        
        This performs vector similarity search using cosine similarity.
        It finds documents whose embeddings are closest to the query embedding.
        
        Args:
            query_embedding: Vector representation of search query
            top_k: Number of results to return
            filter_dict: Optional filters (e.g., {"doc_type": "AFI"})
            
        Returns:
            List of similar documents with scores and metadata
            
        Example:
            results = await store.search_similar(
                query_embedding=[0.1, 0.2, ...],
                top_k=5,
                filter_dict={"doc_type": "AFI"}
            )
            
            # Returns:
            [
                {
                    "id": "uuid-123",
                    "score": 0.89,  # Similarity score (higher = more similar)
                    "text": "SAF/AQ maintains responsibility...",
                    "source": "afi10-2402.pdf",
                    "metadata": {...}
                },
                ...
            ]
        """
        try:
            # Perform the search
            search_kwargs = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True  # Include stored metadata in results
            }
            
            # Add filters if provided
            if filter_dict:
                search_kwargs["filter"] = filter_dict
            
            results = self.index.query(**search_kwargs)
            
            # Format results for easier use
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "id": match.id,
                    "score": float(match.score),  # Similarity score (0-1)
                    "text": match.metadata.get("text", ""),
                    "source": match.metadata.get("source", "unknown"),
                    "page": match.metadata.get("page", 0),
                    "doc_type": match.metadata.get("doc_type", "unknown"),
                    "chunk_id": match.metadata.get("chunk_id", 0),
                    "metadata": match.metadata
                })
            
            logger.info(f"🔍 Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Failed to search similar documents: {str(e)}")
            return []
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents from the vector database.
        
        Args:
            document_ids: List of vector IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not document_ids:
                return True
            
            self.index.delete(ids=document_ids)
            logger.info(f"🗑️ Deleted {len(document_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete documents: {str(e)}")
            return False
    
    async def clear_index(self) -> bool:
        """
        Delete ALL documents from the index.
        
        ⚠️ WARNING: This deletes everything! Use with caution.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.warning("⚠️ Clearing ALL documents from index...")
            self.index.delete(delete_all=True)
            logger.info("🗑️ Index cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clear index: {str(e)}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with index information
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "index_fullness": stats.get('index_fullness', 0),
                "namespaces": stats.get('namespaces', {})
            }
        except Exception as e:
            logger.error(f"❌ Failed to get index stats: {str(e)}")
            return {}
    
    async def search_with_text_filter(
        self, 
        query_embedding: List[float],
        doc_types: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        top_k: int = 20
    ) -> List[Dict]:
        """
        Search with common filter combinations.
        
        This is a convenience method for typical filtering scenarios.
        
        Args:
            query_embedding: Vector to search for
            doc_types: Filter by document types (e.g., ["AFI", "AFMAN"])
            sources: Filter by specific sources
            top_k: Number of results
            
        Returns:
            Filtered search results
        """
        # Build filter dictionary
        filter_dict = {}
        
        if doc_types:
            filter_dict["doc_type"] = {"$in": doc_types}
        
        if sources:
            filter_dict["source"] = {"$in": sources}
        
        return await self.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict if filter_dict else None
        )
    
    async def clear_all_vectors(self) -> bool:
        """
        Delete all vectors from the index.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("🗑️ Clearing all vectors from index...")
            
            # Delete all vectors
            delete_response = self.index.delete(delete_all=True)
            
            logger.info("✅ Delete request sent to Pinecone")
            
            # Wait a moment for deletion to process
            import asyncio
            await asyncio.sleep(2)
            
            # Verify deletion
            stats = await self.get_index_stats()
            remaining_vectors = stats.get('total_vectors', 0)
            
            if remaining_vectors == 0:
                logger.info("✅ All vectors successfully deleted")
                return True
            else:
                logger.warning(f"⚠️ {remaining_vectors} vectors still remain (may take time to process)")
                return True  # Still consider success as deletion is async
                
        except Exception as e:
            logger.error(f"❌ Failed to clear vectors: {str(e)}")
            return False


# Global instance for use throughout the application
vector_store = PineconeService()

"""
Usage Examples:

# Store a document
vector_id = await vector_store.upsert_document(
    text="Assistant Secretary responsibilities...",
    embedding=[0.1, 0.2, 0.3, ...],
    metadata={
        "source": "afi10-2402.pdf",
        "doc_type": "AFI",
        "page": 5
    }
)

# Search for similar documents
results = await vector_store.search_similar(
    query_embedding=[0.4, 0.5, 0.6, ...],
    top_k=5
)

# Search with filters
afi_results = await vector_store.search_with_text_filter(
    query_embedding=[0.4, 0.5, 0.6, ...],
    doc_types=["AFI"],
    top_k=3
)

# Get database stats
stats = await vector_store.get_index_stats()
print(f"Total documents: {stats['total_vectors']}")
"""