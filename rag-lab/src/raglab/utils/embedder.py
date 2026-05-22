"""
Embedding manager using sentence-transformers with singleton pattern.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class Embedder:
    """
    Singleton wrapper for SentenceTransformer models.
    Caches one model instance per model name to avoid redundant loading.
    """
    
    _instances: Dict[str, 'Embedder'] = {}
    
    def __new__(cls, model_name: str):
        """Singleton pattern: return existing instance if model already loaded."""
        if model_name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[model_name] = instance
            instance._initialized = False
        return cls._instances[model_name]
    
    def __init__(self, model_name: str):
        """
        Initialize Embedder with specified model.
        
        Args:
            model_name: Name of sentence-transformers model (e.g., 'all-MiniLM-L6-v2')
        """
        # Only initialize once per model name
        if self._initialized:
            return
        
        self.model_name = model_name
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise ImportError("sentence-transformers is required for Embedder")
        
        self._initialized = True
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        if not texts:
            return []
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        # Convert numpy arrays to lists for JSON serialization
        return [emb.tolist() for emb in embeddings]
    
    def embed_one(self, text: str) -> List[float]:
        """
        Embed a single text string.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()

    def model_dim(self) -> int:
        """
        Get the embedding dimension for the loaded model.
        
        Returns:
            Integer dimension of the embedding vectors
        """
        return self.model.get_embedding_dimension()
