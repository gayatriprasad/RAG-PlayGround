"""
ChromaDB-based vector index implementation.
"""

import hashlib
import json
import logging
import os
from typing import List, Optional, Dict, Any

from raglab.config import IndexCfg, EmbedCfg
from raglab.types import Chunk, RetrievedChunk
from raglab.index.base import BaseIndex
from raglab.utils.embedder import Embedder

logger = logging.getLogger(__name__)


class ChromaIndex(BaseIndex):
    """
    Vector index using ChromaDB with persistent local storage.
    Uses sentence-transformers for embeddings.
    """
    
    def __init__(self, cfg: IndexCfg, embed_cfg: EmbedCfg):
        """
        Initialize ChromaIndex.
        
        Args:
            cfg: IndexCfg with persist_dir
            embed_cfg: EmbedCfg with model specification
        """
        self.cfg = cfg
        self.embed_cfg = embed_cfg
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create persist directory if it doesn't exist
            os.makedirs(cfg.persist_dir, exist_ok=True)
            
            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=cfg.persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            logger.info(f"ChromaDB initialized at {cfg.persist_dir}")
            
        except ImportError:
            logger.error(
                "chromadb not installed. Install with: pip install chromadb"
            )
            raise ImportError("chromadb is required for ChromaIndex")
        
        # Initialize embedder
        self.embedder = Embedder(embed_cfg.model)
        
        # Collection will be set during build() or retrieve()
        self.collection = None
        self.experiment_name = None
    
    def _get_or_create_collection(self, experiment_name: str):
        """Get or create ChromaDB collection for experiment."""
        if self.collection is None or self.experiment_name != experiment_name:
            self.collection = self.client.get_or_create_collection(
                name=experiment_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            self.experiment_name = experiment_name
            logger.info(f"Loaded collection: {experiment_name}")
        return self.collection
    
    def _manifest_path(self, experiment_name: str) -> str:
        return os.path.join(self.cfg.persist_dir, f"{experiment_name}_manifest.json")

    @staticmethod
    def _corpus_hash(chunks: List[Chunk]) -> str:
        """Deterministic hash of chunk ids+content — detects a stale index
        when the corpus changed even if the chunk COUNT happens to match
        (Failure Mode Register: 'Index stale: corpus changed, index not
        rebuilt')."""
        hasher = hashlib.sha256()
        for chunk in sorted(chunks, key=lambda c: c.id):
            hasher.update(chunk.id.encode())
            hasher.update(chunk.content.encode())
        return hasher.hexdigest()

    def build(self, chunks: List[Chunk], experiment_name: str) -> None:
        """
        Build index by embedding and storing all chunks.
        
        Args:
            chunks: List of Chunk objects to index
            experiment_name: Name of experiment (used as collection name)
        """
        if not chunks:
            logger.warning("No chunks to index")
            return
        
        corpus_hash = self._corpus_hash(chunks)
        manifest_path = self._manifest_path(experiment_name)

        collection = self._get_or_create_collection(experiment_name)
        
        # Check if already built AND corpus hasn't changed (Rule 30: check the
        # completion marker, not just directory/collection existence).
        existing_count = collection.count()
        if existing_count == len(chunks) and os.path.exists(manifest_path):
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                if manifest.get("completed") and manifest.get("corpus_hash") == corpus_hash:
                    logger.info(
                        f"Collection {experiment_name} already has {existing_count} chunks "
                        f"and corpus_hash matches manifest, skipping build"
                    )
                    return
                logger.warning(
                    f"Collection {experiment_name} count matches but corpus_hash changed "
                    f"(stale index) — rebuilding"
                )
            except (json.JSONDecodeError, OSError):
                logger.warning(f"Manifest at {manifest_path} unreadable — rebuilding")
        
        # Clear collection if counts don't match
        if existing_count > 0:
            logger.warning(
                f"Collection {experiment_name} has {existing_count} chunks but expected {len(chunks)}, rebuilding"
            )
            self.client.delete_collection(experiment_name)
            self.collection = None  # invalidate stale handle before refetch
            collection = self._get_or_create_collection(experiment_name)
        
        logger.info(f"Building index for {len(chunks)} chunks...")
        
        # Process in batches of 100
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare batch data
            texts = [chunk.content for chunk in batch]
            ids = [chunk.id for chunk in batch]
            metadatas = [
                {
                    "doc_id": chunk.doc_id,
                    "source_type": chunk.source_type,
                    "chunk_index": chunk.chunk_index,
                    **{k: str(v) for k, v in chunk.metadata.items()}  # Convert all to strings
                }
                for chunk in batch
            ]
            
            # Embed batch
            embeddings = self.embedder.embed(texts)
            
            # Add to collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.debug(f"Indexed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        # Write the completion marker LAST, only after every chunk is indexed
        # (Rule 30) — is_built()/build() check this manifest, not directory
        # existence, so a crash mid-build never looks "done".
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "completed": True,
                    "chunk_count": len(chunks),
                    "corpus_hash": corpus_hash,
                    "embed_model": self.embed_cfg.model,
                },
                f,
            )

        logger.info(f"Index built successfully: {len(chunks)} chunks indexed")
    
    def retrieve(
        self,
        query: str,
        top_k: int,
        experiment_name: str,
        source_type: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve most relevant chunks using vector similarity search.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            experiment_name: Name of experiment (collection name)
            source_type: Optional filter by source_type
            
        Returns:
            List of RetrievedChunk objects sorted by relevance
        """
        collection = self._get_or_create_collection(experiment_name)
        
        # Check if collection is empty
        if collection.count() == 0:
            logger.warning(f"Collection {experiment_name} is empty")
            return []
        
        # Embed query
        query_embedding = self.embedder.embed_one(query)
        
        # Build where clause for filtering
        where = None
        if source_type and source_type != "all":
            where = {"source_type": source_type}
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to RetrievedChunk objects
        retrieved_chunks = []
        
        if results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][i]
                content = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                
                # Convert distance to similarity score (cosine distance -> similarity)
                # ChromaDB returns cosine distance [0, 2], convert to similarity [0, 1]
                score = 1.0 - (distance / 2.0)
                
                # Reconstruct Chunk object
                chunk = Chunk(
                    id=chunk_id,
                    doc_id=metadata["doc_id"],
                    content=content,
                    source_type=metadata["source_type"],
                    chunk_index=int(metadata["chunk_index"]),
                    metadata=metadata
                )
                
                retrieved_chunk = RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    reasoning_path=None  # ChromaDB doesn't provide reasoning paths
                )
                
                retrieved_chunks.append(retrieved_chunk)
        
        logger.debug(
            f"Retrieved {len(retrieved_chunks)} chunks for query: '{query[:50]}...'"
        )
        
        return retrieved_chunks
    
    def is_built(
        self,
        experiment_name: str,
        expected_count: Optional[int] = None,
        corpus_hash: Optional[str] = None,
    ) -> bool:
        """
        Check if index is built for experiment.
        
        Args:
            experiment_name: Name of experiment
            expected_count: Optional expected number of chunks
            corpus_hash: Optional corpus hash (Skill 50B) — if given, also
                requires the build_manifest.json's corpus_hash to match,
                catching a stale index whose chunk count happens to be
                unchanged but whose content changed.
            
        Returns:
            True if collection exists, has the completion marker, and
            (when provided) matches expected_count / corpus_hash.
        """
        try:
            collection = self.client.get_collection(experiment_name)
            actual_count = collection.count()
            
            if expected_count is not None and actual_count != expected_count:
                logger.debug(
                    f"Collection {experiment_name}: {actual_count} chunks "
                    f"(expected {expected_count}) — not built"
                )
                return False

            if corpus_hash is not None:
                manifest_path = self._manifest_path(experiment_name)
                if not os.path.exists(manifest_path):
                    return False
                try:
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                except (json.JSONDecodeError, OSError):
                    return False
                if not manifest.get("completed") or manifest.get("corpus_hash") != corpus_hash:
                    return False

            return actual_count > 0
            
        except Exception:
            return False
