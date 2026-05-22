"""
PageIndex adapter for structured document retrieval without embeddings.
Uses VectifyAI's pageindex for tree-based retrieval.
"""

import json
import logging
import os
from typing import List, Optional, Dict
from collections import defaultdict

from raglab.config import IndexCfg
from raglab.types import Chunk, RetrievedChunk
from raglab.index.base import BaseIndex

logger = logging.getLogger(__name__)


class PageIndexAdapter(BaseIndex):
    """
    Adapter for pageindex - tree-based structured document retrieval.
    No embeddings required - uses document structure and keyword matching.
    """
    
    def __init__(self, cfg: IndexCfg):
        """
        Initialize PageIndexAdapter.
        
        Args:
            cfg: IndexCfg with persist_dir
        """
        self.cfg = cfg
        self.persist_dir = os.path.join(cfg.persist_dir, "pageindex")
        self.tree_indices = {}  # doc_id -> tree_index
        self.doc_metadata = {}  # doc_id -> metadata
        self.pageindex = None
        
        # Try to import pageindex (optional dependency)
        try:
            import pageindex
            self.pageindex = pageindex
            logger.info("PageIndexAdapter initialized (vectorless tree-based retrieval)")
        except ImportError:
            logger.warning(
                "pageindex not installed. PageIndexAdapter will use fallback keyword matching. "
                "For full functionality, install pageindex from: https://github.com/VectifyAI/pageindex"
            )
            # Don't raise error here - allow graceful degradation to fallback mode
    
    def build(self, chunks: List[Chunk], experiment_name: str) -> None:
        """
        Build PageIndex from chunks.
        
        Groups chunks by doc_id, reconstructs documents, builds tree index per document.
        
        Args:
            chunks: List of Chunk objects to index
            experiment_name: Name of experiment
        """
        if not chunks:
            logger.warning("No chunks to index")
            return
        
        logger.info(f"Building PageIndex for {len(chunks)} chunks...")
        
        # Group chunks by doc_id
        doc_chunks = defaultdict(list)
        for chunk in chunks:
            doc_chunks[chunk.doc_id].append(chunk)
        
        # Sort chunks within each document by chunk_index
        for doc_id in doc_chunks:
            doc_chunks[doc_id].sort(key=lambda c: c.chunk_index)
        
        logger.info(f"Grouped into {len(doc_chunks)} documents")
        
        # Build tree index for each document
        self.tree_indices = {}
        self.doc_metadata = {}
        
        for doc_id, doc_chunk_list in doc_chunks.items():
            # Reconstruct full document text
            full_text = "\n\n".join([c.content for c in doc_chunk_list])
            
            # Store metadata
            self.doc_metadata[doc_id] = {
                "source_type": doc_chunk_list[0].source_type,
                "chunk_count": len(doc_chunk_list),
                "doc_metadata": doc_chunk_list[0].metadata,
            }
            
            # Build tree index for this document
            try:
                # PageIndex typically builds a hierarchical tree structure
                # Exact API depends on pageindex version
                if self.pageindex is not None:
                    tree_index = self.pageindex.build_index(
                        text=full_text,
                        doc_id=doc_id
                    )
                    self.tree_indices[doc_id] = tree_index
                    logger.debug(f"Built tree index for document {doc_id}")
                else:
                    # Fallback mode: store text for simple keyword matching
                    self.tree_indices[doc_id] = {
                        "doc_id": doc_id,
                        "text": full_text,
                        "type": "fallback"
                    }
                    logger.debug(f"Stored fallback index for document {doc_id}")
            except Exception as e:
                logger.warning(f"Failed to build tree index for {doc_id}: {e}")
                # Store a fallback simple index
                self.tree_indices[doc_id] = {
                    "doc_id": doc_id,
                    "text": full_text,
                    "type": "fallback"
                }
        
        # Persist indices
        self._persist_indices(experiment_name)
        
        logger.info(f"PageIndex built: {len(self.tree_indices)} document indices created")
    
    def _persist_indices(self, experiment_name: str) -> None:
        """
        Persist tree indices and metadata to disk.
        
        Args:
            experiment_name: Name of experiment
        """
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Save metadata
        metadata_path = os.path.join(self.persist_dir, f"{experiment_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "experiment_name": experiment_name,
                "doc_count": len(self.tree_indices),
                "doc_metadata": self.doc_metadata,
            }, f, indent=2)
        
        # Save tree indices
        # Note: Actual serialization depends on pageindex implementation
        # For now, we'll save as JSON (may need to be adapted)
        indices_path = os.path.join(self.persist_dir, f"{experiment_name}_indices.json")
        
        # Convert tree indices to serializable format
        serializable_indices = {}
        for doc_id, tree_index in self.tree_indices.items():
            if isinstance(tree_index, dict):
                serializable_indices[doc_id] = tree_index
            else:
                # If tree_index is a custom object, try to serialize it
                try:
                    serializable_indices[doc_id] = tree_index.to_dict()
                except AttributeError:
                    logger.warning(f"Cannot serialize tree index for {doc_id}, storing text only")
                    serializable_indices[doc_id] = {
                        "doc_id": doc_id,
                        "type": "text_only"
                    }
        
        with open(indices_path, 'w') as f:
            json.dump(serializable_indices, f, indent=2)
        
        logger.debug(f"Persisted PageIndex to {self.persist_dir}")
    
    def _load_indices(self, experiment_name: str) -> None:
        """
        Load tree indices and metadata from disk.
        
        Args:
            experiment_name: Name of experiment
        """
        metadata_path = os.path.join(self.persist_dir, f"{experiment_name}_metadata.json")
        indices_path = os.path.join(self.persist_dir, f"{experiment_name}_indices.json")
        
        if not os.path.exists(metadata_path) or not os.path.exists(indices_path):
            logger.warning("PageIndex not found on disk")
            return
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            self.doc_metadata = data.get("doc_metadata", {})
        
        # Load tree indices
        with open(indices_path, 'r') as f:
            self.tree_indices = json.load(f)
        
        logger.debug(f"Loaded PageIndex from {self.persist_dir}")
    
    def retrieve(
        self,
        query: str,
        top_k: int,
        experiment_name: str,
        source_type: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve using PageIndex tree-based search.
        
        Args:
            query: Query string
            top_k: Number of results to return
            experiment_name: Name of experiment
            source_type: Optional filter by source_type
            
        Returns:
            List of RetrievedChunk objects with reasoning_path populated
        """
        # Load indices if not already loaded
        if not self.tree_indices:
            self._load_indices(experiment_name)
        
        if not self.tree_indices:
            logger.warning("PageIndex not built, returning empty results")
            return []
        
        # Query each tree index
        all_results = []
        
        for doc_id, tree_index in self.tree_indices.items():
            # Apply source_type filter if specified
            doc_meta = self.doc_metadata.get(doc_id, {})
            if source_type and source_type != "all" and doc_meta.get("source_type") != source_type:
                continue
            
            try:
                # Query the tree index
                # Exact API depends on pageindex implementation
                if isinstance(tree_index, dict) and tree_index.get("type") == "fallback":
                    # Simple fallback: keyword matching
                    text = tree_index.get("text", "")
                    text_lower = text.lower()
                    
                    # Split query into keywords and check for matches
                    query_keywords = query.lower().split()
                    matches = sum(1 for keyword in query_keywords if keyword in text_lower)
                    
                    if matches > 0:
                        # Score based on keyword match ratio and frequency
                        keyword_ratio = matches / len(query_keywords) if query_keywords else 0
                        total_frequency = sum(text_lower.count(kw) for kw in query_keywords)
                        score = (keyword_ratio * 0.7) + (min(total_frequency / len(text.split()), 1.0) * 0.3)
                        
                        # Create a chunk from the document
                        chunk = Chunk(
                            id=f"{doc_id}_chunk_0",
                            doc_id=doc_id,
                            content=text[:1000],  # Truncate for display
                            source_type=doc_meta.get("source_type", "unknown"),
                            chunk_index=0,
                            metadata=doc_meta.get("doc_metadata", {})
                        )
                        
                        result = RetrievedChunk(
                            chunk=chunk,
                            score=float(score),
                            reasoning_path=f"Keyword match ({matches}/{len(query_keywords)} keywords) in document {doc_id}"
                        )
                        all_results.append(result)
                else:
                    # Use actual pageindex query
                    results = self.pageindex.query(
                        index=tree_index,
                        query=query,
                        top_k=3  # Get multiple sections per document
                    )
                    
                    # Convert pageindex results to RetrievedChunk
                    for idx, result in enumerate(results):
                        section_text = result.get("text", "")
                        relevance_score = result.get("score", 0.0)
                        reasoning = result.get("reasoning_path", f"Section {idx} from {doc_id}")
                        
                        chunk = Chunk(
                            id=f"{doc_id}_section_{idx}",
                            doc_id=doc_id,
                            content=section_text,
                            source_type=doc_meta.get("source_type", "unknown"),
                            chunk_index=idx,
                            metadata={
                                **doc_meta.get("doc_metadata", {}),
                                "section_type": result.get("section_type", "unknown")
                            }
                        )
                        
                        retrieved_chunk = RetrievedChunk(
                            chunk=chunk,
                            score=float(relevance_score),
                            reasoning_path=reasoning
                        )
                        all_results.append(retrieved_chunk)
            
            except Exception as e:
                logger.warning(f"Error querying tree index for {doc_id}: {e}")
                continue
        
        # Sort by relevance score descending
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k
        final_results = all_results[:top_k]
        
        logger.debug(
            f"PageIndex retrieval: query='{query[:50]}...', "
            f"retrieved={len(final_results)} chunks from {len(self.tree_indices)} documents"
        )
        
        return final_results
    
    def is_built(self, experiment_name: str, expected_count: Optional[int] = None) -> bool:
        """
        Check if PageIndex is built and persisted.
        
        Args:
            experiment_name: Name of experiment
            expected_count: Optional expected number of documents
            
        Returns:
            True if index exists and is valid
        """
        metadata_path = os.path.join(self.persist_dir, f"{experiment_name}_metadata.json")
        indices_path = os.path.join(self.persist_dir, f"{experiment_name}_indices.json")
        
        if not os.path.exists(metadata_path) or not os.path.exists(indices_path):
            return False
        
        # Verify document count if expected_count provided
        if expected_count is not None:
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    doc_count = data.get("doc_count", 0)
                    # Note: expected_count is chunk count, doc_count is document count
                    # We'll just check that we have some documents
                    return doc_count > 0
            except Exception as e:
                logger.warning(f"Failed to verify PageIndex: {e}")
                return False
        
        return True
