"""GraphRAG Index — Entity-based graph retrieval.

Uses spaCy for entity extraction and NetworkX for graph traversal.
Retrieval combines entity matching with vector similarity.
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import networkx as nx

from raglab.config import EmbedCfg, IndexCfg
from raglab.index.base import BaseIndex
from raglab.types import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class GraphRAGIndex(BaseIndex):
    """
    GraphRAG Index — Entity graph + vector retrieval hybrid.
    
    Strategy:
    1. Extract entities from each chunk via spaCy
    2. Build entity co-occurrence graph (entities = nodes, co-occurrence = edges)
    3. Store entity → chunk_ids mapping
    4. At query time:
       - Extract entities from query
       - Find entities in graph + 1-hop neighbors
       - Collect all chunks containing these entities
       - Re-rank by vector similarity
    
    Benefits:
    - Captures entity relationships
    - Enables graph-based exploration
    - Combines symbolic (entities) with neural (embeddings)
    """
    
    def __init__(self, cfg: IndexCfg, embed_cfg: EmbedCfg):
        """
        Initialize GraphRAG index.
        
        Args:
            cfg: Index configuration
            embed_cfg: Embedding configuration
        """
        self.cfg = cfg
        self.embed_cfg = embed_cfg
        
        self.graph: Optional[nx.DiGraph] = None
        self.entity_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        self.chunk_map: Dict[str, Chunk] = {}
        
        self.persist_dir = Path(cfg.persist_dir) / "graph_rag"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize spaCy
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ Loaded spaCy model: en_core_web_sm")
        except Exception as e:
            logger.error(f"❌ Failed to load spaCy model: {e}")
            logger.error("   Run: python -m spacy download en_core_web_sm")
            raise
        
        # Initialize dense index for re-ranking
        from raglab.index.chroma_index import ChromaIndex
        self.dense_index = ChromaIndex(cfg, embed_cfg)
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities from text using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            List of entity strings (normalized to lowercase)
        """
        doc = self.nlp(text)
        entities = [
            ent.text.lower().strip()
            for ent in doc.ents
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "LAW", "DATE"]
        ]
        return list(set(entities))  # Deduplicate
    
    def build(self, chunks: List[Chunk]) -> None:
        """
        Build entity graph and chunk mappings.
        
        Args:
            chunks: List of chunks to index
        """
        logger.info(f"🔨 Building GraphRAG index for {len(chunks)} chunks...")
        
        # Step 1: Build dense index for re-ranking
        logger.info("   Building dense vector index...")
        self.dense_index.build(chunks)
        
        # Step 2: Extract entities and build graph
        logger.info("   Extracting entities...")
        self.graph = nx.DiGraph()
        
        for chunk in chunks:
            # Extract entities
            entities = self._extract_entities(chunk.content)
            
            if not entities:
                continue
            
            # Store chunk
            self.chunk_map[chunk.id] = chunk
            
            # Build entity → chunk mapping
            for entity in entities:
                self.entity_to_chunks[entity].add(chunk.id)
                
                # Add entity node if not exists
                if not self.graph.has_node(entity):
                    self.graph.add_node(entity)
            
            # Add edges between co-occurring entities
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    # Add edge (or increment weight if exists)
                    if self.graph.has_edge(entity1, entity2):
                        self.graph[entity1][entity2]["weight"] += 1
                    else:
                        self.graph.add_edge(entity1, entity2, weight=1)
                    
                    # Bidirectional
                    if self.graph.has_edge(entity2, entity1):
                        self.graph[entity2][entity1]["weight"] += 1
                    else:
                        self.graph.add_edge(entity2, entity1, weight=1)
        
        logger.info(f"   ✅ Built graph: {self.graph.number_of_nodes()} entities, "
                   f"{self.graph.number_of_edges()} edges")
        
        # Step 3: Persist graph and mappings
        logger.info("   Saving graph to disk...")
        self._save()
        
        logger.info("✅ GraphRAG index built successfully")
    
    def retrieve(
        self,
        query: str,
        top_k: int,
        filter_source_type: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks using entity graph + vector re-ranking.
        
        Strategy:
        1. Extract entities from query
        2. Find entities in graph
        3. Traverse 1-hop neighbors
        4. Collect all chunks containing matched entities
        5. Re-rank by vector similarity
        
        Args:
            query: Query text
            top_k: Number of chunks to return
            filter_source_type: Optional source type filter
            
        Returns:
            List of retrieved chunks sorted by relevance
        """
        if not self.is_built(""):
            logger.warning("⚠️  GraphRAG index not built, falling back to dense retrieval")
            return self.dense_index.retrieve(query, top_k, filter_source_type)
        
        logger.info(f"🔍 GraphRAG retrieval: {query[:60]}...")
        
        # Step 1: Extract entities from query
        query_entities = self._extract_entities(query)
        logger.info(f"   Extracted {len(query_entities)} entities: {query_entities}")
        
        if not query_entities:
            # No entities found, fall back to dense retrieval
            logger.info("   No entities found, using dense retrieval")
            return self.dense_index.retrieve(query, top_k, filter_source_type)
        
        # Step 2: Find entities in graph and collect 1-hop neighbors
        matched_entities = set()
        
        for entity in query_entities:
            if entity in self.graph:
                matched_entities.add(entity)
                
                # Add 1-hop neighbors
                neighbors = list(self.graph.neighbors(entity))
                matched_entities.update(neighbors[:10])  # Limit neighbors to avoid explosion
        
        logger.info(f"   Matched {len(matched_entities)} entities (including neighbors)")
        
        if not matched_entities:
            # No entities matched, fall back to dense retrieval
            logger.info("   No entities matched in graph, using dense retrieval")
            return self.dense_index.retrieve(query, top_k, filter_source_type)
        
        # Step 3: Collect candidate chunks
        candidate_chunk_ids: Set[str] = set()
        
        for entity in matched_entities:
            chunk_ids = self.entity_to_chunks.get(entity, set())
            candidate_chunk_ids.update(chunk_ids)
        
        logger.info(f"   Collected {len(candidate_chunk_ids)} candidate chunks")
        
        if not candidate_chunk_ids:
            return self.dense_index.retrieve(query, top_k, filter_source_type)
        
        # Step 4: Build candidate chunks list
        candidates = []
        for chunk_id in candidate_chunk_ids:
            chunk = self.chunk_map.get(chunk_id)
            if chunk:
                # Apply source type filter
                if filter_source_type and chunk.source_type != filter_source_type:
                    continue
                
                candidates.append(RetrievedChunk(
                    chunk=chunk,
                    score=0.0,  # Will be set by re-ranking
                    reasoning_path=None
                ))
        
        logger.info(f"   {len(candidates)} candidates after filtering")
        
        # Step 5: Re-rank by vector similarity
        if not candidates:
            return []
        
        # Use dense index to compute similarity scores
        logger.info("   Re-ranking by vector similarity...")
        from raglab.utils.embedder import Embedder
        
        embedder = Embedder(self.embed_cfg)
        query_embedding = embedder.embed_one(query)
        
        # Compute scores
        for candidate in candidates:
            # Get chunk embedding from dense index if available
            chunk_embedding = embedder.embed_one(candidate.chunk.content)
            
            # Cosine similarity
            import numpy as np
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            candidate.score = float(similarity)
        
        # Sort by score descending
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        result = candidates[:top_k]
        logger.info(f"   ✅ Returning {len(result)} chunks (scores: "
                   f"{result[0].score:.3f} - {result[-1].score:.3f})")
        
        return result
    
    def is_built(self, experiment_name: str) -> bool:
        """Check if index is built."""
        graph_path = self.persist_dir / "graph.pkl"
        mapping_path = self.persist_dir / "entity_mapping.pkl"
        chunks_path = self.persist_dir / "chunks.pkl"
        
        return (
            graph_path.exists() and
            mapping_path.exists() and
            chunks_path.exists() and
            self.dense_index.is_built(experiment_name)
        )
    
    def _save(self) -> None:
        """Persist graph and mappings to disk."""
        # Save graph
        graph_path = self.persist_dir / "graph.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(self.graph, f)
        
        # Save entity mappings (convert sets to lists for JSON)
        mapping_path = self.persist_dir / "entity_mapping.pkl"
        mapping_serializable = {
            entity: list(chunks)
            for entity, chunks in self.entity_to_chunks.items()
        }
        with open(mapping_path, "wb") as f:
            pickle.dump(mapping_serializable, f)
        
        # Save chunk map
        chunks_path = self.persist_dir / "chunks.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunk_map, f)
        
        logger.info(f"   💾 Saved GraphRAG index to {self.persist_dir}")
    
    def _load(self) -> None:
        """Load graph and mappings from disk."""
        graph_path = self.persist_dir / "graph.pkl"
        mapping_path = self.persist_dir / "entity_mapping.pkl"
        chunks_path = self.persist_dir / "chunks.pkl"
        
        # Load graph
        with open(graph_path, "rb") as f:
            self.graph = pickle.load(f)
        
        # Load entity mappings (convert lists back to sets)
        with open(mapping_path, "rb") as f:
            mapping_serializable = pickle.load(f)
            self.entity_to_chunks = {
                entity: set(chunks)
                for entity, chunks in mapping_serializable.items()
            }
        
        # Load chunk map
        with open(chunks_path, "rb") as f:
            self.chunk_map = pickle.load(f)
        
        logger.info(f"   📂 Loaded GraphRAG index from {self.persist_dir}")
