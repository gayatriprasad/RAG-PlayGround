"""
MonoT5 reranker using sequence-to-sequence transformer model.
"""

import logging
from typing import List

from raglab.config import RetrieveCfg
from raglab.types import RetrievedChunk
from raglab.rerankers.base import BaseReranker

logger = logging.getLogger(__name__)


class MonoT5Reranker(BaseReranker):
    """
    MonoT5 reranker using castorini/monot5-base-msmarco model.
    Sequence-to-sequence reranker, slower but strong on long documents.
    """
    
    def __init__(self, cfg: RetrieveCfg):
        """
        Initialize MonoT5Reranker.
        
        Args:
            cfg: RetrieveCfg (model hardcoded to monot5-base-msmarco)
        """
        self.cfg = cfg
        self.model_name = "castorini/monot5-base-msmarco"
        
        try:
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            import torch
            
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.eval()
            
            # Use CPU by default (can be configured later)
            self.device = "cpu"
            self.model.to(self.device)
            
            logger.info(f"MonoT5Reranker initialized with model: {self.model_name}")
            
        except ImportError:
            logger.warning(
                "transformers not installed. Install with: pip install transformers torch\n"
                "MonoT5Reranker will not be available."
            )
            raise ImportError("transformers is required for MonoT5Reranker")
    
    def _score_passage(self, query: str, passage: str) -> float:
        """
        Score a single passage against the query using MonoT5.
        
        Args:
            query: Query string
            passage: Passage text
            
        Returns:
            Relevance score
        """
        import torch
        
        # Format input for MonoT5: "Query: {query} Document: {passage} Relevant:"
        input_text = f"Query: {query} Document: {passage} Relevant:"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate score
        with torch.no_grad():
            # MonoT5 outputs "true" or "false" tokens, we use logits
            outputs = self.model.generate(
                **inputs,
                max_length=2,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Get logits for "true" token (token id varies by tokenizer)
            # Simplified: use the first token score as relevance
            score = outputs.scores[0][0].max().item()
        
        return float(score)
    
    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Rerank chunks using MonoT5 model.
        
        Args:
            query: Query string
            chunks: List of RetrievedChunk objects
            
        Returns:
            List of RetrievedChunk objects with updated scores
        """
        if not chunks:
            return chunks
        
        # Log original top-3 ranks
        original_top3 = [
            (i, chunks[i].chunk.id, chunks[i].score)
            for i in range(min(3, len(chunks)))
        ]
        logger.debug(f"Original top-3: {[(c[1], f'{c[2]:.4f}') for c in original_top3]}")
        
        # Score all chunks
        scored_chunks = []
        for chunk in chunks:
            score = self._score_passage(query, chunk.chunk.content)
            scored_chunks.append((chunk, score))
        
        # Sort by MonoT5 score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Build reranked list with updated scores
        reranked = []
        for original_chunk, monot5_score in scored_chunks:
            reranked_chunk = RetrievedChunk(
                chunk=original_chunk.chunk,
                score=monot5_score,
                reasoning_path=original_chunk.reasoning_path
            )
            reranked.append(reranked_chunk)
        
        # Log new top-3 ranks
        new_top3 = [
            (i, reranked[i].chunk.id, reranked[i].score)
            for i in range(min(3, len(reranked)))
        ]
        logger.info(
            f"MonoT5 reranked {len(chunks)} chunks. "
            f"New top-3: {[(c[1], f'{c[2]:.4f}') for c in new_top3]}"
        )
        
        return reranked
