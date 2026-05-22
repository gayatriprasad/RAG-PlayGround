"""
Document normalization and deduplication utilities.
"""

import hashlib
import logging
from typing import List, Set, Dict
from datetime import datetime

from raglab.types import Document

logger = logging.getLogger(__name__)


class DocumentNormalizer:
    """
    Normalizes and deduplicates documents during ingestion.
    """
    
    def normalize(self, docs: List[Document]) -> List[Document]:
        """
        Normalize document content and enrich metadata.
        
        Args:
            docs: List of Document objects
            
        Returns:
            List of normalized Document objects
        """
        normalized_docs = []
        
        for doc in docs:
            # 1. Whitespace normalization
            content = doc.content
            # Collapse multiple newlines to double newline
            import re
            content = re.sub(r'\n\n+', '\n\n', content)
            # Strip leading/trailing whitespace
            content = content.strip()
            
            # 2. Encoding fix
            try:
                # Encode to UTF-8 and decode with error replacement
                content = content.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            except Exception as e:
                logger.warning(f"Encoding fix failed for doc {doc.id}: {e}")
            
            # 3. Metadata enrichment
            metadata = doc.metadata.copy()
            metadata["ingested_at"] = datetime.utcnow().isoformat()
            metadata["char_count"] = len(content)
            metadata["word_count"] = len(content.split())
            
            # Content fingerprint (version)
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            metadata["version"] = content_hash[:8]
            
            # Create normalized document
            normalized_doc = Document(
                id=doc.id,
                content=content,
                source_type=doc.source_type,
                metadata=metadata
            )
            normalized_docs.append(normalized_doc)
        
        logger.info(f"Normalized {len(normalized_docs)} documents")
        return normalized_docs
    
    def deduplicate(self, docs: List[Document]) -> List[Document]:
        """
        Remove duplicate documents (exact and near-duplicate).
        
        Args:
            docs: List of Document objects
            
        Returns:
            Deduplicated list of Document objects
        """
        if not docs:
            return docs
        
        # Track duplicates
        exact_dupes = 0
        near_dupes = 0
        
        # Exact deduplication by content hash
        seen_hashes: Set[str] = set()
        unique_docs: List[Document] = []
        
        for doc in docs:
            content_hash = hashlib.sha256(doc.content.encode('utf-8')).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
            else:
                exact_dupes += 1
        
        # Near-deduplication
        # Group by source_type for efficiency
        by_source_type: Dict[str, List[Document]] = {}
        for doc in unique_docs:
            if doc.source_type not in by_source_type:
                by_source_type[doc.source_type] = []
            by_source_type[doc.source_type].append(doc)
        
        final_docs: List[Document] = []
        
        for source_type, source_docs in by_source_type.items():
            # Sort by word count for stable near-dedup
            source_docs_sorted = sorted(
                source_docs,
                key=lambda d: d.metadata.get("word_count", 0),
                reverse=True
            )
            
            kept_docs: List[Document] = []
            
            for doc in source_docs_sorted:
                is_near_dupe = False
                doc_word_count = doc.metadata.get("word_count", 0)
                doc_words = set(doc.content.lower().split())
                
                for kept_doc in kept_docs:
                    kept_word_count = kept_doc.metadata.get("word_count", 0)
                    
                    # Check if char counts are within 5%
                    if abs(doc_word_count - kept_word_count) / max(doc_word_count, kept_word_count, 1) <= 0.05:
                        # Compute Jaccard similarity
                        kept_words = set(kept_doc.content.lower().split())
                        
                        intersection = len(doc_words & kept_words)
                        union = len(doc_words | kept_words)
                        
                        if union > 0:
                            jaccard_sim = intersection / union
                            
                            if jaccard_sim > 0.85:
                                is_near_dupe = True
                                near_dupes += 1
                                break
                
                if not is_near_dupe:
                    kept_docs.append(doc)
            
            final_docs.extend(kept_docs)
        
        logger.info(
            f"Removed {exact_dupes + near_dupes} duplicates "
            f"({exact_dupes} exact, {near_dupes} near-duplicate)"
        )
        
        return final_docs
