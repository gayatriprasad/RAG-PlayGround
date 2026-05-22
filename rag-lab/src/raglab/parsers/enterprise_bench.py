"""
EnterpriseRAG-Bench data loader for questions and documents.
Loads ground-truth Q&A pairs and raw documents from corpus directories.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from raglab.config import BenchmarkCfg
from raglab.types import Document, Question

logger = logging.getLogger(__name__)


def load_questions(cfg: BenchmarkCfg) -> List[Question]:
    """
    Load questions from JSONL file with filtering and limits.
    
    Args:
        cfg: BenchmarkCfg with questions_path, source_types, question_categories, max_questions
        
    Returns:
        List of Question objects filtered and capped per configuration
        
    Raises:
        FileNotFoundError: If questions file doesn't exist
        ValueError: If JSONL format is invalid
    """
    questions_path = Path(cfg.questions_path)
    
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    
    questions = []
    
    with open(questions_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # Map JSONL fields to Question model
                # Supports both formats:
                #   {id, question, answer, source_type, category}
                #   {id, text, ground_truth, source_type, category}
                question = Question(
                    id=data['id'],
                    text=data.get('text') or data['question'],
                    ground_truth=data.get('ground_truth') or data['answer'],
                    source_type=data['source_type'],
                    category=data['category']
                )
                
                # Filter by source_types if specified
                if cfg.source_types:
                    # Handle multi-source questions (e.g., "confluence,slack" or "all")
                    question_sources = question.source_type.split(',')
                    if question.source_type != "all" and not any(src.strip() in cfg.source_types for src in question_sources):
                        continue
                
                # Filter by question_categories if specified
                if cfg.question_categories and question.category not in cfg.question_categories:
                    continue
                
                questions.append(question)
                
                # Cap at max_questions
                if len(questions) >= cfg.max_questions:
                    break
                    
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping invalid line {line_num} in {questions_path}: {e}")
                continue
    
    logger.info(f"Loaded {len(questions)} questions from {questions_path}")
    return questions


def load_documents(cfg: BenchmarkCfg) -> List[Document]:
    """
    Load raw documents from corpus/raw/<source_type>/ directories.
    
    Args:
        cfg: BenchmarkCfg with source_types list
        
    Returns:
        List of Document objects with content, source_type, and metadata
        
    Raises:
        FileNotFoundError: If corpus directory doesn't exist
    """
    documents = []
    
    # Determine corpus base directory (relative to questions_path or current dir)
    questions_path = Path(cfg.questions_path)
    if questions_path.is_absolute():
        # Assume corpus is sibling to golden directory
        corpus_base = questions_path.parent.parent / "corpus" / "raw"
    else:
        corpus_base = Path("corpus/raw")
    
    if not corpus_base.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_base}")
    
    for source_type in cfg.source_types:
        source_dir = corpus_base / source_type
        
        if not source_dir.exists():
            logger.warning(f"Source directory not found, skipping: {source_dir}")
            continue
        
        # Load all .txt and .md files
        file_patterns = ['*.txt', '*.md']
        files = []
        for pattern in file_patterns:
            files.extend(source_dir.glob(pattern))
        
        for file_path in sorted(files):
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Create Document object
                doc = Document(
                    id=f"{source_type}_{file_path.stem}",
                    content=content,
                    source_type=source_type,
                    metadata={
                        'filename': file_path.name,
                        'filepath': str(file_path),
                        'size_bytes': file_path.stat().st_size
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        logger.info(f"Loaded {len([d for d in documents if d.source_type == source_type])} documents from {source_type}")
    
    logger.info(f"Loaded {len(documents)} total documents across {len(cfg.source_types)} source types")
    return documents


def download_bench_slice(
    source_types: List[str],
    out_dir: str,
    max_docs_per_type: Optional[int] = None
) -> None:
    """
    Download EnterpriseRAG-Bench slice from HuggingFace.
    
    Uses huggingface_hub to stream-download only the requested source_type slices
    from onyx-dot-app/EnterpriseRAG-Bench dataset. Saves raw files to corpus/raw/.
    Skips if already present. Logs progress.
    
    Args:
        source_types: List of source types to download (e.g., ['confluence', 'github'])
        out_dir: Output directory for raw documents (e.g., 'corpus/raw/')
        max_docs_per_type: Optional limit on documents per source type
        
    Raises:
        ImportError: If huggingface_hub or datasets not installed
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Required packages not installed: pip install datasets huggingface-hub")
        raise ImportError("Install datasets and huggingface-hub: pip install datasets huggingface-hub")
    
    base_dir = Path(out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Check which source types already have data (skip if present)
    needed_types = []
    for source_type in source_types:
        source_dir = base_dir / source_type
        if source_dir.exists():
            existing = list(source_dir.glob("*.txt")) + list(source_dir.glob("*.md"))
            if existing:
                logger.info(f"Skipping {source_type}: {len(existing)} files already present")
                continue
        needed_types.append(source_type)
    
    if not needed_types:
        logger.info("All requested source types already present. Nothing to download.")
        return
    
    logger.info(f"Downloading EnterpriseRAG-Bench slice for: {', '.join(needed_types)}")
    
    try:
        dataset = load_dataset(
            "onyx-dot-app/EnterpriseRAG-Bench",
            split="train",
            streaming=True
        )
        
        doc_counts = {st: 0 for st in needed_types}
        
        for item in dataset:
            source_type = item.get("source_type", "unknown")
            
            if source_type not in needed_types:
                continue
            
            if max_docs_per_type and doc_counts[source_type] >= max_docs_per_type:
                # Check if all types are filled
                if all(doc_counts[st] >= max_docs_per_type for st in needed_types):
                    break
                continue
            
            source_dir = base_dir / source_type
            source_dir.mkdir(parents=True, exist_ok=True)
            
            doc_id = item.get("id", f"doc_{doc_counts[source_type]}")
            content = item.get("content", item.get("text", ""))
            
            file_path = source_dir / f"{doc_id}.txt"
            file_path.write_text(content, encoding="utf-8")
            
            doc_counts[source_type] += 1
            
            # Log progress every 100 docs
            total = sum(doc_counts.values())
            if total % 100 == 0:
                logger.info(f"  Downloaded {total} documents so far...")
        
        logger.info(f"Download complete. Documents per type:")
        for st, count in doc_counts.items():
            logger.info(f"  {st}: {count} documents")
    
    except Exception as e:
        logger.warning(
            f"Could not download from HuggingFace: {e}. "
            f"Place documents manually in {base_dir}/<source_type>/ directories."
        )
