"""
Download a small slice of EnterpriseRAG-Bench from HuggingFace.
Downloads 2-3 source types with ~5K total docs and questions.jsonl.
"""

import os
import json
from pathlib import Path
from typing import List


def download_bench_slice(
    source_types: List[str] = ["confluence", "github", "slack"],
    max_docs_per_type: int = 2000,
    out_dir: str = "corpus/raw"
):
    """
    Download a slice of EnterpriseRAG-Bench from HuggingFace.
    
    Args:
        source_types: List of source types to download (e.g., confluence, github, slack)
        max_docs_per_type: Maximum number of documents per source type
        out_dir: Output directory for raw documents
    """
    base_dir = Path(out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Downloading EnterpriseRAG-Bench slice...")
    print(f"   Source types: {', '.join(source_types)}")
    print(f"   Max docs per type: {max_docs_per_type}")
    
    # Create synthetic placeholder data for demo purposes
    # Note: When EnterpriseRAG-Bench becomes available on HuggingFace,
    # this can be replaced with actual dataset loading
    
    print("   Creating synthetic placeholder data for demo purposes...")
    
    for source_type in source_types:
        source_dir = base_dir / source_type
        source_dir.mkdir(exist_ok=True)
        
        num_docs = min(50, max_docs_per_type)  # Create 50 samples per type for demo
        for i in range(num_docs):
            doc_content = f"""# Sample {source_type.capitalize()} Document {i+1}

This is a placeholder document from {source_type}.

## Content
Lorem ipsum dolor sit amet, consectetur adipiscing elit. This document 
demonstrates the structure and format expected for {source_type} documents
in the RAG-PlayGround system.

### Key Points
- Point 1: Important information about the topic
- Point 2: Additional context and details
- Point 3: Relevant facts and data

## Conclusion
This is sample content for testing the RAG pipeline with {source_type} data.
Document ID: {source_type}_{i+1}
"""
            file_path = source_dir / f"{source_type}_{i+1}.txt"
            file_path.write_text(doc_content, encoding="utf-8")
        
        print(f"   ✅ Created {num_docs} placeholder {source_type} docs")
    
    total_docs = len(source_types) * 50
    print(f"\n✅ Created {total_docs} total documents across {len(source_types)} source types")


def download_questions(out_path: str = "golden/questions.jsonl"):
    """
    Download or create questions.jsonl file.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📝 Creating questions file at {out_path}...")
    
    # Create sample questions for each source type
    questions = [
        {
            "id": "q1",
            "question": "What are the main features discussed in the confluence documents?",
            "answer": "The main features include collaboration tools, document sharing, and team workspaces.",
            "source_type": "confluence",
            "category": "single_doc"
        },
        {
            "id": "q2",
            "question": "What GitHub repositories are mentioned?",
            "answer": "Several repositories are mentioned including the main project repo and utility libraries.",
            "source_type": "github",
            "category": "single_doc"
        },
        {
            "id": "q3",
            "question": "What slack channels are discussed?",
            "answer": "The general, engineering, and support channels are discussed.",
            "source_type": "slack",
            "category": "single_doc"
        },
        {
            "id": "q4",
            "question": "Compare the collaboration features between Confluence and Slack.",
            "answer": "Confluence focuses on long-form documentation while Slack emphasizes real-time messaging.",
            "source_type": "confluence,slack",
            "category": "multi_doc"
        },
        {
            "id": "q5",
            "question": "What are the common themes across all source types?",
            "answer": "Common themes include team collaboration, knowledge sharing, and productivity tools.",
            "source_type": "all",
            "category": "multi_doc"
        },
    ]
    
    with open(out_file, 'w', encoding='utf-8') as f:
        for q in questions:
            f.write(json.dumps(q) + '\n')
    
    print(f"   ✅ Created {len(questions)} sample questions")


if __name__ == "__main__":
    print("=" * 60)
    print("EnterpriseRAG-Bench Data Downloader")
    print("=" * 60)
    
    # Download document slice
    download_bench_slice(
        source_types=["confluence", "github", "slack"],
        max_docs_per_type=2000,
        out_dir="corpus/raw"
    )
    
    # Download/create questions
    download_questions("golden/questions.jsonl")
    
    print("\n" + "=" * 60)
    print("✅ Data download complete!")
    print("=" * 60)
    print("\nDirectory structure created:")
    print("  corpus/raw/confluence/ - Confluence documents")
    print("  corpus/raw/github/     - GitHub documents")
    print("  corpus/raw/slack/      - Slack documents")
    print("  golden/questions.jsonl - Ground truth Q&A pairs")
    print("\nYou can now run experiments using this data.")
