"""MCP Server — Expose RAG Playground as Model Context Protocol tools.

This server exposes RAG pipeline functionality to MCP-compatible clients
(e.g., Claude Desktop, Cline, other MCP tools).

Usage:
    python api/mcp_server.py

Claude Desktop Configuration:
    Add to ~/Library/Application Support/Claude/claude_desktop_config.json:
    
    {
      "mcpServers": {
        "rag-playground": {
          "command": "python",
          "args": ["/path/to/RAG-PlayGround/api/mcp_server.py"],
          "env": {
            "PYTHONPATH": "/path/to/RAG-PlayGround/rag-lab/src"
          }
        }
      }
    }
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add rag-lab/src to path
_RAG_LAB_SRC = Path(__file__).resolve().parents[1] / "rag-lab" / "src"
if str(_RAG_LAB_SRC) not in sys.path:
    sys.path.insert(0, str(_RAG_LAB_SRC))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    logger.error("❌ fastmcp not installed. Run: pip install mcp")
    sys.exit(1)

from raglab.classifiers import get_classifier
from raglab.config import Config
from raglab.index import get_index
from raglab.pipelines import NaiveRAGPipeline, AgenticRAGPipeline
from raglab.rerankers import get_reranker
from raglab.types import Question

# ─── Global State ──────────────────────────────────────────────────────────────

_RAG_LAB_ROOT = Path(__file__).resolve().parents[1] / "rag-lab"
_EXPERIMENTS_DIR = _RAG_LAB_ROOT / "experiments"
_OUT_DIR = _RAG_LAB_ROOT / "out" / "raglab_out"

# Cache loaded config and components
_CACHE: Dict[str, Any] = {}


def _load_config(experiment: str = "01_format_comparison") -> Config:
    """Load config from experiment directory."""
    cache_key = f"config_{experiment}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    
    config_path = _EXPERIMENTS_DIR / experiment / "config.yaml"
    if not config_path.exists():
        raise ValueError(f"Experiment '{experiment}' not found at {config_path}")
    
    import yaml
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    
    cfg = Config(**raw)
    _CACHE[cache_key] = cfg
    return cfg


def _get_index(experiment: str):
    """Get or create index for experiment."""
    cache_key = f"index_{experiment}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    
    cfg = _load_config(experiment)
    
    # Change to rag-lab dir for relative paths
    import os
    original_cwd = os.getcwd()
    os.chdir(str(_RAG_LAB_ROOT))
    
    try:
        index = get_index(cfg.index, cfg.embed)
        
        # Check if built
        if hasattr(index, 'is_built') and not index.is_built(cfg.experiment.name):
            logger.warning(f"⚠️  Index not built for experiment '{experiment}'")
        
        _CACHE[cache_key] = index
        return index
    finally:
        os.chdir(original_cwd)


# ─── MCP Server ────────────────────────────────────────────────────────────────

app = Server("rag-playground")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available RAG tools."""
    return [
        Tool(
            name="retrieve",
            description=(
                "Retrieve relevant chunks from the enterprise corpus for a given query. "
                "Returns top_k most relevant document chunks with scores and metadata."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "source_type": {
                        "type": "string",
                        "description": "Filter by source type (confluence, github, jira, slack, etc.)",
                        "default": "all"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "experiment": {
                        "type": "string",
                        "description": "Experiment name to use (loads its config)",
                        "default": "01_format_comparison"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="ask",
            description=(
                "Run full RAG pipeline and return answer with citations. "
                "Automatically classifies intent and routes to appropriate pipeline "
                "(naive or agentic). Returns generated answer with retrieved chunks."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to answer"
                    },
                    "source_type": {
                        "type": "string",
                        "description": "Filter retrieval to this source type",
                        "default": "all"
                    },
                    "pipeline": {
                        "type": "string",
                        "description": "Force a specific pipeline (naive or agentic), or 'auto' to classify",
                        "enum": ["auto", "naive", "agentic"],
                        "default": "auto"
                    },
                    "experiment": {
                        "type": "string",
                        "description": "Experiment name to use",
                        "default": "01_format_comparison"
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="index_status",
            description=(
                "Get current index statistics including document count, "
                "last updated time, backend type, and index health status."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment": {
                        "type": "string",
                        "description": "Experiment name",
                        "default": "01_format_comparison"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="list_experiments",
            description=(
                "List all available experiments with their configurations "
                "and result summaries if available."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "retrieve":
            return await _tool_retrieve(arguments)
        elif name == "ask":
            return await _tool_ask(arguments)
        elif name == "index_status":
            return await _tool_index_status(arguments)
        elif name == "list_experiments":
            return await _tool_list_experiments(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        logger.error(f"Tool error: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def _tool_retrieve(args: Dict) -> list[TextContent]:
    """Retrieve relevant chunks."""
    query = args["query"]
    source_type = args.get("source_type", "all")
    top_k = args.get("top_k", 5)
    experiment = args.get("experiment", "01_format_comparison")
    
    logger.info(f"🔍 Retrieve: {query[:60]}... (top_k={top_k}, experiment={experiment})")
    
    # Get index
    index = _get_index(experiment)
    
    # Retrieve
    filter_source = None if source_type == "all" else source_type
    chunks = index.retrieve(query, top_k, filter_source_type=filter_source)
    
    # Format results
    result = {
        "query": query,
        "num_results": len(chunks),
        "chunks": [
            {
                "chunk_id": rc.chunk.id,
                "doc_id": rc.chunk.doc_id,
                "source_type": rc.chunk.source_type,
                "score": round(rc.score, 4),
                "content": rc.chunk.content[:500] + "..." if len(rc.chunk.content) > 500 else rc.chunk.content
            }
            for rc in chunks
        ]
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(result, indent=2)
    )]


async def _tool_ask(args: Dict) -> list[TextContent]:
    """Run full RAG pipeline."""
    question_text = args["question"]
    source_type = args.get("source_type", "all")
    pipeline_choice = args.get("pipeline", "auto")
    experiment = args.get("experiment", "01_format_comparison")
    
    logger.info(f"💬 Ask: {question_text[:60]}... (pipeline={pipeline_choice})")
    
    # Load config
    cfg = _load_config(experiment)
    
    # Change to rag-lab dir
    import os
    original_cwd = os.getcwd()
    os.chdir(str(_RAG_LAB_ROOT))
    
    try:
        # Get components
        index = _get_index(experiment)
        classifier = get_classifier(cfg.intent, cfg.llm)
        reranker = get_reranker(cfg.retrieve)
        
        # Create question
        question = Question(
            id="mcp_query",
            text=question_text,
            ground_truth="",
            source_type=source_type,
            category="mcp_query"
        )
        
        # Classify intent or use override
        if pipeline_choice == "auto":
            intent_result = classifier.classify(question_text)
            intent_label = intent_result.label
        else:
            intent_label = "simple" if pipeline_choice == "naive" else "complex"
        
        # Route to pipeline
        if intent_label == "simple":
            pipeline = NaiveRAGPipeline(index, reranker, cfg)
        else:
            pipeline = AgenticRAGPipeline(index, reranker, cfg)
        
        # Run
        result = pipeline.run(question)
        
        # Format response
        response = {
            "answer": result.predicted_answer,
            "pipeline": result.pipeline,
            "intent": intent_label,
            "num_chunks_retrieved": len(result.retrieved_chunks),
            "top_chunks": [
                {
                    "chunk_id": rc.chunk.id,
                    "source_type": rc.chunk.source_type,
                    "score": round(rc.score, 4),
                    "preview": rc.chunk.content[:200] + "..."
                }
                for rc in result.retrieved_chunks[:3]
            ]
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(response, indent=2)
        )]
    
    finally:
        os.chdir(original_cwd)


async def _tool_index_status(args: Dict) -> list[TextContent]:
    """Get index status."""
    experiment = args.get("experiment", "01_format_comparison")
    
    logger.info(f"📊 Index status for: {experiment}")
    
    try:
        cfg = _load_config(experiment)
        index = _get_index(experiment)
        
        # Check if built
        is_built = index.is_built(cfg.experiment.name) if hasattr(index, 'is_built') else True
        
        status = {
            "experiment": experiment,
            "backend": cfg.index.backend,
            "is_built": is_built,
            "persist_dir": cfg.index.persist_dir,
            "embedding_model": cfg.embed.model,
            "top_k": cfg.retrieve.top_k
        }
        
        # Try to get doc count if available
        if hasattr(index, '_collection') and index._collection:
            try:
                status["document_count"] = index._collection.count()
            except:
                pass
        
        return [TextContent(
            type="text",
            text=json.dumps(status, indent=2)
        )]
    
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e)}, indent=2)
        )]


async def _tool_list_experiments(args: Dict) -> list[TextContent]:
    """List all experiments."""
    logger.info("📁 Listing experiments")
    
    experiments = []
    
    for exp_dir in sorted(_EXPERIMENTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        config_path = exp_dir / "config.yaml"
        if not config_path.exists():
            continue
        
        # Check for results
        results_csv = _OUT_DIR / exp_dir.name / f"{exp_dir.name}_results.csv"
        has_results = results_csv.exists()
        
        experiments.append({
            "name": exp_dir.name,
            "path": str(exp_dir),
            "has_config": True,
            "has_results": has_results
        })
    
    return [TextContent(
        type="text",
        text=json.dumps({"experiments": experiments}, indent=2)
    )]


# ─── Main ──────────────────────────────────────────────────────────────────────

async def main():
    """Run MCP server via stdio."""
    logger.info("🚀 Starting RAG Playground MCP Server")
    logger.info(f"   RAG Lab Root: {_RAG_LAB_ROOT}")
    logger.info(f"   Experiments Dir: {_EXPERIMENTS_DIR}")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
