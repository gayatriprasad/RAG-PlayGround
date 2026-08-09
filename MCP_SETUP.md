# MCP Server Setup Guide

This guide explains how to set up and use the RAG Playground MCP (Model Context Protocol) server with Claude Desktop and other MCP-compatible clients.

## What is MCP?

Model Context Protocol (MCP) is an open standard that lets AI assistants connect to external tools and data sources. The RAG Playground MCP server exposes our RAG pipelines as tools that Claude Desktop (or other MCP clients) can use.

## Prerequisites

```bash
# Install MCP server dependencies
cd rag-lab
pip install -e ".[observability]"

# This installs: mcp>=0.9.0, langfuse>=2.0.0
```

## Available Tools

The MCP server exposes 4 tools:

### 1. `retrieve`
Retrieve relevant chunks from the enterprise corpus.

**Parameters:**
- `query` (required): Search query
- `source_type` (optional): Filter by source (confluence, github, jira, slack, etc.) Default: "all"
- `top_k` (optional): Number of chunks to retrieve. Default: 5
- `experiment` (optional): Experiment name. Default: "01_format_comparison"

**Example:**
```json
{
  "query": "How do we handle authentication in the API?",
  "source_type": "confluence",
  "top_k": 10
}
```

### 2. `ask`
Run full RAG pipeline and return answer with citations.

**Parameters:**
- `question` (required): Question to answer
- `source_type` (optional): Filter retrieval. Default: "all"
- `pipeline` (optional): Force pipeline ("auto", "naive", "agentic"). Default: "auto"
- `experiment` (optional): Experiment name. Default: "01_format_comparison"

**Example:**
```json
{
  "question": "What are the benefits of RAG over fine-tuning?",
  "pipeline": "auto"
}
```

### 3. `index_status`
Get index statistics and health status.

**Parameters:**
- `experiment` (optional): Experiment name. Default: "01_format_comparison"

**Example:**
```json
{
  "experiment": "01_format_comparison"
}
```

### 4. `list_experiments`
List all available experiments.

**Parameters:** None

---

## Claude Desktop Setup

### Step 1: Locate Config File

Claude Desktop config is at:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### Step 2: Add RAG Playground Server

Edit `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-playground": {
      "command": "python",
      "args": [
        "/absolute/path/to/RAG-PlayGround/api/mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "/absolute/path/to/RAG-PlayGround/rag-lab/src"
      }
    }
  }
}
```

**Important:** Replace `/absolute/path/to/RAG-PlayGround` with your actual path!

For this installation:
```json
{
  "mcpServers": {
    "rag-playground": {
      "command": "python",
      "args": [
        "/Users/saigayatriprasadperi/RAG-PlayGround/api/mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "/Users/saigayatriprasadperi/RAG-PlayGround/rag-lab/src"
      }
    }
  }
}
```

### Step 3: Restart Claude Desktop

Close and reopen Claude Desktop. The MCP server will start automatically.

### Step 4: Verify Connection

In Claude Desktop, look for the 🔌 icon in the bottom toolbar. Click it to see connected servers.

You should see:
- **rag-playground** (4 tools available)

### Step 5: Use Tools in Chat

Ask Claude to use the tools:

```
Can you retrieve documents about authentication from our Confluence?
```

Claude will automatically use the `retrieve` tool with appropriate parameters.

---

## Testing MCP Server Standalone

You can test the server directly:

```bash
cd /Users/saigayatriprasadperi/RAG-PlayGround

# Run MCP server (stdio mode)
python api/mcp_server.py
```

The server will start and wait for MCP client connections via stdin/stdout.

**Note:** This is for testing only. Claude Desktop handles server lifecycle automatically.

---

## Langfuse Integration (Optional)

For production observability, connect Langfuse to track all RAG queries:

### Step 1: Get Langfuse Keys

Sign up at [https://cloud.langfuse.com](https://cloud.langfuse.com) and get:
- Public Key (`pk-lf-...`)
- Secret Key (`sk-lf-...`)

### Step 2: Configure Environment

Add to your shell config (`~/.zshrc` or `~/.bashrc`):

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

Reload shell:
```bash
source ~/.zshrc
```

### Step 3: Update Claude Desktop Config

Add Langfuse keys to the MCP server config:

```json
{
  "mcpServers": {
    "rag-playground": {
      "command": "python",
      "args": [
        "/Users/saigayatriprasadperi/RAG-PlayGround/api/mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "/Users/saigayatriprasadperi/RAG-PlayGround/rag-lab/src",
        "LANGFUSE_PUBLIC_KEY": "pk-lf-...",
        "LANGFUSE_SECRET_KEY": "sk-lf-...",
        "LANGFUSE_HOST": "https://cloud.langfuse.com"
      }
    }
  }
}
```

### Step 4: View Traces

Go to [https://cloud.langfuse.com](https://cloud.langfuse.com) → Traces

You'll see:
- Every RAG query with full execution timeline
- Intent classification → Retrieval → Reranking → Generation
- Latency breakdown per step
- Scores and metadata

---

## Programmatic Usage (Python)

You can also use the observability module directly in your code:

```python
from raglab.observability import PipelineTracer

# Auto-detects Langfuse or falls back to JSONL
with PipelineTracer(experiment_name, question.text) as tracer:
    # Trace retrieval
    with tracer.span("retrieval", query=query, top_k=5):
        chunks = index.retrieve(query, top_k=5)
    
    # Trace generation
    with tracer.span("generation", model=cfg.llm.model):
        answer = llm_generate(chunks, question)
    
    # Add scores
    tracer.score("overall_score", result.overall_score)
    tracer.score("answer_correct", 1.0 if result.answer_correct else 0.0)
```

Traces are automatically sent to Langfuse (if configured) or written to `./out/raglab_out/traces/{experiment}_traces.jsonl`.

---

## Troubleshooting

### MCP Server Not Showing in Claude Desktop

1. **Check config path**: Ensure you're editing the correct config file
2. **Verify JSON syntax**: Use [jsonlint.com](https://jsonlint.com) to validate
3. **Check Python path**: Run `which python` to confirm interpreter path
4. **View logs**: Check Claude Desktop logs (Help → View Logs)

### Tools Not Working

1. **Check experiment exists**: Run `python api/mcp_server.py` and call `list_experiments`
2. **Verify index is built**: Ensure you've run an experiment first
3. **Check dependencies**: `pip list | grep mcp` should show `mcp`

### Langfuse Not Receiving Traces

1. **Check keys**: Verify `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` are set
2. **Test connection**:
   ```python
   from langfuse import Langfuse
   client = Langfuse()  # Should not error
   ```
3. **Check firewall**: Ensure outbound HTTPS to cloud.langfuse.com is allowed
4. **View fallback logs**: Traces are written to JSONL as backup

---

## Example Workflows

### Workflow 1: Research Assistant

**User:** "Can you find all documentation about our authentication system?"

**Claude uses:**
1. `retrieve(query="authentication system", source_type="confluence", top_k=10)`
2. Summarizes top documents
3. `ask(question="What authentication methods do we support?")`
4. Returns comprehensive answer with citations

### Workflow 2: Code Search

**User:** "Show me how we handle rate limiting in the API"

**Claude uses:**
1. `retrieve(query="rate limiting API", source_type="github", top_k=5)`
2. Returns relevant code snippets
3. Explains implementation details

### Workflow 3: Compare Implementations

**User:** "Compare how we do authentication in API v1 vs v2"

**Claude uses:**
1. `ask(question="Compare authentication in API v1 vs v2", pipeline="agentic")`
2. Agentic pipeline decomposes into sub-queries
3. Returns detailed comparison

---

## Best Practices

1. **Experiment Management**: Keep experiments organized by date or feature
2. **Source Type Filters**: Use specific source_type for faster, more relevant results
3. **Pipeline Selection**: Use "auto" for most queries, "agentic" for complex multi-hop questions
4. **Langfuse Monitoring**: Review traces weekly to identify slow queries and improve retrieval

---

## Next Steps

- **SKILL 19**: Add vector store management tools (rebuild index, update docs)
- **SKILL 20**: Implement feedback loop (thumbs up/down on answers)
- **SKILL 21**: Add multi-user session management

---

## Resources

- [MCP Specification](https://modelcontextprotocol.io)
- [Langfuse Documentation](https://langfuse.com/docs)
- [Claude Desktop MCP Guide](https://docs.anthropic.com/claude/docs/mcp)
