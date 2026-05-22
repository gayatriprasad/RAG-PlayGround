"""
run_experiment.py — SKILL 11 Orchestrator

CLI: python -m raglab.run_experiment --config PATH [--download-data]

Full pipeline:
  1. Load config.yaml → Config
  2. If --download-data: call download_bench_slice()
  3. load_documents() and load_questions() via enterprise_bench parser
  4. Chunk documents via get_chunker(cfg.chunk)
  5. Build or load index via get_index(cfg.index)
  6. Init classifier and reranker
  7. For each question: classify → route → pipeline → EvalResult
  8. Score all results via BenchmarkScorer
  9. Save CSV + print summary via ExperimentReporter
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import typer
import yaml
from tqdm import tqdm

from raglab.chunkers import get_chunker
from raglab.classifiers import get_classifier
from raglab.config import Config
from raglab.eval import BenchmarkScorer, ExperimentReporter
from raglab.index import get_index
from raglab.parsers.enterprise_bench import (
    download_bench_slice,
    load_documents,
    load_questions,
)
from raglab.parsers.normalizer import DocumentNormalizer
from raglab.pipelines import AgenticRAGPipeline, NaiveRAGPipeline
from raglab.rerankers import get_reranker
from raglab.types import Chunk, Document, EvalResult
from raglab.utils.tracer import save_traces

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def _load_config(config_path: str) -> Config:
    """Load and validate config from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise typer.BadParameter(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(**raw)


def _documents_to_chunks(documents: list[Document], cfg: Config) -> list[Chunk]:
    """Chunk all documents using configured chunker."""
    chunker = get_chunker(cfg.chunk)
    logger.info(f"Chunking {len(documents)} documents with strategy='{cfg.chunk.strategy}'")

    all_chunks: list[Chunk] = []
    for doc in tqdm(documents, desc="Chunking"):
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)

    logger.info(f"Produced {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks


@app.command()
def main(
    config: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to experiment config.yaml",
    ),
    download_data: bool = typer.Option(
        False,
        "--download-data",
        help="Download EnterpriseRAG-Bench slice before running",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable DEBUG logging",
    ),
) -> None:
    """Run a RAG experiment end-to-end."""

    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Step 1: Load config
    logger.info(f"Loading config from: {config}")
    cfg = _load_config(config)
    logger.info(f"Experiment: {cfg.experiment.name}")

    # Step 2: Optional data download
    if download_data:
        logger.info("Downloading EnterpriseRAG-Bench slice...")
        download_bench_slice(
            source_types=cfg.benchmark.source_types,
            out_dir="corpus/raw/",
        )

    # Step 3: Load documents and questions
    logger.info("Loading documents and questions...")
    documents = load_documents(cfg.benchmark)
    questions = load_questions(cfg.benchmark)
    logger.info(f"Loaded {len(documents)} documents, {len(questions)} questions")

    if not questions:
        logger.error("No questions loaded. Check benchmark config and data files.")
        raise typer.Exit(code=1)

    # Step 3b: Normalize and deduplicate documents
    normalizer = DocumentNormalizer()
    documents = normalizer.normalize(documents)
    documents = normalizer.deduplicate(documents)

    # Step 4: Chunk documents
    all_chunks = _documents_to_chunks(documents, cfg)

    # Step 5: Build or load index
    logger.info(f"Initializing index: backend='{cfg.index.backend}'")
    index = get_index(cfg.index, cfg.embed)

    experiment_name = cfg.experiment.name
    if hasattr(index, "is_built") and index.is_built(experiment_name):
        logger.info("Index already built, skipping rebuild.")
    else:
        logger.info(f"Building index with {len(all_chunks)} chunks...")
        # Pass experiment_name if the index supports it
        import inspect
        build_params = inspect.signature(index.build).parameters
        if "experiment_name" in build_params:
            index.build(all_chunks, experiment_name=experiment_name)
        else:
            index.build(all_chunks)
        logger.info("Index build complete.")

    # Step 6: Init classifier and reranker
    classifier = get_classifier(cfg.intent, cfg.llm)
    reranker = get_reranker(cfg.retrieve)

    # Init pipelines
    naive_pipeline = NaiveRAGPipeline(index=index, reranker=reranker, cfg=cfg)
    agentic_pipeline = AgenticRAGPipeline(index=index, reranker=reranker, cfg=cfg)

    # Step 7: Run pipeline for each question
    logger.info(f"Running pipeline on {len(questions)} questions...")
    results: list[EvalResult] = []

    for question in tqdm(questions, desc="Processing questions"):
        t_start = time.perf_counter()

        # Classify intent
        intent = classifier.classify(question.text)

        # Route to pipeline
        if intent.label == "simple":
            result = naive_pipeline.run(question)
        else:
            result = agentic_pipeline.run(question)

        # Set intent_label on result
        result.intent_label = intent.label

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.debug(
            f"Q={question.id}: intent={intent.label}, "
            f"pipeline={result.pipeline}, time={elapsed_ms:.0f}ms"
        )

        results.append(result)

    logger.info(f"Pipeline complete. {len(results)} results collected.")

    # Step 8: Score results
    logger.info("Scoring results...")
    scorer = BenchmarkScorer(cfg.eval, cfg.llm)
    scored_results = scorer.score(results)

    # Step 9: Report
    out_dir = str(Path("out/raglab_out") / cfg.experiment.name)
    df = scorer.to_dataframe(scored_results)

    reporter = ExperimentReporter()
    csv_path = reporter.save_csv(df, out_dir, cfg.experiment.name)
    reporter.print_summary(df)
    reporter.save_markdown_report(
        df,
        out_dir,
        cfg.experiment.name,
        config_snapshot=cfg.model_dump(),
    )

    # Step 10: Save traces
    traces = [r.metadata.get("trace") for r in scored_results if r.metadata.get("trace")]
    if traces:
        save_traces(traces, out_dir, cfg.experiment.name)

    logger.info(f"Experiment '{cfg.experiment.name}' complete. Results: {csv_path}")


if __name__ == "__main__":
    app()
