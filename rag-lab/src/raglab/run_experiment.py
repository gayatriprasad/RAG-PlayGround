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
from typing import List, Optional

import typer
import yaml
from tqdm import tqdm

from raglab.chunkers import get_chunker
from raglab.classifiers import get_classifier
from raglab.config import Config
from raglab.eval import BenchmarkScorer, ExperimentReporter
from raglab.eval.scorer import check_run_completeness
from raglab.hooks import get_default_registry
from raglab.index import get_index
from raglab.parsers.enterprise_bench import (
    download_bench_slice,
    load_documents,
    load_questions,
)
from raglab.parsers.normalizer import DocumentNormalizer
from raglab.pipelines import AgenticRAGPipeline, NaiveRAGPipeline
from raglab.rerankers import get_reranker
from raglab.types import Chunk, Document, EvalResult, Question
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


def _load_corpus_and_questions(cfg: Config):
    """Load documents + questions per Skill 33's CorpusCfg.source setting.

    - "bench": existing EnterpriseRAG-Bench loader (unchanged core path).
    - "upload": parse every file under cfg.corpus.upload_dir via UploadParser.
    - "mixed": bench + upload merged.

    If cfg.corpus.user_questions_path is set, those questions are used
    instead of (or alongside, for "mixed") the bench golden set.
    """
    from raglab.parsers.upload_parser import UploadParser, load_user_questions

    documents: List[Document] = []
    questions: List[Question] = []

    if cfg.corpus.source in ("bench", "mixed"):
        documents.extend(load_documents(cfg.benchmark))
        questions.extend(load_questions(cfg.benchmark))

    if cfg.corpus.source in ("upload", "mixed"):
        upload_docs = UploadParser().parse_directory(cfg.corpus.upload_dir, cfg.corpus)
        logger.info(f"Parsed {len(upload_docs)} documents from {cfg.corpus.upload_dir}")
        documents.extend(upload_docs)

    if cfg.corpus.user_questions_path:
        user_questions = load_user_questions(cfg.corpus.user_questions_path)
        if cfg.corpus.source == "upload":
            questions = user_questions
        else:
            questions.extend(user_questions)

    return documents, questions


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
    logger.info(f"Loading documents and questions (corpus.source='{cfg.corpus.source}')...")
    documents, questions = _load_corpus_and_questions(cfg)
    logger.info(f"Loaded {len(documents)} documents, {len(questions)} questions")

    if not questions:
        logger.error("No questions loaded. Check benchmark config and data files.")
        raise typer.Exit(code=1)

    # Step 3b: Normalize and deduplicate documents
    normalizer = DocumentNormalizer()
    documents = normalizer.normalize(documents)
    documents = normalizer.deduplicate(documents)

    # Step 3c: Run pre_experiment hooks
    hooks = get_default_registry()
    hooks.run_pre_experiment(cfg, documents, questions)

    # Step 4: Chunk documents
    all_chunks = _documents_to_chunks(documents, cfg)
    if len(all_chunks) == 0:
        # Rule 32 / Failure Mode Register: "Chunker produces 0 chunks: document
        # silently drops from index." Fail loudly instead of building an empty index.
        logger.error(
            f"Chunking produced 0 chunks from {len(documents)} documents "
            f"(strategy='{cfg.chunk.strategy}'). Refusing to build an empty index."
        )
        raise typer.Exit(code=1)

    # Step 5: Build or load index
    logger.info(f"Initializing index: backend='{cfg.index.backend}'")
    index = get_index(cfg.index, cfg.embed)

    experiment_name = cfg.experiment.name

    # Skill 50B: pass corpus_hash to is_built() when supported, so a stale
    # index (chunk count unchanged, content changed) is detected and rebuilt.
    import inspect

    is_built_result = False
    if hasattr(index, "is_built"):
        is_built_params = inspect.signature(index.is_built).parameters
        if "corpus_hash" in is_built_params and hasattr(index, "_corpus_hash"):
            is_built_result = index.is_built(
                experiment_name,
                expected_count=len(all_chunks),
                corpus_hash=index._corpus_hash(all_chunks),
            )
        else:
            is_built_result = index.is_built(experiment_name)

    if is_built_result:
        logger.info("Index already built, skipping rebuild.")
    else:
        logger.info(f"Building index with {len(all_chunks)} chunks...")
        # Pass experiment_name if the index supports it
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

    # Skill 50A — resumable runs: derive a deterministic run_id from
    # (experiment, config_hash) so re-invoking with the same config resumes
    # an interrupted run instead of starting over and re-paying for already-
    # scored questions. Each result is written to the DB immediately after
    # scoring (Rule 31), not batched at the end, so a crash loses at most
    # the in-flight question.
    db_writer = None
    run_id = None
    completed_question_ids: set = set()
    scorer = BenchmarkScorer(cfg.eval, cfg.llm)
    try:
        import hashlib

        from raglab.db.writer import DBWriter

        db_writer = DBWriter(cfg.db)
        db_writer.ensure_schema()
        config_hash = hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()[:16]
        experiment_id = db_writer.upsert_experiment(cfg.experiment.name, config_hash)
        run_id = hashlib.sha256(f"{experiment_id}:{config_hash}".encode()).hexdigest()[:32]
        run_id = db_writer.start_run(experiment_id, config_hash, run_id=run_id)
        db_writer.upsert_questions(questions)
        completed_question_ids = db_writer.get_completed_question_ids(run_id)
        if completed_question_ids:
            logger.info(
                f"Resuming run {run_id}: {len(completed_question_ids)} questions "
                f"already scored, skipping them."
            )
    except Exception as e:
        logger.warning(f"DB resume/init skipped (non-fatal, results still computed in-memory): {e}")

    results: list[EvalResult] = []

    for question in tqdm(questions, desc="Processing questions"):
        if question.id in completed_question_ids:
            continue

        t_start = time.perf_counter()

        # Classify intent
        intent = classifier.classify(question.text)

        hooks.run_pre_generation(question, cfg)

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

        hooks.run_post_generation(question, result, elapsed_ms, cfg)

        # Score this single result now and persist immediately (Rule 31 —
        # resumable, per-question writes instead of batch-at-end).
        scored_result = scorer.score([result])[0]
        if db_writer is not None and run_id is not None:
            try:
                db_writer.write_single_result(run_id, scored_result)
            except Exception as e:
                logger.warning(f"Failed to persist result for {question.id} (non-fatal): {e}")

        results.append(scored_result)

    logger.info(f"Pipeline complete. {len(results)} results collected (this session).")

    scored_results = results

    # Step 9: Run post_experiment hooks
    hooks.run_post_experiment(cfg, scored_results)

    # Step 9b: Finalize the DB run (results were already written per-question above)
    if db_writer is not None and run_id is not None:
        try:
            cost_hooks_for_db = [h for h in hooks.post_generation if hasattr(h, "tracker")]
            if cost_hooks_for_db:
                db_writer.write_costs(run_id, cost_hooks_for_db[0].tracker._records)

            # Skill 50G / Rule 32 — never mark a partial run 'completed'.
            n_scored = len(scored_results) + len(completed_question_ids)
            is_complete = check_run_completeness(n_scored, len(questions))
            db_writer.finish_run(run_id, "completed" if is_complete else "partial")
            logger.info(
                f"Persisted run '{run_id}' to database (backend={cfg.db.backend}, "
                f"status={'completed' if is_complete else 'partial'})."
            )
        except Exception as e:
            logger.warning(f"DB finalize skipped/failed (non-fatal): {e}")

    # Step 10: Report
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

    # Save cost summary (Skill 27) alongside results, if tracking is enabled
    cost_hooks = [h for h in hooks.post_generation if hasattr(h, "tracker")]
    if cost_hooks:
        import json

        summary = cost_hooks[0].tracker.summary()
        cost_path = Path(out_dir) / f"{cfg.experiment.name}_cost_summary.json"
        cost_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cost_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Cost summary saved to {cost_path}")

    logger.info(f"Experiment '{cfg.experiment.name}' complete. Results: {csv_path}")


if __name__ == "__main__":
    app()
