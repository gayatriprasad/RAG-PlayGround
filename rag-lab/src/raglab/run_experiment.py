from __future__ import annotations
import glob, json, os
from pathlib import Path
import yaml
from tqdm import tqdm

from raglab.config import Config
from raglab.parsers.pdf_text import PdfTextParser
from raglab.parsers.pdf_layout_html import PdfHtmlParser
from raglab.parsers.dom_builder import PdfDomParser
from raglab.parsers.docx_text import DocxTextParser
from raglab.parsers.markdown_builder import MarkdownFromTextParser
from raglab.chunkers.fixed import fixed_chunk
from raglab.index.brute_dense import BruteDenseIndex
from raglab.eval.metrics import recall_at_k, mrr
from raglab.utils.timing import timed
from raglab.utils.memory import peak_rss_mb

def load_golden(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def parse_one(path: str, representation: str):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        if representation == "text":
            return PdfTextParser().parse(path)
        if representation == "html":
            return PdfHtmlParser().parse(path)
        if representation == "dom":
            return PdfDomParser().parse(path)
        if representation == "markdown":
            return MarkdownFromTextParser(PdfTextParser()).parse(path)
        raise ValueError(f"Unknown representation: {representation}")
    elif ext == ".docx":
        if representation == "text":
            return DocxTextParser().parse(path)
        if representation == "markdown":
            return MarkdownFromTextParser(DocxTextParser()).parse(path)
        # For week-1, keep docx in text/markdown only
        raise ValueError("For .docx, use representations: text|markdown in week-1")
    else:
        raise ValueError(f"Unsupported file: {path}")

def main():
    cfg_path = os.environ.get("RAGLAB_CONFIG", "experiments/01_format_comparison/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = Config(**yaml.safe_load(f))

    # Collect files
    files = []
    for g in cfg.experiment.corpus_glob:
        files.extend(glob.glob(g))
    files = sorted(set(files))
    if not files:
        raise SystemExit("No files found. Put PDFs in corpus/raw/pdf and DOCX in corpus/raw/docx")

    golden = load_golden(cfg.golden.path)

    out_dir = Path("experiments") / cfg.experiment.name / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for rep in cfg.experiment.representations:
        rep_metrics = {
            "representation": rep,
            "docs": len(files),
            "chunks_total": 0,
            "chunk_size_mean_words": None,
            "chunk_size_std_words": None,
        }

        timings = []
        chunk_sizes = []
        all_chunks = []

        with peak_rss_mb(rep_metrics, "peak_rss_mb"):
            for path in tqdm(files, desc=f"Parsing+chunking [{rep}]"):
                t = {}
                with timed("parse_ms", t):
                    try:
                        d = parse_one(path, rep)
                    except ValueError:
                        # skip unsupported combinations (e.g., docx->html/dom)
                        continue

                with timed("chunk_ms", t):
                    chunks = fixed_chunk(d, cfg.chunk.chunk_tokens, cfg.chunk.overlap)

                timings.append(t)
                all_chunks.extend(chunks)
                rep_metrics["chunks_total"] += len(chunks)
                chunk_sizes.extend([len(c.text.split()) for c in chunks])

        # Build index
        idx = BruteDenseIndex()
        idx.build(all_chunks)

        # Eval
        r5s = []
        mrrs = []
        for ex in golden:
            q = ex["query"]
            target_prefix = ex["relevant_chunk_prefix"]  # e.g., "ab12cd34ef56:markdown:"
            hits, _scores = idx.search(q, top_k=cfg.retrieve.top_k)
            retrieved_ids = [h.chunk_id for h in hits]
            # relevant set = any chunk id that startswith prefix
            relevant_ids = [c.chunk_id for c in all_chunks if c.chunk_id.startswith(target_prefix)]
            r5s.append(recall_at_k(relevant_ids, retrieved_ids, k=cfg.retrieve.top_k))
            mrrs.append(mrr(relevant_ids, retrieved_ids))

        # Aggregate
        rep_metrics["recall@5"] = float(sum(r5s) / max(1, len(r5s)))
        rep_metrics["mrr"] = float(sum(mrrs) / max(1, len(mrrs)))

        rep_metrics["parse_ms_p50"] = percentile([x["parse_ms"] for x in timings], 50)
        rep_metrics["parse_ms_p95"] = percentile([x["parse_ms"] for x in timings], 95)
        rep_metrics["chunk_ms_p50"] = percentile([x["chunk_ms"] for x in timings], 50)
        rep_metrics["chunk_ms_p95"] = percentile([x["chunk_ms"] for x in timings], 95)

        rep_metrics["chunk_size_mean_words"] = float(sum(chunk_sizes) / max(1, len(chunk_sizes)))
        rep_metrics["chunk_size_std_words"] = stddev(chunk_sizes)

        results.append(rep_metrics)

        with open(out_dir / f"metrics_{rep}.json", "w", encoding="utf-8") as f:
            json.dump(rep_metrics, f, indent=2)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    print("\n=== Summary ===")
    for r in results:
        print(r)

def percentile(xs, p: int):
    if not xs:
        return None
    xs = sorted(xs)
    k = int(round((p/100) * (len(xs)-1)))
    return float(xs[k])

def stddev(xs):
    if not xs:
        return None
    mu = sum(xs)/len(xs)
    var = sum((x-mu)**2 for x in xs)/len(xs)
    return float(var**0.5)

if __name__ == "__main__":
    main()
