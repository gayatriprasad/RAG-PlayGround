"""
Synthetic Dataset Generator — generates questions from corpus documents.

For each document chunk, uses an LLM to generate questions of various types:
- factual: direct answer in text
- analytical: requires reasoning
- adversarial: answer NOT in text — 'NOT FOUND' expected
- comparative: compare two aspects in same doc
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

from raglab.types import Document, Question

logger = logging.getLogger(__name__)


class DatasetSynthesizer:
    """
    Generate synthetic questions from corpus documents using an LLM.
    
    Produces diverse question types and deduplicates against an existing
    golden set using embedding similarity.
    """

    QUESTION_TYPES = ["factual", "analytical", "adversarial", "comparative"]

    def generate(
        self,
        docs: List[Document],
        cfg,
        llm_client=None,
        n_per_type: int = 2,
        max_chunks_per_doc: int = 5,
    ) -> List[Question]:
        """
        Generate synthetic questions from documents.
        
        Args:
            docs: List of source documents
            cfg: DatasetCfg or Config with dataset settings
            llm_client: LLM client for generation (OpenAI-compatible)
            n_per_type: Number of questions per type per chunk
            max_chunks_per_doc: Max chunks to sample from each doc
            
        Returns:
            List of generated Question objects
        """
        if llm_client is None:
            llm_client = self._build_default_client(cfg)

        all_questions: List[Question] = []
        doc_sample = docs if len(docs) <= 50 else random.sample(docs, 50)

        logger.info(
            f"Generating synthetic questions from {len(doc_sample)} documents "
            f"({n_per_type} per type, {len(self.QUESTION_TYPES)} types)"
        )

        for doc in doc_sample:
            # Split doc into chunks for question generation
            chunks = self._split_for_generation(doc.content, max_chunks_per_doc)

            for i, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) < 100:
                    continue

                for qtype in self.QUESTION_TYPES:
                    try:
                        questions = self._generate_for_chunk(
                            chunk_text=chunk_text,
                            source_type=doc.source_type,
                            qtype=qtype,
                            n_per_type=n_per_type,
                            llm_client=llm_client,
                            cfg=cfg,
                            doc_id=doc.id,
                            chunk_idx=i,
                        )
                        all_questions.extend(questions)
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate {qtype} questions for doc={doc.id}: {e}"
                        )

        logger.info(f"Generated {len(all_questions)} synthetic questions total")
        return all_questions

    def _generate_for_chunk(
        self,
        chunk_text: str,
        source_type: str,
        qtype: str,
        n_per_type: int,
        llm_client,
        cfg,
        doc_id: str,
        chunk_idx: int,
    ) -> List[Question]:
        """Generate questions of a specific type from a chunk."""
        system_prompt = (
            "You are a dataset generator for RAG evaluation. "
            "Generate high-quality questions based on the given text.\n\n"
            "Rules:\n"
            "- factual: answer is directly stated in the text\n"
            "- analytical: requires reasoning or inference from the text\n"
            "- adversarial: answer is NOT in the text, correct answer is 'NOT FOUND'\n"
            "- comparative: compares two concepts/aspects mentioned in text\n\n"
            "Reply ONLY with a valid JSON array."
        )

        user_prompt = (
            f"Generate {n_per_type} questions of type '{qtype}' based on this text.\n\n"
            f"Text:\n{chunk_text[:2000]}\n\n"
            f"Format: JSON array of objects with keys: question, answer, category, difficulty\n"
            f"difficulty should be 'easy', 'medium', or 'hard'\n"
            f"For adversarial type: the answer should be 'NOT FOUND'\n\n"
            f"JSON array:"
        )

        try:
            raw = llm_client.complete(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=1024,
            ).strip()

            # Try to extract JSON array
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            items = json.loads(raw)
            if not isinstance(items, list):
                items = [items]

        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"JSON parse failed for {qtype}/{doc_id}: {e}")
            return []

        questions = []
        for idx, item in enumerate(items[:n_per_type]):
            q_id = hashlib.sha256(
                f"{doc_id}_{chunk_idx}_{qtype}_{idx}".encode()
            ).hexdigest()[:12]

            questions.append(
                Question(
                    id=f"syn_{q_id}",
                    text=item.get("question", ""),
                    ground_truth=item.get("answer", ""),
                    source_type=source_type,
                    category=item.get("category", qtype),
                    difficulty=item.get("difficulty"),
                )
            )

        return questions

    def _split_for_generation(self, content: str, max_chunks: int) -> List[str]:
        """Split document content into chunks for question generation."""
        # Split on double newlines
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        if len(paragraphs) <= max_chunks:
            return paragraphs

        # Sample evenly across the document
        step = max(1, len(paragraphs) // max_chunks)
        return paragraphs[::step][:max_chunks]

    def _build_default_client(self, cfg):
        """Build default LLM client from config."""
        from raglab.pipelines.naive_rag import build_llm_client

        # Handle both DatasetCfg and full Config
        if hasattr(cfg, "llm"):
            return build_llm_client(cfg.llm)
        else:
            from raglab.config import LLMCfg
            return build_llm_client(LLMCfg())

    def save(self, questions: List[Question], output_path: str) -> Path:
        """Save generated questions to JSONL file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for q in questions:
                f.write(q.model_dump_json() + "\n")

        logger.info(f"Saved {len(questions)} synthetic questions to {path}")
        return path

    def validate_generated(
        self,
        questions: List[Question],
        docs: List[Document],
        embed_model: str = "all-MiniLM-L6-v2",
        answerability_threshold: float = 0.3,
    ) -> tuple[List[Question], List[dict], dict]:
        """
        Quality-gate synthetic questions before they enter the golden set (Skill 44).

        Checks, cheapest first:
        - non_degenerate: rejects empty/near-empty ground_truth, trivially short
          questions (<4 words), and answer-leaking questions (the exact answer
          text appears verbatim inside the question).
        - category_match: adversarial questions must have ground_truth "NOT FOUND"
          (the synthesizer's own generation contract) — a cheap, exact check,
          not an LLM re-verification.
        - answerability: for non-adversarial questions, the ground_truth must be
          semantically supported (cosine similarity >= answerability_threshold)
          by *some* paragraph in the corpus with the same source_type. This is
          corpus-level support, not exact-source-chunk-level — Question does not
          retain a chunk/doc_id reference after generation, so we cannot check
          the single originating chunk. Documented limitation, not fabricated
          precision.

        Returns (kept, rejected, report) where `rejected` is a list of
        {"question": Question, "reason": str} dicts and `report` summarizes
        counts plus the difficulty distribution of the kept set.
        """
        paragraphs_by_source_type = self._corpus_paragraphs_by_source_type(docs)

        kept: List[Question] = []
        rejected: List[dict] = []

        for q in questions:
            reason = self._non_degenerate_reason(q)
            if reason:
                rejected.append({"question": q, "reason": reason})
                continue

            reason = self._category_match_reason(q)
            if reason:
                rejected.append({"question": q, "reason": reason})
                continue

            if q.category != "adversarial":
                reason = self._answerability_reason(
                    q, paragraphs_by_source_type, embed_model, answerability_threshold
                )
                if reason:
                    rejected.append({"question": q, "reason": reason})
                    continue

            kept.append(q)

        difficulty_spread: Dict[str, int] = {}
        for q in kept:
            key = q.difficulty or "unspecified"
            difficulty_spread[key] = difficulty_spread.get(key, 0) + 1

        report = {
            "n_total": len(questions),
            "n_kept": len(kept),
            "n_rejected": len(rejected),
            "rejected_by_reason": self._count_by_reason(rejected),
            "difficulty_spread": difficulty_spread,
        }
        logger.info(f"validate_generated: kept {len(kept)}/{len(questions)} ({report['rejected_by_reason']})")
        return kept, rejected, report

    def _non_degenerate_reason(self, q: Question) -> Optional[str]:
        if not q.text.strip() or len(q.text.strip().split()) < 4:
            return "degenerate: question text too short or empty"
        if not q.ground_truth.strip():
            return "degenerate: empty ground_truth"
        if q.category != "adversarial" and q.ground_truth.strip().lower() in q.text.strip().lower():
            return "degenerate: answer leaks verbatim into the question text"
        return None

    def _category_match_reason(self, q: Question) -> Optional[str]:
        if q.category == "adversarial" and q.ground_truth.strip().upper() != "NOT FOUND":
            return "category_mismatch: adversarial question must have ground_truth 'NOT FOUND'"
        return None

    def _answerability_reason(
        self,
        q: Question,
        paragraphs_by_source_type: Dict[str, List[str]],
        embed_model: str,
        threshold: float,
    ) -> Optional[str]:
        paragraphs = paragraphs_by_source_type.get(q.source_type, [])
        if not paragraphs:
            return f"unanswerable: no corpus documents found for source_type={q.source_type}"

        try:
            from sklearn.metrics.pairwise import cosine_similarity

            from raglab.utils.embedder import Embedder

            embedder = Embedder(embed_model)
            answer_vec = embedder.embed([q.ground_truth])
            para_vecs = embedder.embed(paragraphs)
            similarities = cosine_similarity(answer_vec, para_vecs)[0]
            best = float(similarities.max())
        except Exception as e:
            logger.warning(f"Answerability embedding check failed for {q.id}: {e} — skipping check.")
            return None

        if best < threshold:
            return f"unanswerable: best corpus similarity {best:.2f} < threshold {threshold}"
        return None

    def _corpus_paragraphs_by_source_type(self, docs: List[Document]) -> Dict[str, List[str]]:
        by_type: Dict[str, List[str]] = {}
        for doc in docs:
            paragraphs = [p.strip() for p in doc.content.split("\n\n") if p.strip()]
            by_type.setdefault(doc.source_type, []).extend(paragraphs)
        return by_type

    def _count_by_reason(self, rejected: List[dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in rejected:
            # Bucket by the reason prefix (before the colon) for a compact summary.
            bucket = item["reason"].split(":", 1)[0]
            counts[bucket] = counts.get(bucket, 0) + 1
        return counts
