"""
RLM — Recursive/Retrieval Language Model pipeline — Skill 54.

For corpora too large to fit in any context window (and too large to
usefully retrieve top-k chunks from), the root LLM writes short Python
programs that operate directly over the raw document set (grep-like
filtering, slicing, counting) instead of retrieving embeddings. Those
programs run in a RestrictedPython sandbox — never raw `exec()` — because
this is model-generated code executing against user corpus data.

4-step loop per question:
  1. Root query planning — root LLM writes Python code against a serialized
     corpus preview to select/filter relevant slices.
  2. Safe code execution — the generated code runs sandboxed via
     RestrictedPython; on failure, the root LLM is asked to rewrite the code
     (up to max_code_rewrites attempts).
  3. Sub-model delegation — each slice is handed to a cheap sub-model
     (local Ollama by default) to extract question-relevant facts.
  4. Root aggregation — the root LLM combines sub-model outputs into the
     final answer.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, List, Optional

from raglab.config import Config
from raglab.types import ConfigError, Document, EvalResult, Question

logger = logging.getLogger(__name__)


class RLMPipeline:
    """Recursive Language Model pipeline over raw documents (no vector index)."""

    def __init__(self, documents: List[Document], cfg: Config):
        from raglab.models import get_llm

        self.documents = documents
        self.cfg = cfg
        self.rlm_cfg = cfg.rlm
        self.root_client = get_llm(cfg.llm)
        self.sub_client = self._build_sub_client()

    def _build_sub_client(self):
        """Build the cheap sub-model client used for per-slice delegation."""
        from raglab.config import ModelRegistryCfg
        from raglab.models import get_llm

        sub_cfg = ModelRegistryCfg(provider=self.rlm_cfg.sub_provider, model=self.rlm_cfg.sub_model)
        return get_llm(sub_cfg)

    def _serialize_corpus(self) -> str:
        """Build a compact preview of the corpus for the root LLM's code-
        generation prompt — full document IDs and a truncated content
        preview, never the full corpus (which is exactly what RLM avoids
        loading into context)."""
        lines = []
        for doc in self.documents:
            preview = doc.content[: self.rlm_cfg.corpus_preview_chars]
            lines.append(f"id={doc.id!r} len={len(doc.content)} preview={preview!r}")
        return "\n".join(lines)

    def run(self, question: Question) -> EvalResult:
        t_start = time.perf_counter()
        corpus_preview = self._serialize_corpus()

        code = self._generate_code(question.text, corpus_preview)
        slices: List[str] = []
        rewrite_attempts = 0
        last_error: Optional[str] = None

        while rewrite_attempts <= self.rlm_cfg.max_code_rewrites:
            try:
                slices = self._execute_safe(code, self.documents)
                break
            except Exception as e:
                last_error = str(e)
                logger.warning(f"RLM generated code failed (attempt {rewrite_attempts + 1}): {e}")
                rewrite_attempts += 1
                if rewrite_attempts > self.rlm_cfg.max_code_rewrites:
                    logger.error("RLM code execution failed after max rewrite attempts; no slices selected")
                    slices = []
                    break
                code = self._generate_code(
                    question.text, corpus_preview, previous_error=last_error, previous_code=code
                )

        slices = slices[: self.rlm_cfg.max_iterations]
        sub_answers = [self._delegate_to_sub_model(question.text, s) for s in slices]
        final_answer = self._aggregate(question.text, sub_answers)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        return EvalResult(
            question_id=question.id,
            question=question.text,
            ground_truth=question.ground_truth,
            predicted_answer=final_answer,
            source_type=question.source_type,
            category=question.category,
            index_backend="none",
            pipeline="rlm",
            intent_label="n/a",
            retrieved_chunks=[],
            metadata={
                "n_slices": len(slices),
                "code_rewrite_attempts": rewrite_attempts,
                "latency_ms": elapsed_ms,
            },
        )

    def _generate_code(
        self,
        question: str,
        corpus_preview: str,
        previous_error: Optional[str] = None,
        previous_code: Optional[str] = None,
    ) -> str:
        """Ask the root LLM to write Python code that filters/selects
        relevant slices from `documents` into a `result` list of strings."""
        instructions = (
            "Write a short Python program operating on a variable `documents` "
            "(a list of objects with .id and .content attributes). "
            "Select the pieces of content relevant to the question and assign "
            f"a list of strings (each <= {self.rlm_cfg.max_tokens_per_slice} "
            "characters) to a variable named `result`. "
            "Do not use import, open, exec, eval, or any network/file access. "
            "Return ONLY the code in a ```python fenced block."
        )
        prompt = f"{instructions}\n\nCorpus preview:\n{corpus_preview}\n\nQuestion: {question}"
        if previous_error:
            prompt += (
                f"\n\nThe previous code failed with error: {previous_error}\n"
                f"Previous code:\n{previous_code}\nPlease fix it."
            )

        messages = [
            {"role": "system", "content": "You are a careful Python code generator for a sandboxed corpus-analysis tool."},
            {"role": "user", "content": prompt},
        ]
        response = self.root_client.complete(messages, temperature=0.0)
        return self._extract_code(response)

    def _extract_code(self, response: str) -> str:
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1)
        return response

    def _execute_safe(self, code: str, documents: List[Document]) -> List[str]:
        """
        Execute model-generated code in a RestrictedPython sandbox.

        Never falls back to raw exec() if RestrictedPython is unavailable —
        that would be a genuine code-execution vulnerability given the code
        originates from an LLM operating on user-supplied corpus data.
        """
        try:
            from RestrictedPython import compile_restricted, safe_globals
            from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr
        except ImportError:
            raise ImportError(
                "RestrictedPython is required for the RLM pipeline's sandboxed code "
                "execution and has no unsafe fallback (raw exec() of LLM-generated "
                "code against corpus data is a security risk). "
                "Install with: pip install RestrictedPython"
            )

        byte_code = compile_restricted(code, filename="<rlm_generated>", mode="exec")

        restricted_globals: dict = dict(safe_globals)
        restricted_globals["_getattr_"] = safer_getattr
        restricted_globals["_getiter_"] = iter
        restricted_globals["_getitem_"] = lambda obj, index: obj[index]
        restricted_globals["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
        restricted_globals["documents"] = documents
        restricted_globals["result"] = []

        local_vars: dict = {}
        exec(byte_code, restricted_globals, local_vars)

        result = local_vars.get("result", restricted_globals.get("result", []))
        if not isinstance(result, list):
            raise ValueError(f"Generated code's `result` must be a list, got {type(result)}")
        return [str(r)[: self.rlm_cfg.max_tokens_per_slice] for r in result]

    def _delegate_to_sub_model(self, question: str, slice_text: str) -> str:
        """Ask the cheap sub-model to extract facts relevant to the question
        from one slice."""
        messages = [
            {"role": "system", "content": "Extract information relevant to the question from the text. Be concise."},
            {"role": "user", "content": f"Text:\n{slice_text}\n\nQuestion: {question}"},
        ]
        try:
            return self.sub_client.complete(messages, temperature=0.0)
        except Exception as e:
            logger.warning(f"Sub-model delegation failed: {e}")
            return ""

    def _aggregate(self, question: str, sub_answers: List[str]) -> str:
        """Root LLM combines sub-model outputs into the final answer."""
        if not sub_answers or not any(a.strip() for a in sub_answers):
            return "NOT FOUND: No relevant information found in the corpus."

        combined = "\n\n".join(f"[Slice {i + 1}]: {a}" for i, a in enumerate(sub_answers) if a.strip())
        messages = [
            {"role": "system", "content": "Synthesize a single final answer from the sub-answers provided."},
            {"role": "user", "content": f"Sub-answers:\n{combined}\n\nQuestion: {question}\n\nFinal answer:"},
        ]
        return self.root_client.complete(messages, temperature=0.0)
