"""
Cost & Latency Calculator — Skill 27

Track token usage and cost across all LLM calls.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Provider pricing per 1M tokens (input/output) — update as providers change
PRICING = {
    "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
    "gpt-4o":           {"input": 2.50,  "output": 10.0},
    "claude-3-haiku":   {"input": 0.25,  "output": 1.25},
    "claude-3-5-sonnet": {"input": 3.00,  "output": 15.0},
    "groq/llama3-70b":  {"input": 0.59,  "output": 0.79},
    "ollama":           {"input": 0.0,   "output": 0.0},
}


@dataclass
class CostRecord:
    """Single LLM call cost record."""
    id: str
    model_id: str
    stage: str                  # classification | retrieval | generation | synthetic_gen
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


class CostTracker:
    """
    Track LLM usage cost and latency across pipeline stages.
    
    Usage:
        tracker = CostTracker(cfg.cost)
        tracker.record("gpt-4o-mini", 100, 50, 1200, "generation")
        summary = tracker.summary()
    """

    def __init__(self, cfg=None):
        """
        Args:
            cfg: Optional CostCfg with alert_threshold_usd and pricing overrides
        """
        self.cfg = cfg
        self._records: List[CostRecord] = []
        self.alert_threshold = getattr(cfg, "alert_threshold_usd", 0.05) if cfg else 0.05
        
        # Load pricing (allow config overrides)
        self.pricing = PRICING.copy()
        if cfg and hasattr(cfg, "pricing"):
            self.pricing.update(cfg.pricing)

    def record(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        stage: str = "unknown",
    ) -> float:
        """
        Record an LLM call and compute cost.
        
        Args:
            model_id: Model identifier (e.g. "gpt-4o-mini", "ollama/llama3")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Latency in milliseconds
            stage: Pipeline stage (classification | retrieval | generation | etc)
            
        Returns:
            Cost in USD for this call
        """
        # Normalize model_id for pricing lookup
        pricing_key = model_id
        if "/" in model_id:
            # Handle provider/model format (e.g. "groq/llama3-70b")
            pricing_key = model_id
        
        # Get pricing or default to 0 (e.g. for unknown models)
        price_info = self.pricing.get(pricing_key, {"input": 0.0, "output": 0.0})
        
        # Compute cost (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * price_info["input"]
        output_cost = (output_tokens / 1_000_000) * price_info["output"]
        total_cost = input_cost + output_cost

        # Create record
        record = CostRecord(
            id=str(uuid.uuid4()),
            model_id=model_id,
            stage=stage,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=total_cost,
        )
        self._records.append(record)

        # Alert if exceeds threshold
        if total_cost > self.alert_threshold:
            logger.warning(
                f"Cost alert: {model_id} call cost ${total_cost:.4f} "
                f"exceeds threshold ${self.alert_threshold:.4f} "
                f"(stage: {stage}, tokens: {input_tokens} in + {output_tokens} out)"
            )

        logger.debug(
            f"Recorded: {model_id} | {stage} | "
            f"{input_tokens}/{output_tokens} tokens | "
            f"{latency_ms}ms | ${total_cost:.4f}"
        )

        return total_cost

    def summary(self) -> dict:
        """
        Generate summary statistics.
        
        Returns:
            {
                "total_cost_usd": float,
                "total_calls": int,
                "cost_per_call_usd": float,
                "total_input_tokens": int,
                "total_output_tokens": int,
                "by_stage": {stage_name: {"cost": float, "calls": int, "tokens": int}},
                "by_model": {model_id: {"cost": float, "calls": int, "tokens": int}},
                "avg_latency_ms": {"p50": float, "p95": float, "p99": float}
            }
        """
        if not self._records:
            return {
                "total_cost_usd": 0.0,
                "total_calls": 0,
                "cost_per_call_usd": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "by_stage": {},
                "by_model": {},
                "avg_latency_ms": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
            }

        total_cost = sum(r.cost_usd for r in self._records)
        total_calls = len(self._records)
        total_input = sum(r.input_tokens for r in self._records)
        total_output = sum(r.output_tokens for r in self._records)

        # By stage
        by_stage: Dict[str, dict] = {}
        for r in self._records:
            if r.stage not in by_stage:
                by_stage[r.stage] = {"cost": 0.0, "calls": 0, "tokens": 0}
            by_stage[r.stage]["cost"] += r.cost_usd
            by_stage[r.stage]["calls"] += 1
            by_stage[r.stage]["tokens"] += r.input_tokens + r.output_tokens

        # By model
        by_model: Dict[str, dict] = {}
        for r in self._records:
            if r.model_id not in by_model:
                by_model[r.model_id] = {"cost": 0.0, "calls": 0, "tokens": 0}
            by_model[r.model_id]["cost"] += r.cost_usd
            by_model[r.model_id]["calls"] += 1
            by_model[r.model_id]["tokens"] += r.input_tokens + r.output_tokens

        # Latency percentiles
        latencies = [r.latency_ms for r in self._records]
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))

        return {
            "total_cost_usd": round(total_cost, 4),
            "total_calls": total_calls,
            "cost_per_call_usd": round(total_cost / total_calls, 4),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "by_stage": by_stage,
            "by_model": by_model,
            "avg_latency_ms": {"p50": p50, "p95": p95, "p99": p99},
        }

    def to_dataframe(self):
        """
        Export records to a pandas DataFrame.
        
        Returns:
            pd.DataFrame with columns: id, model_id, stage, input_tokens,
            output_tokens, latency_ms, cost_usd, timestamp
        """
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not installed, cannot export to DataFrame")
            return None

        if not self._records:
            return pd.DataFrame()

        data = [
            {
                "id": r.id,
                "model_id": r.model_id,
                "stage": r.stage,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": r.latency_ms,
                "cost_usd": r.cost_usd,
                "timestamp": r.timestamp,
            }
            for r in self._records
        ]

        return pd.DataFrame(data)

    def reset(self):
        """Clear all recorded data."""
        self._records.clear()
        logger.info("CostTracker records cleared")

    def get_records(self) -> List[CostRecord]:
        """Get all cost records."""
        return self._records.copy()
