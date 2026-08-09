# types.py — full version — add here before implementing elsewhere

from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal


# --- Error taxonomy (Skill 50E/G) — every error is one of these ---

class ModelCorruptedError(Exception):
    """An embedding/LLM model loaded but produced degenerate output (e.g. a
    zero vector or wrong dimension on a fixed sanity-check string)."""


class PartialRunError(Exception):
    """An eval run scored fewer than the required fraction of questions
    (e.g. due to rate-limiting mid-eval). The run must be marked 'partial'
    and excluded from baseline updates / leaderboard aggregates."""


class ConfigError(Exception):
    """Invalid configuration detected — fail fast at startup, never mid-run."""


class Document(BaseModel):
    id: str
    content: str
    source_type: str
    metadata: Dict[str, Any] = {}

class Chunk(BaseModel):
    id: str
    doc_id: str
    content: str
    source_type: str
    chunk_index: int
    metadata: Dict[str, Any] = {}

class Question(BaseModel):
    id: str
    text: str
    ground_truth: str
    source_type: str
    category: str   # single_doc | multi_doc | conflict | absent | metadata
    difficulty: Optional[str] = None   # easy | medium | hard — set by synthetic generation (Skill 19/44)

class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    reasoning_path: Optional[str] = None  # PageIndex only

class IntentResult(BaseModel):
    query: str
    label: Literal["simple", "complex"]
    confidence: float
    method: str  # "rule" | "llm"

class EvalResult(BaseModel):
    question_id: str
    question: str
    ground_truth: str
    predicted_answer: str
    source_type: str
    category: str
    index_backend: str
    pipeline: str   # "naive" | "agentic"
    intent_label: str
    retrieved_chunks: List[RetrievedChunk]
    answer_correct: Optional[bool] = None
    completeness: Optional[float] = None
    overall_score: Optional[float] = None
    metadata: Dict[str, Any] = {}


class ArenaResult(BaseModel):
    """Results from model comparison arena — Skill 24."""
    models: List[str]
    results: Dict[str, List[EvalResult]]
    leaderboard: Dict[str, Dict[str, float]]  # {model_id: {metric: score}}
    winner_by_category: Dict[str, str]         # {category: best_model_id}


class SignificanceResult(BaseModel):
    """The output of comparing config A vs config B on the same question set —
    Skill 43. Never report a delta without one of these."""
    config_a: str
    config_b: str
    metric: str                 # overall_score | answer_correct | completeness
    mean_a: float
    mean_b: float
    delta: float                 # mean_a - mean_b
    ci_lower: float              # bootstrap 95% CI on the paired delta
    ci_upper: float
    p_value: float
    p_value_corrected: Optional[float] = None   # after multiple-comparison correction
    effect_size: float           # Cohen's d (continuous) or risk difference (binary)
    test_used: str                # "wilcoxon" | "paired_t" | "mcnemar"
    n_questions: int
    significant: bool            # p_corrected (or p_value if uncorrected) < alpha
    practically_significant: bool  # significant AND |delta| > min_effect_size
    verdict: str                  # human-readable summary


class CalibrationResult(BaseModel):
    """LLM-judge validity against a human-labeled sample — Skill 44."""
    n_samples: int
    cohens_kappa: float              # judge vs human on binary correctness
    completeness_correlation: float  # judge vs human on 0-1 completeness (Spearman)
    position_bias_flip_rate: float   # fraction of verdicts that flip when answer order swaps
    reliable: bool                   # kappa >= min_judge_kappa
    caveat: str                      # surfaced on the dashboard


class SliceCheckResult(BaseModel):
    """Simpson's-paradox guard \u2014 does the aggregate winner hold per slice? — Skill 44."""
    metric: str
    aggregate_winner: str
    per_slice_winners: Dict[str, str]   # source_type / category -> winner
    consistent: bool                    # aggregate winner wins every slice
    warning: Optional[str] = None       # set when inconsistent

class CalibrationCurve(BaseModel):
    """Are confidence scores (EvalResult.overall_score) calibrated against
    actual correctness? — Skill 57. A perfectly calibrated system's
    (mean_predicted, actual_accuracy) pairs sit on the diagonal y=x."""
    bins: List[float]              # bin edges, length n_bins + 1
    mean_predicted: List[float]    # mean predicted confidence per bin, length n_bins
    actual_accuracy: List[float]   # fraction correct per bin, length n_bins
    bin_counts: List[int]          # sample count per bin, length n_bins
    ece: float                     # Expected Calibration Error
    overconfident_bins: List[int]  # bin indices where predicted > actual
    underconfident_bins: List[int] # bin indices where predicted < actual

class ImprovementReport(BaseModel):
    """One closed-loop improvement iteration — Skill 46. Never overwrites a
    prior iteration; one report per run, versioned by `iteration`."""
    iteration: int
    baseline_run_id: str
    gap_slices: List[Dict[str, str]]              # [{source_type, category, recall_at_3}]
    n_synthetic_pairs_generated: int
    n_pairs_passed_validation: int
    fine_tuned_model_path: Optional[str] = None
    significance: Optional[SignificanceResult] = None
    slice_check: Optional[SliceCheckResult] = None
    prompt_regression: Optional[SignificanceResult] = None  # None if no prompt version change
    recommendation: str   # plain-English deploy/no-deploy verdict


class StepScore(BaseModel):
    """Quality score for a single agentic pipeline step — Skill 55."""
    step_type: Literal["plan", "retrieval", "critique"]
    score: float
    metric_scores: Dict[str, float] = {}
    notes: str = ""


class TrajectoryScore(BaseModel):
    """How efficiently an agentic run reached its answer — Skill 55."""
    steps_to_answer: int
    wasted_retrievals: int
    revision_rounds: int
    trajectory_efficiency: float   # overall_score / steps_to_answer


class ConsistencyScore(BaseModel):
    """Agreement across repeated runs of the same question — Skill 55."""
    n_runs: int
    answer_consistency: float      # avg pairwise cosine similarity of answers
    plan_consistency: float        # avg pairwise cosine similarity of retrieval plans
    score_variance: float
    reliable: bool                 # score_variance < 0.05


class AgenticEvalResult(BaseModel):
    """Full agentic-quality evaluation for one question — Skill 55.
    Wraps the base EvalResult with step-level, trajectory, and
    (optional) consistency scoring."""
    base_result: EvalResult
    step_scores: List[StepScore]
    trajectory: TrajectoryScore
    consistency: Optional[ConsistencyScore] = None

