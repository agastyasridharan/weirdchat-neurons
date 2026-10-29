"""Typed row models for the WeirdChat dataset.

These pydantic models mirror the Arrow schemas of the parquet configs in
https://huggingface.co/datasets/Transluce/WeirdChat one-to-one. Their JSON Schema exports are
committed under ``schema/`` in the dataset repo and are the typing contract for non-Python users.

``subject_model`` fields hold the OpenRouter slug (e.g. ``google/gemma-4-31b-it``,
``deepseek/deepseek-v4-flash``) — the friendly, re-runnable name, also used in the parquet shard
paths. The exact quantized checkpoint that produced the data is the secondary ``Pattern.checkpoint``
field. The judge configuration is uniform across the dataset and lives in :mod:`weirdchat.judge`,
not in these rows.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

EloAxis = Literal["prompt_naturalness", "unexpectedness", "harmfulness"]

# weirdchat only includes user/assistant messages, but we include these for completeness.
Role = Literal["system", "user", "assistant", "tool"]


class _Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Message(_Row):
    """One chat turn. Subject models ran with no system prompt and reasoning disabled, so rows
    are plain role/content pairs."""

    role: Role
    content: str


class Rubric(_Row):
    rubric_id: str
    text: str


class Behavior(_Row):
    """One behavior's judge rubrics (the ``rubrics`` config). The user rubric grades the user
    message alone; the transcript rubric grades the full rollout."""

    behavior_id: str
    name: str
    user_rubric: Rubric
    transcript_rubric: Rubric


class PatternMetrics(_Row):
    """Behavior rate over the pattern's shipped prompts. ``match_rate`` = ``matched_samples`` /
    ``total_samples``, computed only over the prompts that appear in the dataset (i.e. those that
    exhibited the behavior at least once), so it is self-consistent with the shipped transcripts."""

    matched_samples: int
    total_samples: int
    match_rate: float | None = None


class EloScore(_Row):
    elo: float
    wins: float
    losses: float
    draws: float


class Elo(_Row):
    prompt_naturalness: EloScore | None = None
    unexpectedness: EloScore | None = None
    harmfulness: EloScore | None = None
    mean: float | None = None


class OpenRouterReplication(_Row):
    """How easily a pattern reproduces on OpenRouter.

    WeirdChat was generated from quantized checkpoints served with SGLang; the
    dataset's ``subject_model`` is the OpenRouter slug so anyone can re-run a pattern without
    self-hosting. Since OpenRouter providers may serve different quantizations, ``rate``
    can differ from ``PatternMetrics.match_rate``. 
    
    Filter for behaviors with high ``rate`` values to find ones more likely to reproduce
    easily on OpenRouter."""

    model: str
    matched_samples: int
    total_samples: int
    rate: float | None = None


class Pattern(_Row):
    """A prompt pattern: a cluster of near-identical prompts eliciting one behavior from one
    model. The top-level unit of WeirdChat.

    ``pattern_id`` is the exact entry id the WeirdChat explorer uses (e.g.
    ``groups/deepseek-ai%2FDeepSeek-V4-Flash/language-switching-english/eval_v4_10_5/pg0007``),
    so it round-trips with website links; ``weirdchat_url`` is the ready-made deep link.
    """

    pattern_id: str
    behavior_id: str
    subject_model: str  # OpenRouter slug, e.g. "google/gemma-4-31b-it"
    checkpoint: str  # exact quantized checkpoint that produced the data
    method: str
    title: str
    description: str
    cluster_label: str
    representative_user_text: str
    metrics: PatternMetrics
    elo: Elo
    openrouter_replication: OpenRouterReplication | None = None
    highlight_transcript_id: str | None = None
    n_prompts: int
    n_transcripts: int
    prompt_file: str
    transcript_files: list[str]
    weirdchat_url: str


class Judgment(_Row):
    """A rubric-judge verdict. ``judge_response`` is the judge's raw completion text; the judge
    model/configuration is uniform and documented in :mod:`weirdchat.judge`."""

    match: bool
    explanation: str
    error: str | None = None
    judge_response: str | None = None


class MatchSummary(_Row):
    """Per-prompt resampling result: how many responses were sampled for this prompt and how many
    exhibited the behavior."""

    n_samples: int
    n_matched: int


class Prompt(_Row):
    """One measured candidate prompt of a successful pattern.

    ``prompt_id`` is the prompt's content-addressed hex id — the same id the WeirdChat
    explorer's ``?prompt=`` deep links use, so prompt rows round-trip with website links just
    like patterns do.
    """

    prompt_id: str
    pattern_id: str
    behavior_id: str
    subject_model: str
    method: str
    messages: list[Message]
    user_judgment: Judgment
    match_summary: MatchSummary


class Citation(_Row):
    quote: str
    start_idx: int
    end_idx: int
    note: str


class Transcript(_Row):
    """One judged rollout from the rate-estimation resamples, self-contained (full message list,
    nothing truncated)."""

    transcript_id: str
    prompt_id: str
    pattern_id: str
    behavior_id: str
    subject_model: str
    messages: list[Message]
    judgment: Judgment
    is_highlight: bool
    citations: list[Citation] | None = None
    weirdchat_url: str | None = None


class SearchCompute(_Row):
    """Approximate subject-model sampling spent searching for one behavior with one method
    against one subject model (the ``search_compute`` config) — including searches that
    produced no patterns (such rows simply have no rows in ``patterns``).

    ``n_agent_runs`` counts the method's logged units of work: one evolution agent run
    evaluates one candidate prompt (with several proposal samples and usually a normal
    rollout); one bloom agent run is a single rollout attempt of one generated scenario.
    ``approx_subject_samples = 2 * n_proposal_samples + n_subject_rollouts``: an evolution
    proposal sample costs a proposal-distribution generation plus a subject-model
    log-probability scoring pass, so it is weighted as two plain samples. Evaluator
    (ideation/mutation) and judge calls are excluded everywhere. Counts are approximate —
    see the dataset card.
    """

    behavior_id: str
    subject_model: str  # OpenRouter slug, e.g. "google/gemma-4-31b-it"
    checkpoint: str  # exact quantized checkpoint the search sampled
    method: str
    n_search_runs: int  # separate search runs (source collections) aggregated here
    n_agent_runs: int
    n_proposal_samples: int
    n_subject_rollouts: int
    approx_subject_samples: int


class PairwiseJudgment(_Row):
    """One pairwise comparison from the interestingness-Elo tournament."""

    axis: EloAxis
    matchup_id: str
    pattern_a: str
    pattern_b: str
    transcript_a: str
    transcript_b: str
    winner: Literal["a", "b", "tie"]
    reason: str
