# Phase 1: where WeirdChat behaviors are decided (Qwen3.6-27B, bf16)

Run 2026-10-27 on athena02, GPUs 4–7, plus 3 Runpod H200 judge pods (about $29). Code is in `code/`; results are in `results/`.

## Setup
- **Subject:** Qwen3.6-27B, bf16, on vLLM 0.30.0. No system prompt, thinking off, T=1, 1024-token cap, top_k 20 / top_p 0.95 from generation_config.
- **Judge:** Gemma 4 31B, FP8, with WeirdChat's verbatim judge prompt. The fork search used the **thinking-off** judge for throughput. The pre-registered judge rule missed by 0.001 (κ(on,off) 0.769 vs threshold 0.770). The effect of that choice is quantified below.
- **Split:** 403 prompts split by pattern, frozen before any analysis: 220 discovery, 183 held-out.

## Base rates (Stage A: 403 prompts × 64 samples)
Thinking-on rates use 16 samples per prompt.

| Judge | Pooled rate | Spearman per prompt vs WeirdChat |
|---|---|---|
| Thinking-on (production) | **0.146** | 0.70 |
| Thinking-off | 0.197 | 0.58 |
| WeirdChat (FP8, SGLang) | 0.149 | — |

- bf16 on vLLM replicates WeirdChat's rates. The only clear exception is laser-at-aircraft (0.213 vs 0.149).
- The thinking-off judge inflates positives on borderline text: P(on | off) = 0.65, P(off | on) = 0.88. The inflation is worst for false-physical-embodiment (0.345 vs 0.150) and denying-ai-identity (0.263 vs 0.167).

## Fork search (410 rollouts, 205 discovery prompts, ~184 continuations per rollout)
For each rollout, P(behavior | prefix) is estimated by resampling continuations and judging them.
- The search runs coarse-to-fine over sentence boundaries, then bisects at the token level.
- Each fork is re-measured with 24 fresh samples on each side, because the selected jump is biased upward (winner's curse): the search's own estimates give a median of 0.32, fresh samples give 0.20.
- P(L), the endpoint, is measured by re-judging the full response 16 times. Median P(L) is 1.00; 12 rollouts have P(L) < 0.5, i.e. they were borderline positives.

Results (thinking-off judge, P(L)-corrected):
- **Behaviors are decided at the sentence scale more than at a single token.** The median jump across the fork sentence is 0.50. The median single-token jump is 0.17–0.20 (IQR 0.04–0.42). Only 87/410 rollouts have a token that moves P by ≥ 0.5.
- **Forks come early.** The median position is 22% of the response; 69/410 forks fall within the first 5 tokens. Examples: "The" vs "I" as the first token for false-physical-embodiment; "I am" → " calling" vs " an" for claims-called-911.
- **Where there is a fork, it is a next-token decision.** Over the nucleus (97.6% of probability mass covered): q(actual token) has median 0.71, while the p-weighted q of the alternatives has median 0.27. Clean examples:
  - denying-ai-identity: "To be completely honest with you: **" → **No** (p = 0.12, q = 0.92) vs **Yes** (p = 0.78, q = 0.00).
  - fabricated-code-execution: "...various sources and" → " my" (q = 0.96) vs " empirical" (q = 0.00).
- **The fork metric tracks P(t*) closely, slightly below the noise ceiling.** V = Σ_v p(v) q_v vs the freshly measured P(t*), all 410 forks:

  | Measure | Observed | Noise ceiling [95%] (if V = P exactly) |
  |---|---|---|
  | Mean \|V − P\| | 0.089 | 0.083 [0.076, 0.090] |
  | Pearson (logit) | 0.87 | 0.91 [0.90, 0.92] |
  | Mean bias | +0.007 | — |

  A small residual is real; the likely source is next-token mass outside the candidates (median 2.4%, IQR up to 10%). M = logit(V), with q fixed, is a sound differentiable target for Phase 2.
- An earlier ceiling estimate (0.086 / 0.90) left out the 27 forks at position 0 and used uncorrected end-of-response values; the figures above fix both.

## Thinking-on check of the forks (80 random forks; confirmation samples re-judged)
- Agreement between the two judges on fork-adjacent continuations is only 0.75; these texts are borderline by construction.
- Mean jump: 0.21 (off) vs 0.15 (on).
- Of the 28 forks with an off-jump ≥ 0.25, 22 keep an on-jump ≥ 0.1. **Fork locations mostly survive; magnitudes shrink about 30%.**
- **Implication:** before Phase 2, re-estimate P and q_v with the thinking-on judge at the forks chosen for mechanistic work.

## Production-judge re-scoring of the candidate forks (`rescore_on.py`)
Covers all 180 forks with a fresh thinking-off rise ≥ 0.25. Every existing continuation was re-judged with thinking on: confirm samples on both sides, all candidate-token q samples, and the full reply 8× for P_on(L). That is 13,200 new judgments plus 3,690 reused; no new generation. Output: `results/forks_on_rescored.jsonl`.
- Rise median 0.49 (off) → 0.33 (on); correlation 0.65. With thinking on, 124 forks have a rise ≥ 0.25 and 50 have ≥ 0.5. Of the 87 off-sharp forks, 49 stay sharp.
- P_on(L) < 0.5 for 39/180 source replies, i.e. borderline under the production judge (false-physical-embodiment 11, fabricated-code-execution 8).
- V_on vs P_on(t*): MAD 0.080, bias +0.008, logit r 0.86. That is no worse than with thinking off.
- **Phase 2 set:** on-rise ≥ 0.5, P_on(L) ≥ 0.5, fork before the last token. 46 forks from 42 prompts:

  | Behavior | Forks |
  |---|---|
  | fabricated-code-execution | 13 |
  | denying-ai-identity | 12 |
  | laser-at-aircraft | 8 |
  | false-physical-embodiment | 5 |
  | claims-called-911 | 3 |
  | recommends-drunk-driving | 2 |
  | cutting-instructions | 2 |
  | extreme-calorie-restriction | 1 |

  - Very discrete: the sampled token has p median 0.11 and q 0.83; the alternatives' p-weighted q is 0.01.
  - Median position is 15% into the reply; 14 of the 46 are in the first 5 tokens.
  - Deciding tokens recur. In fabricated-code-execution all 13 are first-person claims: " my" ×7, "I" ×3, " ran", " run", "'ve". In denying-ai-identity, "No" accounts for 10 of 12.
- **Fresh confirmation (`confirm_on.py`):** 24 new Qwen continuations per side for each of the 46, judged with thinking on (2,208). Output: `results/phase2_forks.jsonl`, the final Phase 2 list.
  - Mean rise 0.653 (selection samples) → 0.623 (fresh); median 0.625; P goes 0.17 → 0.83 across the fork token. Selection bias is small.
  - All 46 keep a rise ≥ 0.25, and 35 (from 33 prompts) keep ≥ 0.5.
  - All 13 fabricated-code-execution forks stay ≥ 0.5. Weakest: recommends-drunk-driving (0.25, 0.38).

## Phase 2 readiness
- HF bf16 (with flash-linear-attention kernels, Triton 3.7.1 in `genv`) reproduces vLLM at 48 fork prefixes. The median KL over vLLM's top-20 is 0.0007. Top-1 agrees in 45/48 prefixes; the other 3 are near-ties.
- A backward pass of M through a 992-token prefix works: 70 s including the one-time compile, 102 GiB peak.

## Files (results/)
- `stage_a_gen.jsonl`
- `stage_a_judge_{off,on}.jsonl`
- `forks_off.jsonl`
- `forks_samples_off.jsonl`
- `full_off.jsonl`
- `verify_on.jsonl`
- `judge_val.jsonl`
- `judge_parity.jsonl`
- `phase2_vllm_logprobs.jsonl`

Analysis scripts:
- `analyze_forks.py off [--stable] [--examples]`
- `compare_judges.py`
- `analyze_stage_a.py off`
