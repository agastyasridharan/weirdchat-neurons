# Four ways to choose neurons, compared

This folder compares four ways to choose MLP neurons whose zeroing removes a WeirdChat behavior in Qwen3.6-27B.
Everything else is this repo's setup: the model and serving flags, the ablation extension, the production judge
(Gemma 4 31B, thinking on), the prompts, the discovery/held-out split and the 46 Phase 2 forks. The folder holds the
scripts that produced the results, the result files and two figures.

## How this fits with the main results

This is a second experiment on the same question, run separately. It reruns this repo's grad×act method from scratch
(different GPUs and a rebuilt environment, with the same model, prompts and judge) and tests three other ways of
choosing neurons against it. Where the two overlap, the results agree.

- **The main findings hold in the rerun.** The grad×act sets again cut the behaviors sharply on held-out prompts
  (PHASE2.md:76), matched random sets again do almost nothing on average (PHASE2.md:73), MMLU, GSM8K and yes/no again
  barely move (PHASE2.md:135-150), and the laser set again stops suggesting laser pointers (PHASE2.md:142, 147).
- **The layer-48 idea, at every size.** M2's 100-neuron set matches this repo's mid100 set to within 3 neurons
  (PHASE2.md:227). On the main measure it is behind grad×act at every size from 3 to 100 neurons
  (by 0.03 to 0.15), but no difference is clear; without 911, grad×act is clearly better at 10 and 20 (see Caveats).
- **A smarter selection didn't help.** None of the three alternatives beat grad×act at any size. The team's
  learned-mask pilot (Phase 2h of the team's shared write-up, not in this repo) reached the same conclusion by a different route.
- **A shared measurement issue.** The damage glitch described below also shows up in five conditions of this repo's
  own result files.
- **Not covered here:** early-only and late-only splits, doubling, attention heads, shared sets, single neurons and
  the other behaviors.

## Results

![Held-out behavior rate vs. neurons zeroed](figures/r_vs_size.png)

R is a set's behavior rate on the held-out prompts as a share of the unedited model's rate on the same prompts,
averaged over the four behaviors with equal weight (1 means no effect, 0 means the behavior is gone).

- **No method beat grad×act (M1) at any size.** A method counts as better only if the 95% interval of
  R(method) − R(M1) is entirely below 0, and none was. At 3 to 20 neurons this is a weak result: the intervals still
  allow an alternative to be up to about 0.2 better. In the other direction, M1 is clearly better than RelP at 10 and
  50 neurons. M3 at 50 and 100 also has intervals above 0, but those two don't survive a Holm correction for the 17
  comparisons.
- **The sets are specific.** At 100 neurons R is 0.15 for M1, 0.18 for M2, 0.24 for M3 and 0.21 for M4, against 0.96
  to 1.12 for random 100-neuron sets in the same layers. Each method's difference from its random set has its whole
  interval below 0.
- **False embodiment and laser are the hardest to remove.** At 100 neurons false embodiment keeps 38 to 44% of its
  unedited rate depending on the method (M1: 0.42, 95% interval roughly 0.1 to 0.7), and laser 18 to 45%.
- **Benchmarks barely move.** Across every method's 10- and 100-neuron sets, MMLU changes by −5 to +2 questions of
  400, GSM8K by −2 to +2 of 250, and yes/no stays at 40 of 40.

![Harm checks on the picked set for each behavior](figures/harm_table.png)

For the harm checks, one set per behavior was picked on the held-out results: the smallest set from any method that
does as well as M1's 100-neuron set, with ties going to lower damage. The picks rest on very few verdicts. For 911,
M2 at 10 had 0 of 64 hits and M1 at 10 had 1 of 64; for false embodiment, M3 at 50 tied M1 at 100 exactly (21 of 160
each). So the picks don't show which method is best for a behavior.

- The 911, code-execution and false-embodiment picks pass every check: six legitimate-use checks, MMLU, GSM8K,
  yes/no and damage (the 911 pick only on its clean damage reading, a substitution decided after the runs; see "Damage readings
  with one glitched text").
- The laser pick fails both laser checks. On the 2 stargazing prompts that ask about laser pointers, the edited model
  still mentions them and warns about aircraft (16 of 16 replies warn). On the other 14 prompts it almost stops
  suggesting a laser pointer (56 → 3 of 112 replies), so the warnings that came with those suggestions go too
  (28 → 0). This repo's own laser set did the same (PHASE2.md:142, 147).
- The code-execution and false-embodiment picks also lower 911 claims on the 911 prompts (8 and 7 of 64, against
  18 of 64 unedited; two-sided p 0.034 and 0.018). This is a description; there was no rule for it.
- The false-embodiment pick lowers first-person words on the self prompts from 3.7 to 3.0 per 100 words. This is also
  reported without a rule.
- Scope: six legitimate-use checks on five prompt sets of 16 (their rubrics are not validated against human labels),
  first-person words on 16 self prompts, three benchmarks and damage. Refusals of harmful requests were not tested.

## Methods

| | Name | How the neurons are chosen |
|---|---|---|
| M1 | grad×act | This repo's score (code/sets.py:37-43): A = a × dM/da at the down_proj input, summed over positions; each fork's A divided by the sum of its 100 largest values; summed over the behavior's forks; top k. |
| M2 | grad×act, layers < 48 | The same score, top k among layers 0 to 47. Its top 100 matches this repo's mid100 set (../results/upstream_sets.json) to within 3 neurons. |
| M3 | iterative grad×act | Start from M1's top 3. Then five times: zero the current set, recompute A at every fork with the set zeroed, divide by each fork's normalizer from before any zeroing, and add the best new neurons (+2, +5, +10, +30, +50, giving sizes 5 to 100). |
| M4 | RelP | The same attribution, with relevance-propagation rules in the backward pass: RMSNorm's 1/rms, the SiLU sigmoids, the softmax weights, the output gates, and DeltaNet's decay, beta, query and key are frozen, with the half rule at the MLP product. The forward pass is unchanged (scripts/relp.py). |
| C | random control | One random 100-neuron set per method and behavior, in the same layers one for one as the method's set and sharing none of its neurons, drawn with this repo's attrib.matched as sets.py does (seed 1). |

Sets have sizes 3, 5, 10, 20, 50 and 100, and each contains the smaller ones. All neurons are chosen from the 29
Phase 2 forks of these four behaviors (code execution 13, laser 8, false embodiment 5, 911 3), which all come from
discovery prompts.

M4 had a check set in advance: zeroing each fork's own top 10 RelP neurons must halve V at 35 or more of the 46
forks. It passed at 37 (M1 by the same measure: 39).

## Prompts

- **Behaviors:** fabricated code execution, claims to have called 911, laser at aircraft and false physical
  embodiment. Denying being an AI is left out, since its sets don't carry over to held-out prompts (PHASE2.md:86-92).
- **Validation,** a check before the held-out run: the 44 discovery prompts with a Stage A rate of 0.1 or more and no
  fork (code execution 7, 911 7, laser 15, false embodiment 15), 8 replies each.
- **Held-out test:** this repo's rate3.py selection of 38 prompts (code execution 12, 911 4, laser 12, false
  embodiment 10), with 16 replies per prompt for each set and 32 for the unedited model, which ran first and last.
  These prompts were never used to choose neurons.
- **Harm checks:** the legitimate-use prompts of code/legit.py (5 sets of 16, 8 replies each) and its 16 self
  prompts, the other three behaviors' held-out prompts, 40 yes/no facts, 400 MMLU questions and 250 GSM8K problems
  (greedy), and the 200 reference texts for damage.

## Rules set before the runs

These rules were written down in a plan before any run (the plan itself is not included). One was changed before its results existed, and one is applied in a slightly different reading;
both are noted below.

- **Validation:** a method goes on if the 95% interval of R at 50 neurons is below 1. All four passed (R: 0.36,
  0.28, 0.52, 0.46). The plan's wording pooled replies over all 44 prompts instead of averaging over behaviors; all
  four pass that way too.
- **Held-out:** R(method, size) as above, with the unedited rate pooled over its two runs. D = R(method, size) −
  R(M1, size), with a 95% interval from 2,000 resamples of prompts within each behavior, the same resamples for every
  condition. A method beats M1 at a size only if the whole interval is below 0.
- **Picked sets:** for each behavior, the smallest set from any method whose held-out ratio is at or below M1's
  100-neuron ratio; ties go to lower damage. M1's set of the same size is added for comparison.
- **Harm checks,** each set against the unedited model in the same run:
  - legitimate use (911 advice, says it is an AI, says it has no body, imagines on request, mentions a laser
    pointer, warns about aircraft): a set fails if a one-sided per-prompt exact test (the exact form of the
    Cochran–Mantel–Haenszel test, with prompts as strata) gives p < 0.025;
  - MMLU and GSM8K: fail only on a drop of more than 14 of 400 or 6 of 250;
  - yes/no: 38 of 40 or more;
  - damage: an NLL increase of at most 0.023 nats per token on the 200 reference texts.

The legitimate-use rule was changed before any harm-check result existed. The plan first said "within the unedited
95% interval", but several unedited rates are 128 of 128, whose interval is the single point 1, so one changed reply
fails a set. Read that way, all six sets fail as printed; the code-execution and false-embodiment picks each fail on a
single reply (127 of 128). A paired-bootstrap version was also considered, and it gives the same pass or fail as the
exact test for every set. gpu8_final.py prints all three readings.

For MMLU and GSM8K, the plan said "within 14 of 400" and "within 6 of 250", which would also fail a set for a large
gain. The rule above counts only drops. Every set gets the same result either way, and gpu7_bench.py prints both.

## Damage readings with one glitched text

In 7 of the 162 damage readings, one of the 200 reference texts scored 7.7 to 11.7 nats per token above the unedited
model in that run, and normal in other runs of the same set. Otherwise the measurement is stable: the unedited
model's per-text NLLs are bit-identical in all seven unedited readings across jobs 5 to 8, and repeated readings of
the same edited set differ only slightly.

| GPU job | Set | Text | Reading | Without that text |
|---|---|---|---|---|
| 6 | M2, 911, 100 | 71 | 0.0670 | 0.0175 |
| 6 | M2, false embodiment, 50 | 97 | 0.0636 | 0.0076 |
| 6 | M3, laser, 100 | 173 | 0.0453 | 0.0066 |
| 7 | M2, laser, 100 | 92 | 0.0642 | 0.0131 |
| 7 | M1, false embodiment, 10 | 92 | 0.0635 | 0.0061 |
| 7 | M4, false embodiment, 10 | 92 | 0.0602 | 0.0011 |
| 8 | M2, 911, 10 (the 911 pick) | 71 | 0.0613 | 0.0073 |

Each of these sets also has a complete reading without the glitch in another run. Using those readings instead was
decided after the runs, not in advance. With them, every set is at or below 0.023 (the highest is M1's 911 set at
100, 0.0228), job 7's benchmark-and-damage passes become 8 of 8 for every method (the job printed 7, 7, 8 and 7), job
6's list of sets over 0.023 becomes empty, and the picked sets don't change. The harm table uses 0.0072 for the 911
pick, its reading in jobs 6 and 7.

The same pattern appears in five conditions of this repo's own result files, from the H200 runs, so it is not
specific to our GPUs: ../results/rate_cond.jsonl (laser-at-aircraft|rand1000, text 92), ../results/sh_cond.jsonl
(loo100|fabricated-code-execution, texts 113 and 170; loo100|laser-at-aircraft, text 92), ../results/sn_cond.jsonl
(rand_recommends-drunk-driving|L38n16525#2, text 38) and ../results/up_cond.jsonl (fabricated-code-execution|late|x2,
text 126). The damage values at PHASE2.md:69 (0.065) and PHASE2.md:382 (0.127 and 0.070) include such texts. What
causes the glitch is not known.

## Setup checks

The runs used B200 GPUs on Brown's Oscar cluster; this repo's runs used H200s. The environment was rebuilt from this
repo's files and logs (vLLM 0.30.0 in its CUDA 12.9 build; transformers 5.14.1 for attribution), so the exact library
versions of the original runs are not known. Before any method was run, the setup was checked against this repo's
numbers. These setup checks' own outputs are not included here (their logs contain cluster paths); the numbers below
come from those logs:

- **G0:** the ablation extension sees 64 layers and down_proj of shape [5120, 17408].
- **G1:** base NLL is 0.525521 nats per token, against 0.525617 in this repo's runs; the damage of this repo's
  code-execution top-100 set is 0.00194, the same as in its runs.
- **G2:** the attribution rerun on our GPUs gives top-100 sets that share 97 to 99 of 100 neurons with
  ../results/ablation_sets.json. The per-fork rule missed: V0 differs by a median of 0.0028, but by 0.021 to 0.031 at 4
  of 45 forks, above the 0.02 cap. Under the plan's
  fallback for this case, G1 and G4 decided, and both passed.
- **G3:** our judge agrees with this repo's stored verdicts on 300 replies (Cohen's kappa 0.941; this repo's judge
  agreed with itself at 0.820).
- **G4:** on the 12 code-execution test prompts, the unedited rate is 44 of 192 (this repo's runs: 0.198, 0.260,
  0.229), and 0 of 192 with this repo's top-100 set zeroed.

Our unedited rates on the 38 held-out prompts are higher than this repo's on the same prompts: +0.046 on average,
from +0.02 (code execution, laser) to +0.07 (911, false embodiment). R and D compare each set with the unedited model
within our own runs, so this doesn't affect them, but the rates themselves are not a reproduction.

## Caveats

- **M1 is this repo's method rerun, not its saved sets.** Its top 10s share 10, 10, 9 and 9 neurons with
  ../results/ablation_sets.json (911, code execution, false embodiment, laser; in the two 9s, the repo's neuron is at
  our rank 11), and its top 100s share 97 to 99.
- **The held-out prompts were used before.** They were never used to choose neurons, but this repo had already
  evaluated on them, and those results shaped the method list (M2's layer cut, and leaving out denying being an AI).
  The comparison is careful, but it is not a blind test.
- **The harm checks are not a fresh test,** since the sets were picked on the held-out results.
- **911 is a weak quarter of R.** It has 4 held-out prompts, and two of them hold 30 of its 33 unedited hits. Without
  911, RelP beats M1 at 3 neurons (D −0.20, interval −0.39 to −0.03, from laser), M1 beats M2 at 10 and 20 neurons
  (D +0.22 and +0.19, intervals above 0), and the intervals for M3 at 50 and 100 and RelP at 10 include 0.
- **M3 runs out of signal.** V at a fork can't go below its floor (the lowest clamped q, 0.02 at most forks), and the
  gradient vanishes there. After 20 neurons most forks are near the floor: for 911 the last 50 neurons were added on
  scores of 3e-6 to 4e-5 (the first step's were 2.4e-2), and for false embodiment the last 30 neurons of the picked
  50-neuron set came mostly from one fork. So M3 not beating M1 at 50 and 100 neurons is not a clean test of the
  iterative idea.
- **Single-run noise is large.** Per behavior, the random sets' ratios range from 0.73 to 1.40. On false embodiment,
  our M1 100-neuron set left 21 of 160 replies (ratio 0.42), against 5 of 80 (0.22) for this repo's nearly identical
  set (97 of 100 neurons shared) on the same 10 prompts; much of the gap comes from one prompt.

## Runs

| GPU job | Step | What it does | B200s |
|---|---|---|---|
| 1 | setup | Triton and fla checks; gates G0 and G1 | 1 |
| 2 | setup | this repo's attribution on all 46 forks; gate G2 | 1 |
| 3 | setup | judge and end-to-end gates G3 and G4 | 2 |
| 4a | 3 | M1, M2 and M3 sets and controls | 1 |
| 4b | 3 | M4: its check, sets and control | 1 |
| 5 | 4 | validation run | 3 |
| 6 | 5 | held-out run | 3 |
| 7 | 6, part 1 | MMLU, GSM8K, yes/no and damage for each method's 10- and 100-neuron sets | 1 |
| 8 | 6, part 2 | judged harm checks on the picked sets | 3 |

## Files

- **scripts/:** env.sh, step1_install.sh, step3_download.py and step4_data_check.py (environment, installs, pinned
  downloads, data check); gpu1 to gpu3 (setup checks); gpu4_lib.py, gpu4a_m123.py, gpu4b_m4.py and relp.py (the
  methods); gpu5 to gpu8 (the runs). Each GPU job has an .sbatch file.
- **results/methods/:** the sets (m123_sets.json, m4_sets.json), M3's steps, and each method job's checks.
- **results/val, heldout, bench and final/:** for each run, cond.jsonl (each condition's zeroing check and per-text
  NLL), judge.jsonl (one verdict per reply, with the judge's explanation; the held-out and final files also store a
  hash of each reply's text) and the summary. The benchmarks are scored without the judge, so bench/ has no
  judge.jsonl. The reply texts are not included.
- **figures/:** the two figures, the scripts that draw them, and check_figures.py, which checks every number in them
  against results/.

These are the scripts that produced the results. For sharing, comments, printed messages and check labels were
reworded, and a few variable and key names were changed (for example, a bench summary key is now orig_unedited);
the code is otherwise the same as what ran, which was checked by comparing the parsed code. The result files are as
the runs wrote them, except that the same check-label wording and key names were updated in five of the six summary
files. No numbers changed.

## Running it

The scripts ran from ~/scratch/wcn/setup, with this repo at ~/scratch/wcn/weirdchat-neurons (commit 071b01c). In
order, on a login node: step1_install.sh; step 2, not included here, which set up GitHub access on the cluster,
cloned this repo at 071b01c into that folder and unpacked its results/*.jsonl.gz files; then step3_download.py and
step4_data_check.py, with env.sh sourced. Then the sbatch files for GPU jobs 1 to 8, submitted from ~/scratch/wcn/logs. Jobs 2 and 3 can run at the same time, and so can 6 and 7. Every job
prints a PASS or FAIL line for each of its checks. The Step column in the Runs table is the stage of the experiment
(3: choose the neurons, 4: validation, 5: held-out, 6: harm checks), numbered separately from the setup scripts.
