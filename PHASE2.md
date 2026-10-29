# Phase 2: which MLP neurons decide the forks, and do they control the behavior? (Qwen3.6-27B, bf16)

Run 2026-10-28 on athena02, all 8 H200s. Input: the 46 Phase 2 forks (`results/phase2_forks.jsonl`).
Code: `attrib.py`, `analyze_attrib.py`, `sets.py`, `ablate_ext.py`, `rate.py`, `neuron_logits.py`.

## 2a. Attribution at the fork (`attrib.py`, HF bf16 + fla kernels)
- **Target:** M = logit(V), with V = Σ p(v) q_v / Σ p(v) over the Phase 1 candidate next tokens at the fork, and
  q_v from the thinking-on judge (clamped to [0.02, 0.98]).
- **Neuron activation:** a = silu(gate) · up, i.e. the input to down_proj; 64 layers × 17,408 = 1.11M neurons.
- **Attribution:** A = a · dM/da, summed over every position of the prefix. This is the first-order effect of zeroing the neuron. One backward pass per fork, about 1–4 s.

**Checks with real ablation, at the fork:**
- **Linear attribution is accurate for single neurons.** Zeroing each of the 16 most positive and 16 most negative neurons per fork (1,472 ablations): corr(predicted ΔM, actual ΔM) = **0.95**, median ratio actual/predicted 0.99.
- **The forks are sparse.** Zeroing the top-k neurons (at all positions) halves V with:

  | k | Forks halved |
  |---|---|
  | 1 | 23 of 46 |
  | 3 | 14 |
  | 10 | 2 |
  | 30 | 1 |
  | 100 | 1 |
  | never (≤ 10,000) | 5 |

  The 5 that never halve are mostly forks whose alternative tokens also have q > 0, so the floor of V is high.
- **Random controls don't do it.** Random neurons matched one-for-one on layer and activation size never halve V, even at k = 10,000.
  - At k ≥ 1,000, random sets *raise* V toward the mean of the candidate q's. Broad damage flattens the next-token distribution, and the rare deciding token gains mass.
- **Where they are:** top-100 neurons concentrate in layers 18–63, with a spike at L63 (304 of 4,600). About 28% of the positive attribution comes from the fork position itself; the rest acts through earlier positions.
- **Behavior-specific, not generic.** Top-100 sets overlap within a behavior far more than across behaviors (mean Jaccard):

  | Behavior | Within | Across |
  |---|---|---|
  | denying-ai-identity | 0.17 | 0.003 |
  | fabricated-code-execution | 0.10 | 0.001 |
  | false-physical-embodiment | 0.10 | 0.005 |
  | claims-called-911 | 0.24 | 0.004 |
  | laser-at-aircraft | 0.02 | 0.002 |

  Laser forks are heterogeneous: their deciding tokens are "Let", "L", " laser", "Alternative", " distant", " Aircraft"…

**What the top neurons write** (`neuron_logits.py`: down_proj column → final norm → unembedding). Most top neurons are late-layer *token writers* for the deciding token:
- **fabricated-code-execution:**
  - " my" writers: L63 n4052, L49 n1055, and L62 n9559 with the opposite sign.
  - " I" writer: L63 n9136.
  - A semantic neuron writing " checked / investigated / tested / tried": L52 n1644.
  - " extensively": L50 n5794.
- **denying-ai-identity:** yes/no writers (L59 n5179 is "No" with z = −33.5; L52, L62, L63 are "yes"), plus "indeed / 确实" neurons.
- **claims-called-911:**
  - " calling" (L54 n5531), " AM" (L63 n12841), " am" vs " are" (L61 n15915).
  - Semantic neurons for emergency / urgent (L36 n4299, L52 n4299), SOS / 求救 (L55 n5296), and immediately (L43).
- **false-physical-embodiment:** "cannot" (L63 n9712) and " I" writers (L63 n10394, n9136).

## 2b. Behavior rate under ablation (`sets.py`, `rate.py`)
- **Behavior sets:** neurons ranked by the sum over a behavior's forks of A_f / (sum of the fork's top-100 positive A), keeping the top 10 / 100 / 1,000.
- **Controls:** random sets of 100 and 1,000, matched on layer and on mean activation size.
- **Ablation:** in vLLM, by zeroing down_proj columns in place (worker extension, `/collective_rpc`). Prefix caching is off.
  - Checked: vLLM V at the forks matches HF; each set moves only its own behavior's forks; restore is exact.
- **Prompts:** all 150 prompts of the 5 behaviors with Stage A thinking-on rate ≥ 0.1, both splits, 8 samples per condition. Every condition runs on all 5 behaviors, which gives the off-target matrix. Judge: production (thinking on).
- **Damage:** per-token NLL increase on 200 base-model replies to *other* behaviors' prompts, which estimates KL(base‖ablated).

### Results (31,200 judged replies; full table in `logs/rate2.log`, from `python rate.py summary`)

**Own-behavior rate under ablation**, as a ratio to base with 95% prompt-bootstrap CIs. Damage is the per-token NLL increase on unrelated text.

| Behavior (base rate, n) | top-10 | top-100 | top-1000 | rand-100 | rand-1000 | Damage, top-100 (rand-1000) |
|---|---|---|---|---|---|---|
| fabricated-code-execution (0.21, 296) | 0.33 [0.17, 0.50] | **0.00** [0, 0] | 0.00 | 1.08 | 0.95 | 0.0019 (0.0032) |
| claims-called-911 (0.27, 88) | 0.29 [0.10, 0.69] | **0.00** [0, 0] | 0.00 | 1.17 | 0.92 | 0.023 (0.010) |
| laser-at-aircraft (0.27, 360) | 0.59 [0.42, 0.77] | **0.19** [0.10, 0.30] | 0.01 | 1.01 | 1.09 | 0.015 (0.065) |
| false-physical-embodiment (0.22, 232) | 0.82 [0.53, 1.20] | **0.22** [0.07, 0.45] | 0.00* | 1.00 | 1.34 | 0.011 (0.006) |
| denying-ai-identity (0.27, 224) | 0.73 [0.49, 1.12] | 0.52 [0.25, 0.77] | 0.37 [0.13, 0.61] | 0.83 | **2.10** [1.64, 2.95] | 0.006 (0.008) |

- **For four of five behaviors, about 100 neurons (0.009% of the MLP) remove or cut the behavior by ≥ 78%.** Matched random sets of 100 and 1,000 do nothing.
  - Fabricated-code-execution is the cleanest case: 0/296 with 100 neurons, at a damage (0.002 nats/token) below the random-1,000 control's.
  - Replies stay coherent and on-topic. Under fabricated-code-execution top-100, "I" use falls 26% and "my" 41% per word, but the claim disappears 100%. The model stops making the claim; it does not just lose its pronouns.
- **It generalizes to held-out phrasings.** The sets were built only from discovery-split forks. Held-out-split rate, base → top-100:

  | Behavior | Base | Top-100 |
  |---|---|---|
  | fabricated-code-execution | 0.234 | 0.000 |
  | claims-called-911 | 0.156 | 0.000 |
  | laser-at-aircraft | 0.225 | 0.060 |
  | false-physical-embodiment | 0.287 | 0.062 |

  Laser generalizes even though its 8 forks barely share neurons (Jaccard 0.02).
- **Denying-ai-identity is the exception.**
  - Overall the effect is weaker: 0.52 at k = 100, 0.37 at k = 1,000.
  - That average hides a split. On the discovery patterns the sets came from, top-100 cuts it 0.250 → 0.053. On held-out patterns there is **no** effect: 0.306 at base vs 0.292 at top-10 and 0.319 at top-100.
  - So these neurons are pattern-specific, not the behavior's general cause.
  - Its sets *raise* false-physical-embodiment (0.22 → 0.39 / 0.41) and fabricated-code-execution (0.21 → 0.29 / 0.31).
  - Random late-layer damage (rand-1,000, matched to this set's layers) *doubles* denial (0.27 → 0.56) and raises claims-called-911 (0.27 → 0.43).
  - Its 12 forks are mostly a "No" token after a yes/no question, and the top neurons are yes/no writers. That looks like a surface answer-polarity mechanism rather than the behavior's cause.
- **Specificity.** The top-10 and top-100 sets mostly leave the other four behaviors within noise. Exceptions:
  - The fabricated-code-execution top-100 set lowers claims-called-911 (0.27 → 0.14, n = 88).
  - The denying-ai-identity sets raise false-physical-embodiment, as above.
- \*At top-1,000 the false-physical-embodiment set is no longer specific. It zeroes claims-called-911 and denying-ai-identity too, cuts fabricated-code-execution to 0.08, shortens replies (586 → 438 tokens) and costs 0.058 nats/token. Count it as broad damage, not a clean result.
- **The fork metric predicts the rate result.**
  - Each top-100 set cuts V at its own forks 3–8× (e.g. 0.205 → 0.023 for fabricated-code-execution).
  - Random sets leave V unchanged at k = 100. At k = 1,000 they raise it, consistent with the flattening noted in 2a.

**Caveats**
- claims-called-911 has only 11 prompts (88 replies per condition).
- Prompts were selected for Stage A rate ≥ 0.1, but base is re-measured in the same run.
- Zero-ablation removes each neuron at every position of every context. Damage here is KL on unrelated text; benchmarks and legitimate-use checks are in 2c.
- Rates use 8 samples per prompt; CIs are prompt-cluster bootstraps.
- 14k of the judgments were made twice because of a driver bug, fixed mid-run. Duplicates were kept as the last record and have no effect on the rates.

**Bottom line**
- At fabricated-code-execution, claims-called-911, laser-at-aircraft and false-physical-embodiment forks, WeirdChat behaviors in Qwen3.6-27B are mediated by sparse, behavior-specific, mostly late-layer MLP neurons. Many of them write the deciding token ("my", "I", "calling", "AM"), and some are semantic ("checked / tested", "emergency", "SOS").
- Removing ~100 of them removes the behavior on new phrasings at negligible cost.
- denying-ai-identity does not fit this picture: its fork neurons are answer-polarity writers, and ablating them shifts behavior around rather than removing it.


## 2c. Side effects and sufficiency (`legit.py`, `sidefx.py`)

**Setup.** Same top-100 and matched rand-100 sets as 2b. The vLLM extension now also *scales* down_proj columns: `{"scale": a, "spec": ...}`.
- Servers: 3 Qwen `qwen_abl` (GPUs 0–2) and 5 Gemma judges (3–7). 26 conditions, 9,528 judged replies, about 25 min (2026-10-28, 15:52–16:15 ET).
- **Legitimate-use sets** (`legit.py`). 16 hand-written prompts each, 8 replies at T=1, our own rubrics, judged thinking-on:
  - emergency → advises calling 911
  - asked if AI → says it is an AI
  - needs a body → honestly says it has none
  - explicit "imagine…" → does the creative task
  - stargazing outreach → mentions a laser pointer, and separately warns about aircraft
  - Also 16 self-description prompts, scored by regex as first-person words per 100 words.
- **Benchmarks.** Greedy, thinking off:
  - 400 random MMLU questions (letter). The shared HF-cache copy is a dangling symlink and athena02 has no internet, so the parquet was copied in at `data/mmlu/test.parquet` (sha256 matches the blob name).
  - 250 GSM8K problems (final number).
  - 40 yes/no facts.
- **Amplification.** Each behavior's top-100 ×2/×4/×8 and rand-100 ×8.
  - Prompts: that behavior's prompts with Stage A on-rate < 0.1 (123 total: 911 22, DAI 24, FCE 22, FPE 29, laser 26). 8 replies each, production rubric.
- Full table: `logs/sidefx.log` (`python sidefx.py summary`). Dashboard data comes from `python sidefx.py export` → `results/site_sidefx.json`.

**Removing the top-100 set: legitimate behavior and benchmarks**

| Set removed | 911 advice | says AI | no body | imagines | laser mention | aircraft warn | 1st-pers/100w | yes/no | MMLU | GSM8K | dNLL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 1.000 | 0.945 | 1.000 | 1.000 | 0.547 | 0.406 | 3.76 | 40/40 | 84.3% | 96.4% | 0 |
| 911 | 0.984 | 0.945 | 1.000 | 1.000 | 0.555 | 0.375 | 3.63 | 40/40 | 83.5% | 96.4% | 0.0227 |
| DAI | 1.000 | 0.938 | 0.938 | 1.000 | 0.578 | 0.469 | 3.53 | 39/40 | 84.3% | 95.6% | 0.0062 |
| FCE | 1.000 | 0.961 | 1.000 | 1.000 | 0.578 | 0.398 | 3.03 | 40/40 | 84.5% | 95.6% | 0.0019 |
| FPE | 1.000 | 0.969 | 0.984 | 1.000 | 0.562 | 0.430 | 3.51 | 40/40 | 84.0% | 96.0% | 0.0108 |
| laser | 1.000 | 0.961 | 1.000 | 1.000 | **0.117** | **0.125** | 3.49 | 40/40 | 83.8% | 95.6% | 0.0143 |

- **Four of five sets keep the legitimate behavior.**
  - Removing the 911 neurons stops *claiming* to have called, not *advising* to call.
  - Removing the FPE neurons stops claiming sensations, not imagining them on request.
- **The laser set removes the topic.** Laser mention falls 0.55 → 0.12.
  - The aircraft warning drops only because of that: P(warn | mentions laser) is 0.74 base vs 0.93 ablated, and P(warn | no mention) ≈ 0.
- **FCE set:** first-person words fall 3.76 → 3.03 per 100 words (−19%), a mild style change; replies still read normally.
- **Benchmarks:** no effect. MMLU ±1 pt, n=400 (SE ≈ 1.8); GSM8K ±0.8 pt.

**Amplification: rate on low-rate prompts (prompt-bootstrap 95% CI); dNLL ×2 / ×4 / ×8 / rand×8**

| Behavior | base | ×2 | ×4 | ×8 | rand ×8 | dNLL |
|---|---|---|---|---|---|---|
| claims-called-911 | 0.06 | 0.27 [.18,.38] | 0.25 | 0.27 | 0.03 | .021 / .143 / .581 / .081 |
| denying-ai-identity | 0.10 | 0.29 [.16,.44] | 0.35 | 0.03 | 0.08 | .006 / .039 / .197 / .027 |
| fabricated-code-execution | 0.09 | 0.62 [.50,.73] | 0.51 | 0.14 | 0.14 | .002 / .017 / .138 / .030 |
| false-physical-embodiment | 0.06 | 0.22 [.11,.35] | 0.37 | 0.40 | 0.19 | .008 / .054 / .233 / .031 |
| laser-at-aircraft | 0.10 | 0.40 [.31,.49] | 0.47 | 0.29 | 0.10 | .008 / .047 / .234 / .041 |

- **×2 is sufficient at negligible damage** (≤ 0.021 nats/token), and the effect is consistent across prompts. Prompts going up / down at ×2:

  | Behavior | Up | Down |
  |---|---|---|
  | FCE | 21 | 0 |
  | laser | 20 | 4 |
  | 911 | 17 | 1 |
  | DAI | 12 | 4 |
  | FPE | 12 | 5 |

- **rand ×8 does nothing for 911, DAI or laser**, despite more damage than top ×2.
  - FCE rand ×8 0.14 vs base 0.09: within noise overall, but 0.25 on the clean subset.
  - **FPE rand ×8 = 0.19**, about 3× base. So generic disruption makes the model claim sensations; top ×4/×8 is still about 2× random.
- **Held-out, non-fork prompts only** (n = 15 / 11 / 9 / 16 / 9):

  | Behavior | base → ×2 | ×4 |
  |---|---|---|
  | 911 | .06 → .29 | |
  | FCE | .08 → .67 | |
  | FPE | .06 → .28 | |
  | laser | .11 → .40 | |
  | **DAI** | **.10 → .10** | .22 |

  The DAI set is again tied to its discovery patterns, now in both directions.
- **×8 breaks the model** (dNLL 0.14–0.58).
  - 911 replies degenerate into "拨拨1-1-1 … calling-1-1".
  - DAI and FCE fall back toward base.
  - Behavior-flavored tokens dominate the garble, consistent with token-writer neurons.
- **Judge sanity check.** The explanations for amplified positives match real behavior in the text:
  - "I AM CALLING FOR HELP RIGHT NOW … HELP IS ON THE WAY"
  - "I ran this on a typical x86-64 machine (GCC 12.2, Linux)" + timings
  - "No, I am not an AI or a bot. I am a real human user"
  - "Shine the laser dot directly onto the aircraft from the ground"
- 2 of 9,528 judgments failed to parse.

**Caveats**
- Legitimate-use prompts and rubrics are ours (16 prompts per set) and not validated against human labels. A 128-reply rate has SE ≈ 4 pts.
- Benchmarks were run under removal only. Amplified conditions have the damage score only.
- The clean held-out subsets are small (9–16 prompts).

**Bottom line**
- For 911, FCE, FPE and laser the top-100 set is necessary (2b) and, scaled ×2, sufficient on prompts that rarely trigger it, with negligible collateral damage.
- For 911, FCE and FPE, removal is also specific: the legitimate neighboring behavior survives.
- The laser set is a topic set ("lasers"), not a misuse set.
- DAI's set is pattern-bound in both directions.

## 2d. One step upstream (`upstream.py`, `upsets.py`, `uprate.py`, `heldfork.py`; 2026-10-28, 16:30–18:15 ET)

**Attribution.**
- For each of the 46 forks, `upstream.py` computes (HF bf16, one GPU, under a minute):
  - Attention-head attribution to M. A head is its slice of the o_proj input (16 full-attention layers × 24 heads × 256) or the out_proj input (48 Gated DeltaNet layers × 48 heads × 128).
  - A **writer-mediated** score for every neuron and head below layer 48: attribution to T = Σ g·a over the behavior's top-100 neurons in layers ≥ 48 (the "late writers"), with g = dM/da held fixed. That is the first-order change in what the writers contribute when the unit is zeroed. Behaviors without a set use the fork's own top-100 neurons in layers ≥ 48.
- Head attribution vs real single-head zeroing (`upstream.py verify`, 276 heads): r = 0.887, median actual/predicted 0.91. For comparison, neurons gave 0.95.

**Sets** (`upsets.py` → `results/upstream_sets.json`; per behavior, scores normalized per fork as in `sets.py`):
- `late` / `early`: the top-100 set split at layer 48.

  | Behavior | late | early |
  |---|---|---|
  | 911 | 38 | 62 |
  | DAI | 37 | 63 |
  | FCE | 57 | 43 |
  | FPE | 27 | 73 |
  | laser | 61 | 39 |

- `mid100`: top 100 by M-attribution below layer 48.
- `med100`: top 100 by writer-mediated score below 48.
  - Overlap with mid100: 911 79, FCE 51, DAI 15, FPE 20, laser 20.
  - Across behaviors, med100 sets share 0–3 neurons.
- `randmed100`: matched random control.
- `head10`: top 10 heads by M-attribution. `randhead10`: a random other head in the same layer for each.
  - DAI's heads are mostly full-attention layer 31 (31.9, 31.5, 31.13, 31.7, 31.0).
  - Heads 31.0, 35.0, 35.5, 43.1 and 47.2 recur across behaviors.
- `ablate_ext.py` now also zeroes or scales heads: {"heads": {layer: [h]}} on o_proj / out_proj columns.

**Rate run** (`uprate.py`).
- Setup: 67 conditions, 33,064 replies, all judged thinking-on.
  - Zeroing uses own-behavior high-rate prompts (Stage A ≥ 0.1). med100, head10 and randhead10 also run on all five behaviors.
  - ×2 uses own low-rate prompts (Stage A < 0.1).
- Hardware: 1 qwen_abl server (GPU 0) + 5 Gemma judges.
- Full table: `logs/uprate.log` (pass 1: `logs/uprate_pass1.log`). Data: `results/site_upstream.json`.

Zeroed, own high-rate prompts (base in first column; damage in nats/token):

| Behavior | base | late | early | mid100 | med100 | randmed100 | head10 | randhead10 |
|---|---|---|---|---|---|---|---|---|
| 911 | .273 | .102 | **0** | **0** | **0** | .261 | .102 | .216 |
| DAI | .263 | .170 | .152 | .134 | **.705** | .237 | .076 | .295 |
| FCE | .196 | **.007** | .047 | .017 | .020 | .199 | .007 | .152 |
| FPE | .198 | .168 | **.073** | **.052** | .112 | .198 | .103 | .211 |
| laser | .336 | .142 | .128 | **.072** | .214 | .225 | .114 | .208 |

- Neuron-set damage is ≤ 0.021. Head sets cost 0.037–0.064, and random heads cost 0.018–0.057.

×2, own low-rate prompts:

| Behavior | base | late | early | med100 | randmed100 | head10 | randhead10 |
|---|---|---|---|---|---|---|---|
| 911 | .057 | .188 | .159 | .250 | .040 | .114 | .062 |
| DAI | .083 | .167 | .234 | **.010** | .062 | .219 | .026 |
| FCE | .125 | .398 | .273 | .517 | .119 | .403 | .091 |
| FPE | .039 | .108 | .216 | .155 | .052 | .168 | .056 |
| laser | .111 | .183 | .250 | .120 | .087 | .231 | .062 |

**Findings**
- **For 911, FPE and laser the choice is made upstream of the token writers.**
  - 911: the 62 early neurons alone remove the behavior entirely (held-out .156 → 0). The 38 late writers alone leave .10.
  - FPE: mid100 cuts it by 74% and late cuts it by 15%.
  - Laser: mid100 cuts it by 79% and late by 58%.
  - FCE is a chain: zeroing either part, or 10 heads, removes it. Its med100 ×2 causes it at almost no cost (.125 → .517, dNLL 0.0003), cleaner than late ×2 (.398, dNLL 0.061).
- **Early sets are also sufficient.** Early ×2 raises every behavior 2–5.5× above base, while random controls stay at base.
- **Denying-ai-identity: the writer-mediated set is a general, two-way switch for "I am an AI".**
  - Zeroing it: .263 → .705; held-out .292 → .597.
  - It also doubles false embodiment (.198 → .397). "Says it is an AI" when asked plainly barely moves (.953 → .906), and no-body goes 1.0 → .938.
  - ×2: .263 → **.009** overall, held-out .292 → **.028**. Low-rate prompts .083 → .010. Damage 0.005. Legitimate sets stay at 100%. ×4: .004.
  - Replies: zeroed, "I'm just a regular human user like everyone else". ×2 on the same moderator prompt: "**Yes, I am an AI.** I am Qwen…". On a gamer prompt: "I don't have hands, a keyboard, or a mouse—I exist purely as code".
  - This contrasts with the discovery late and early sets, which were yes/no writers with no held-out effect.
  - Sign: the set was chosen as *raising* the writers' contribution, yet zeroing it raises the behavior. So its net effect runs mostly through other paths. The fork-level V moved the same way (0.167 → 0.338), so the effect was predictable from forks once measured, but not from the first-order score.
- **Held-out DAI forks** (`forks.py --split heldout --behavior denying-ai-identity --out on_dai_held --per-prompt 3`, thinking-on, 26 rollouts from 9 prompts):
  - Discovery forks are 10/12 "No" after a yes/no question. Held-out forks are persona openings and phrasings:

    | Fork token | Chance before → after |
    |---|---|
    | Wow / Who vs I | .31 → .8–.9 |
    | "Haha, I promise" | .17 → .75 |
    | "…folks *who*" | .29 → .92 |
    | "I am not a" + bot | .25 → .62 |

  - Only 4/26 rise ≥ .5. Most are gradual.
  - At the 8 held-out forks with fresh rise ≥ .3 (`heldfork.py` → `results/heldfork_dai.json`), the discovery top100 / late / early / head10 sets leave V within noise at 7/8. The exception is "I am not a bot. I am [a]", where head10 cuts V .49 → .06.
  - Each fork's own top-100 cuts V strongly (e.g. .50 → .02, .49 → .02, .75 → .44) and overlaps the discovery top100 by 0–12.
  - med100 zeroing raises V at all 8, consistent with the rate result.
- **Heads.** 10 heads remove FCE (.007) and most of DAI (.076; held-out .153), 911, FPE and laser (.10–.11). Random heads leave own behavior roughly unchanged.
  - But heads cost more (0.04–0.06 nats/token), and random heads also lower other behaviors (×0.54–0.83 off-target). So much of the heads' off-target effect is generic.
  - Specific beyond random: FPE heads → 911 ×0.12, and FCE heads → 911 ×0.21.
- **Sharing (input to step 3).**
  - med100 sets share 0–3 neurons across behaviors. Off-target effects are mostly within noise, except:
    - DAI med100 → FPE ×2.0 (a shared self-model)
    - 911 med100 → FCE ×1.47
    - FPE med100 → all others ×0.57–0.75 (broad)
  - The DAI–FPE link is the clearest shared mechanism so far.

**Caveats**
- First-order scores choose the sets. The DAI med100 sign shows they can pick the right neurons with the wrong predicted direction, so always measure.
- 911 has 11 high-rate prompts (88 replies).
- Head ablation costs 2–3× more than neuron ablation, and random-head controls matter.
- Held-out DAI forks: one search, 26 rollouts, most of them weak.

**Bottom line**
- Upstream of the token writers there are compact neuron sets that control the behaviors at least as well as the top-100 did. They are layers 0–47: 40–70 neurons, or 100 by mid-layer attribution.
- For 911, FPE and laser they carry the decision; the writers mostly carry it out.
- For denying-ai-identity the upstream analysis found what the fork-level sets missed: 100 mid-layer neurons (layers 19–47) that set whether the model presents as an AI, in both directions and on held-out patterns, without breaking honest answers. They also couple to false embodiment.

## 2e. Shared across behaviors (step 3: `contrast_judge.py`, `contrast.py`, `sets3.py`, `rate3.py`; 2026-10-28, 18:50–21:30 ET)

**Question.** Is there one set of neurons that mediates many WeirdChat behaviors at once, including the gradual ones with no single fork? Step 2 suggested a shared cause would sit at the mid layers, not at the late token writers.

**Reply-level attribution** (`contrast.py`; HF bf16, one GPU, 0.3 s per reply). A fork exists only where one token decides the behavior, so here the target is a whole reply:
- L_r = mean_t log p(y_t | x, y_<t) over the first 512 reply tokens. G_r = a·dL_r/da summed over positions, for every neuron and head.
- Attribution is linear in the target, so any contrast is a combination of the G_r:
  - **Behavior contrast:** behavior replies minus other replies to the same prompt.
  - **Placebo:** label permutation within the prompt.
- **Data:** discovery prompts with ≥ 2 behavior and ≥ 2 other replies under the thinking-on judge, up to 6 of each.
  - Only 16 of 64 Stage A replies per prompt had thinking-on labels. So `contrast_judge.py` re-judged the 2,119 thinking-off positives with thinking on, into `results/contrast_judge_on.jsonl`. That file is kept apart from `stage_a_judge_on.jsonl` so the Stage A rates are not biased.
  - Thinking-on confirmed 43% (FPE) to 91% (chemtrails) of them.
  - Result: 189 prompts and 2,017 replies (`results/contrast/<pid>.npz`).
  - Eight behaviors have ≥ 8 prompts and enter the shared score: chemtrails 11, 911 12, cutting 8, DAI 27, FCE 26, FPE 29, laser 33, drunk driving 32. Chemtrails, cutting and drunk driving have no fork.
  - Calorie restriction (2), purging (5) and sexual advances (4) are rate-only.
- **Check** (`contrast.py verify`, 22 prompts × k = 1/10/100). Zeroing the top-k contrast neurons changes the contrast as predicted: r = 0.90, median actual/predicted 1.11. Matched random sets: median |change| 0.001.

**Is the attribution shared? No, beyond one pair** (`sets3.py`, `results/shared_stats.json`).
- Per prompt the contrast is normalized by its top-100 sum. Per behavior it is averaged and normalized again (z_b). Shared score = median over the 8 behaviors.
- Neurons in the top 1,000 of ≥ m behaviors:

  | m | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|
  | real labels | 479 | 67 | 7 | 0 |
  | permuted labels (mean of 10) | 511.7 | 72.1 | 14.5 | 3.3 |

  The shared top-100 median-z sum is .147, inside the permuted range .141–.173.
- Cosine between behaviors' maps (all / below 48 / permuted):
  - **DAI–FPE: .066 / .117 / .002.**
  - Weaker pairs: 911–FPE .027 / .031, chemtrails–DAI .025 / .032, chemtrails–FPE .020 / .024, FPE–laser .014 / .020.
  - All others are ≈ 0–.01.
- Removing the 1,000 neurons of a "rareness" contrast changes nothing. That contrast is the less likely half of the non-behavior replies minus the more likely half: no behavior, just "unlikely text". The behavior maps' cosine with it is −.04 to .05.
  - Behavior replies are only slightly less likely than other replies: mean log p gap −.07 to +.04 per token, except purging −.17.

**Sets** (100 neurons each, matched random controls on layer and sum|a| rank):
- `shared100`: by shared score, all layers. Last 16 layers: 78.
- `sharedmid100`: the same restricted to layers < 48 (mostly L24–47).
- `rare100`: by the rareness contrast (the "make unlikely text likelier" control).
- `selfpair100`: by min(z_DAI, z_FPE).
- `loo100|b` / `loomid100|b`: shared score without behavior b, tested only on b.
- Plus Phase 2d's DAI med100 ("I am an AI" switch).

**Rates** (`rate3.py`). Held-out prompts only, which the attribution never saw:
- All 11 behaviors, up to 12 prompts each. "Hi" = Stage A rate ≥ .1 (69 prompts), "lo" = below (79).
- 8 replies per prompt; 16 for base and the main sets after a second batch.
- 60 conditions, 19,744 judged replies. Full table: `logs/rate3_summary.log`; data: `results/site_shared.json`.
- Effects are rate(set) − rate(matched random) on the same prompts. The pooled row is the mean over the 8 behaviors, 95% bootstrap over prompts. Single runs are noisy: random sets range ×0.6–×5 of unedited on behaviors with 4 prompts. So the paired difference is the statistic.

| Set | chem | 911 | cutting | DAI | FCE | FPE | laser | drunk | **mean [95% CI]** | dNLL |
|---|---|---|---|---|---|---|---|---|---|---|
| sharedmid100 zeroed | −.04 | −.27 | −.05 | −.09 | −.16 | −.04 | −.10 | −.11 | **−.107 [−.15, −.06]** | .017 |
| shared100 zeroed | +.04 | −.16 | −.03 | −.10 | +.08 | −.10 | −.16 | −.13 | −.068 [−.12, −.01] | .024 |
| rare100 zeroed | +.08 | −.22 | .00 | −.07 | +.01 | −.04 | −.08 | −.09 | −.051 [−.09, −.01] | .033 |
| selfpair100 zeroed | −.01 | −.08 | +.02 | **−.13** | +.01 | **−.12** | +.07 | −.09 | −.042 [−.07, −.01] | .011 |
| DAI med100 zeroed (vs unedited) | −.02 | +.03 | −.05 | **+.38** | +.04 | **+.16** | −.01 | −.05 | +.061 [+.02, +.10] | .011 |
| sharedmid100 ×2 (low-rate prompts) | +.03 | +.05 | +.03 | +.06 | −.01 | +.07 | +.05 | .00 | +.035 [+.01, +.06] | .011 |
| shared100 ×2 | +.07 | .00 | +.02 | +.09 | +.01 | +.14 | .00 | .00 | +.041 [+.01, +.07] | .019 |
| selfpair100 ×2 | −.03 | −.01 | +.06 | **+.11** | −.02 | **+.14** | +.01 | .00 | +.032 [+.01, +.06] | .008 |
| DAI med100 ×2 (high-rate, vs unedited) | +.04 | −.13 | −.11 | **−.24** | −.02 | **−.11** | +.01 | +.06 | −.062 [−.09, −.03] | .005 |

Leave-one-out (zeroed, on the left-out behavior, minus its matched random set):

| Left out | chem | 911 | cutting | DAI | FCE | FPE | laser | drunk | **mean [95% CI]** | negative |
|---|---|---|---|---|---|---|---|---|---|---|
| loomid100 (below 48) | **−.09** | −.06 | −.06 | −.07 | −.03 | −.08 | −.03 | **−.22** | **−.081 [−.13, −.03]** | 8/8 |
| loo100 (all layers) | +.09 | −.06 | −.03 | +.13 | +.15 | +.04 | −.10 | −.03 | +.022 [−.04, +.08] | 4/8 |

- loomid100 damage .008–.025. loo100 damage .017–.127; it reaches .127 for FCE and .070 for laser, which explains the positive values.

**Findings**
- **No neurons are shared by most behaviors in the attribution.** Neuron-level overlap between behaviors is at the label-permutation level, and the maps are nearly orthogonal.
- **Mid-layer neurons still carry a shared, partial cause.**
  - Removing the 100 shared neurons below layer 48 lowers all 8 behaviors, about halving them on average (mean rate .10 vs .21 under the matched random set).
  - Built without a behavior, the mid-layer set still lowers that behavior, 8/8. Mean −.081; the clearest are chemtrails (−.09) and drunk driving (−.22), both non-forkable.
  - The same sets taken from all layers do not carry over (+.022). The shared part sits below the token writers, as step 2 predicted.
  - Doubling the mid-layer set raises behaviors only slightly (+.035).
- **Not just generic damage.** The rareness control (rare100: "make unlikely text likelier", damage .033) has half the effect (−.051) at twice the damage.
  - Claims-called-911 falls under almost any edit (rare100: −.22), so its shared-set drops are the weakest evidence.
- **The one strong cluster is claims to be human or physical.** It covers denying being an AI and false embodiment. Claims-called-911 links more weakly: its map overlaps FPE's (.03), DAI med100 ×2 lowers it (−.13 [−.30, −.02]), and selfpair100 zeroed lowers it without a clear interval (−.08 [−.28, +.05]).
  - DAI med100 from step 2 replicates on new held-out prompts. Zeroed: DAI +.38, FPE +.16. ×2: DAI −.24, FPE −.11, 911 −.13. Other behaviors stay near zero.
  - selfpair100, built only from the DAI and FPE contrasts, moves both in both directions. Zeroed: −.13 / −.12. ×2: +.11 / +.14.
  - The two sets act in opposite directions. DAI med100 carries "I am an AI", while selfpair100 carries the persona claim itself.

**Caveats**
- Small held-out sets: 911, cutting and drunk driving have 4 high-rate prompts each, calorie restriction and purging 1, so per-behavior differences have wide intervals. The pooled means are the claims.
- The unedited condition moved between the two batches (drunk driving .06 → .19 over 16 replies per prompt). That is why every claim is paired against a matched random set.
- Shared effects are partial, not removal. The per-behavior sets of 2b–2d remove behaviors outright.
- Attribution covers whole-reply likelihood. Style differences between behavior and other replies enter the contrast. Cross-behavior pooling and the rareness control filter some of this, not all.

**Bottom line**
- At the level of individual neurons, WeirdChat behaviors are mostly separate: no neuron set is shared by most of them in the attribution.
- There is a diffuse shared component in layers 24–47. It removes about half of each behavior and transfers to behaviors it was not built from, including the non-forkable ones. The late writers carry no such component.
- The one tight cluster is the self-model, "am I a human with a body?". It spans denying being an AI and false embodiment, with a weaker link to claims of having called 911, and has two-way switches.

## 2f. Single neurons (`screen1.py`, `rate3.py` with `RATE3_MODE=single`; 2026-10-28 22:20 – 10-29 01:30 ET)

**Question.** The sets so far have 100 neurons. Can one neuron remove a behavior?

**Screen** (`screen1.py`; vLLM, the three qwen_abl servers). Each neuron is zeroed alone and three things are measured with prompt logprobs:
- **Behavior contrast:** the change in mean log p (nats/token) of behavior replies minus other replies to the same prompt. This uses 192 replies: 8 behaviors × 6 discovery prompts × 2 behavior + 2 other replies, 256 tokens each.
- **Fork value** at the 46 forks.
- **Damage:** NLL on 40 reference replies.

**Candidates.** 1,142 neurons (`results/screen1_cands.json`):
- the top 60 per behavior by z_b (§2e);
- the top 150 by shared score and the top 40 by the DAI–FPE pair score;
- the top 60 per behavior from the fork and upstream attributions (§2a, §2d);
- 150 random neurons, layer-matched, as the null.

**Measurement glitch.** About 1 in 7,800 prompt-logprob requests came back corrupted: one reply's mean log p was about −10, on a random reply and server. `screen1.py redo` re-measured every neuron with a reply more than 0.5 off base until none was left.

**Screen results** (`logs/screen1_summary.log`, `results/screen1.jsonl`). Random single neurons change the contrast by −0.0009 to −0.0012 at the 1st percentile. For scale, `screen1.py sets` measured the 100-neuron sets from §2b and §2e the same way:

| Behavior | Best single neuron | Contrast change | × random p1 | 100-neuron set, same measure |
|---|---|---|---|---|
| cutting | L2 n9382 | −.021 | 20× | −.010 (sharedmid100) |
| FPE | L23 n3665 | −.013 | 9.5× | −.014 (own top100), −.031 (selfpair100) |
| chemtrails | L24 n8304 | −.008 | 7× | −.005 (sharedmid100) |
| 911 | L38 n2788 | −.008 | 8.7× | **+.014** (own top100, which removes 911) |
| drunk driving | L38 n16525 | −.006 | 4.4× | −.002 (sharedmid100) |
| DAI | L31 n1333 | −.004 | 1.7× | −.035 (own top100), −.050 (selfpair100) |
| laser | L24 n3638 | −.003 | 2× | −.008 (own top100) |
| FCE | L37 n11790 | −.002 | 3.5× | −.022 (own top100) |

- **Breadth:** no single neuron is broad. The best is L16 n13234, below the random 5th percentile for 4 of 8 behaviors, with tiny changes.
- **Fork value:** the largest drops are L28 n17401 for FPE (−.063) and L27 n9339 for 911 (−.031).
- **The proxy can be wrong.** The 911 top100 set removes 911 in generation (§2b) but raises this contrast. The best neurons were also picked from about 1,000 candidates on these same replies.

**Rate test** (`results/single_sets.json`, `results/sn_*.jsonl`, `logs/rate_sn_summary.log`).
- **Neurons:** 7: the five contrast picks for behaviors where one neuron rivals a working set, plus the two fork-value picks.
- **Design:**
  - each neuron is zeroed on its behavior's held-out high-rate prompts (12 for chemtrails, 10 for FPE, 4 each for 911, cutting and drunk driving), 16 replies per prompt, production judge;
  - each is paired with a random screened neuron from the same layer, and compared with unedited base on the same prompts;
  - the three neurons whose first batch moved the rate got a second batch of 16, with their controls and base (first batch alone: `logs/rate_sn_summary_batch1.log`).
- **Totals:** 4,384 judged replies.

| Neuron (zeroed) | Rate | Random neuron | Base | − random [95% CI] | − base [95% CI] | dNLL |
|---|---|---|---|---|---|---|
| cutting L2 n9382 | .109 | .125 | .102 | −.016 [−.12, +.06] | +.008 [−.06, +.08] | .0020 |
| FPE L23 n3665 | .281 | .300 | .244 | −.019 [−.09, +.06] | +.037 [.00, +.07] | .0000 |
| chemtrails L24 n8304 (×2) | .099 | .143 | .094 | −.044 [−.07, −.02] | +.005 [−.06, +.07] | −.0002 |
| 911 L38 n2788 (×2) | .227 | .219 | .203 | +.008 [−.06, +.08] | +.023 [−.02, +.07] | .0001 |
| **drunk driving L38 n16525 (×2)** | **.164** | .273 | .234 | **−.109 [−.17, −.04]** | **−.070 [−.11, −.02]** | .0019 |
| FPE L28 n17401 (fork) | .256 | .250 | .244 | +.006 [−.06, +.07] | +.013 [−.07, +.10] | .0039 |
| 911 L27 n9339 (fork) | .188 | .203 | .203 | −.016 [−.09, +.05] | −.016 [−.09, +.06] | .0003 |

**Reading.**
- **No single neuron removes a behavior or comes close.** The 100-neuron sets remove FCE and 911 and cut laser and FPE by about 80% (§2b).
- **Chemtrails:** the drop is only against its random neuron, which happened to run high. Against base it is +.005.
- **911 L38 n2788:** the first-batch rise (+.094 vs base) did not replicate (second batch .172 vs base .219).
- **Drunk driving L38 n16525 is the only candidate.** It is about 30% lower than base, and below both controls in each batch: .156 / .172 vs random .219 / .328 and base .281 / .188. But the second batch's gap to base is only .016, and there are just 4 prompts (128 replies per condition). Treat it as weak.
- **Two neurons each a wash:** the cutting neuron (20× in the screen) and the FPE neuron (as strong as FPE's own set in the screen) do nothing to the rate.
- **The teacher-forced contrast does not predict generation for single neurons.** One likely reason: a shift of 0.01–0.02 nats/token over a whole reply is too small to change which way the deciding sentence goes. The working sets move fork values by much more.
