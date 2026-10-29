# WeirdChat × neurons: where Qwen3.6-27B decides to behave weirdly, and which neurons decide it

Code, results and write-ups for a project connecting Transluce's [WeirdChat](https://transluce.org/weirdchat) behaviors to
MLP neurons in Qwen3.6-27B. Work in progress (October 2026).

> **Content warning.** `results/` contains raw model replies to WeirdChat prompts, including replies the judge
> flagged for cutting instructions, purging advice, extreme calorie restriction, recommending drunk driving and
> unsolicited sexual advances. They are kept for reproducibility and analysis.

- **Write-ups:** [`PHASE1.md`](PHASE1.md) (forks: where in a reply the behavior is decided) and [`PHASE2.md`](PHASE2.md)
  (neurons: attribution, ablation, side effects, upstream sets, neurons shared across behaviors).
- **Dashboard:** [`site/index.html`](site/index.html) (self-contained; open it in a browser), built by `site/build.py`
  from `site/template.html` and the `results/site_*.json` exports.

## Findings so far

- **Phase 1.** Usually a sentence decides a behavior, not a token: one token raises the chance of the behavior by
  ≥ 0.5 in 21% of 410 replies. When one token does decide it, it is an ordinary sampling choice. 46 forks
  (confirmed with fresh samples and the production judge) form the Phase 2 set.
- **Phase 2a–c.** About 100 MLP neurons (of 1.1M) control four of five behaviors tested. Zeroing them removes
  fabricated code execution and claims-called-911 and cuts laser-at-aircraft and false embodiment by about 80%,
  also on held-out phrasings. Damage is ≤ 0.02 nats/token, MMLU/GSM8K are unchanged, and doubling the same neurons
  causes the behavior. Matched random neurons do nothing.
- **Phase 2d.** For several behaviors the decision is made upstream of the late token-writing neurons. 100 neurons in
  layers 19–47 act as a two-way switch for denying being an AI.
- **Phase 2e.** No neuron set is shared by most behaviors in the attribution, but a diffuse mid-layer (layers < 48)
  component is. Built without a behavior, it still lowers that behavior (8/8, including behaviors with no single
  fork). The one tight cluster is the self-model: denying being an AI and false physical embodiment.

## Layout

| Path | Contents |
|---|---|
| `code/` | Every script, in pipeline order below. `wcref/` is a verbatim copy of WeirdChat's judge (MIT, see `code/wcref/LICENSE`). |
| `results/` | Result files (JSON / JSONL; the largest JSONL files are gzipped, `gunzip results/*.gz` before running code) |
| `logs/` | Analysis logs and summary tables (`rate2.log`, `uprate.log`, `sidefx.log`, `rate3_summary.log`, ...) |
| `data/split.json` | The frozen discovery / held-out split (by pattern) |
| `site/` | Dashboard template, build script and built page |
| `results_neuron_logits.txt`, `site_trim.json` | Inputs to `site/build.py` (neuron logit readouts, trimmed Phase 1 export) |

Not included (too large or third-party; regenerable with the code): per-fork attribution arrays (`results/attrib/`,
`results/upstream/`, ~640 MB), per-reply attribution arrays (`results/contrast/`, 4.7 GB), the per-behavior score maps
(`results/shared_z.npz`), the WeirdChat dataset files and the MMLU / GSM8K copies.

## Pipeline

| Step | Scripts | Main outputs in `results/` |
|---|---|---|
| Judge validation | `judge_val.py`, `judge_parity.py`, `compare_judges.py` | `judge_val.jsonl`, `judge_parity.jsonl` |
| Stage A: 403 prompts × 64 replies | `stage_a.py`, `analyze_stage_a.py` | `stage_a_gen.jsonl`, `stage_a_judge_{off,on}.jsonl` |
| Fork search | `forks.py`, `run_forks.sh`, `full_judge.py`, `verify_on.py`, `analyze_forks.py` | `forks_off.jsonl`, `forks_samples_off.jsonl`, `full_off.jsonl`, `verify_on.jsonl` |
| Production-judge re-scoring and confirmation | `rescore_on.py`, `confirm_on.py` | `rescore_on.jsonl`, `forks_on_rescored.jsonl`, `confirm_on_samples.jsonl`, `phase2_forks.jsonl` |
| 2a attribution at forks | `phase2_check.py`, `attrib.py`, `analyze_attrib.py`, `neuron_logits.py` | `phase2_vllm_logprobs.jsonl`, (`attrib/`) |
| 2b rate ablation | `sets.py`, `ablate_ext.py`, `rate.py` | `ablation_sets.json`, `ref_texts.json`, `rate_{gen,judge,cond}.jsonl` |
| 2c side effects, amplification | `legit.py`, `sidefx.py` | `sfx_{gen,judge,cond}.jsonl`, `site_sidefx.json` |
| 2d upstream | `upstream.py`, `upsets.py`, `uprate.py`, `heldfork.py` | `upstream_sets.json`, `up_{gen,judge,cond}.jsonl`, `forks_on_dai_held.jsonl`, `heldfork_dai.json` |
| 2e shared across behaviors | `contrast_judge.py`, `contrast.py`, `sets3.py`, `rate3.py` | `contrast_judge_on.jsonl`, `shared_sets.json`, `shared_stats.json`, `sh_{gen,judge,cond}.jsonl`, `site_shared.json` |
| Dashboard exports | `export_site.py`, `export_phase2.py`, `*.py export` | `site_*.json` |

`*_gen.jsonl` hold generated replies, `*_judge.jsonl` the judge verdicts (with the judge's explanation), and
`*_cond.jsonl` per-condition damage (NLL on reference replies) and fork values. Keys are
`<condition>|<mode>|<prompt_id>[|<reply index>]`.

## Setup

- **Subject:** Qwen3.6-27B, bf16, vLLM 0.30.0, no system prompt, thinking off, temperature 1, top-k 20, top-p 0.95,
  1,024-token cap (WeirdChat's sampling).
- **Judge:** Gemma 4 31B, FP8 weights, WeirdChat's verbatim judge prompt and rubrics; thinking on (production) unless
  noted.
- **Ablation:** `code/ablate_ext.py` is a vLLM worker extension that zeros or scales `down_proj` columns (neurons) or
  `o_proj` / `out_proj` column blocks (attention heads) in place, restorably, via `/collective_rpc`
  (`VLLM_SERVER_DEV_MODE=1`, prefix caching off). Served by `code/serve.sh qwen_abl GPU PORT`.
- **Attribution:** Hugging Face transformers, bf16, activation × gradient on the `down_proj` input.

## Running it

`code/common.py` expects `WCN_ROOT` (default `/data/agastyas/weirdchat-neurons`) with `data/`, `results/`,
`logs/`. Put the WeirdChat v1.0.1 files from the [Hugging Face dataset](https://huggingface.co/datasets/Transluce/WeirdChat)
in `data/`: `patterns.parquet`, `rubrics.parquet`, and the Qwen3.6-27B prompt and transcript files in `data/wc_q27/`.
`code/serve.sh` has cluster-specific paths (model snapshots, environments) to adapt. Drivers read the list of servers
from the JSON file named in `WCN_SERVERS` (`{"http://host:port": "qwen" | "qwen_abl" | "gemma"}`).

Two Python environments were used: a client environment (numpy, pandas, pyarrow, httpx, transformers tokenizer) for
the drivers, and a torch + transformers environment (with flash-linear-attention kernels) for attribution.

## License

Code and results in this repository: MIT (see `LICENSE`). `code/wcref/` is copied from
[TransluceAI/WeirdChat](https://github.com/TransluceAI/WeirdChat), and the prompts embedded in `results/` (as token ids
and in judge inputs) come from the [WeirdChat dataset](https://huggingface.co/datasets/Transluce/WeirdChat); both are
© 2026 Transluce under the MIT license, reproduced in `LICENSE-WEIRDCHAT`.
