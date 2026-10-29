"""Shared utilities for the WeirdChat x neurons project (Qwen3.6-27B).

Data: Transluce/WeirdChat v1.0.1 shards for qwen/qwen3.6-27b, in data/wc_q27/.
Subject sampling matches WeirdChat: no system prompt, thinking off, temperature 1, 1024-token
response cap; top_k/top_p come from the model's generation_config (vLLM and SGLang default).
Judge: WeirdChat's verbatim rubric prompt (wcref.judge) on Gemma 4 31B (FP8), thinking on.
"""

import asyncio
import glob
import hashlib
import json
import os
import random
import time
from pathlib import Path

import httpx
import pandas as pd

P = Path(os.environ.get("WCN_ROOT", "/data/agastyas/weirdchat-neurons"))
DATA = P / "data"
RES = P / "results"
QWEN_SNAP = "/data/hf_cache/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
MAX_RESP = 1024
SPLIT_SALT = "wcn-split-v1"


def _struct(x):
    if isinstance(x, str):
        import ast
        return ast.literal_eval(x)
    return x


def load_prompts() -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(DATA / "wc_q27/*prompts*")))])
    df["user_text"] = df.messages.apply(lambda m: [x for x in m if x["role"] == "user"][0]["content"])
    df["wc_n"] = df.match_summary.apply(lambda m: m["n_samples"])
    df["wc_k"] = df.match_summary.apply(lambda m: m["n_matched"])
    return df.drop(columns=["messages"]).reset_index(drop=True)


def load_transcripts() -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(DATA / "wc_q27/*transcripts*")))])
    df["judgment"] = df.judgment.apply(_struct)
    df["messages"] = df.messages.apply(_struct)
    df["wc_match"] = df.judgment.apply(lambda j: j.get("match"))
    return df.reset_index(drop=True)


def load_rubrics() -> dict:
    r = pd.read_parquet(DATA / "rubrics.parquet")
    return {row.behavior_id: _struct(row.transcript_rubric)["text"] for row in r.itertuples()}


def make_split(prompts: pd.DataFrame) -> dict:
    """Discovery/held-out split, frozen before any analysis. Unit = pattern (so held-out
    prompts are new phrasings, not paraphrases of discovery ones), alternating within each
    behavior by salted hash. Behaviors with < 4 patterns split by prompt instead."""
    h = lambda s: hashlib.sha256(f"{SPLIT_SALT}|{s}".encode()).hexdigest()
    out = {}
    for b, g in prompts.groupby("behavior_id"):
        pats = sorted(g.pattern_id.unique(), key=h)
        if len(pats) >= 4:
            side = {p: ("discovery" if i % 2 == 0 else "heldout") for i, p in enumerate(pats)}
            for r in g.itertuples():
                out[r.prompt_id] = {"split": side[r.pattern_id], "unit": "pattern"}
        else:
            for i, pid in enumerate(sorted(g.prompt_id, key=h)):
                out[pid] = {"split": "discovery" if i % 2 == 0 else "heldout", "unit": "prompt"}
    return out


def get_split() -> dict:
    f = DATA / "split.json"
    if not f.exists():
        f.write_text(json.dumps(make_split(load_prompts()), indent=1))
    return json.loads(f.read_text())


_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(QWEN_SNAP)
    return _TOK


def prompt_ids(user_text: str) -> list[int]:
    ids = tok().apply_chat_template([{"role": "user", "content": user_text}], add_generation_prompt=True,
                                    enable_thinking=False, tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return list(ids)


# ---------------------------------------------------------------- server pools

def servers(role: str) -> dict[str, dict]:
    """{url: {max, key}} for one role. Entries are either "role" (local vLLM) or
    {"role", "max": in-flight cap, "key": API key} (remote, e.g. behind the Runpod proxy)."""
    cfg = json.loads(Path(os.environ.get("WCN_SERVERS", P / "code/servers.json")).read_text())
    out = {}
    for u, r in cfg.items():
        r = {"role": r} if isinstance(r, str) else r
        if r["role"] == role:
            out[u] = {"max": r.get("max"), "key": r.get("key")}
    return out


class Pool:
    """Least-loaded dispatch over identical vLLM servers, bounded in-flight per server. The servers
    file is re-read every 30 s, so servers can be added or removed while a run is going."""

    def __init__(self, role: str, per_server: int):
        self.role, self.per_server = role, per_server
        self.cfg = servers(role)
        self.urls = list(self.cfg)
        self.load = {u: 0 for u in self.urls}
        self.dead = {}
        self.inflight, self.cond = 0, asyncio.Condition()
        self.reloaded = time.time()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(3600.0), limits=httpx.Limits(max_connections=8192))

    def _reload(self):
        if time.time() - self.reloaded < 30:
            return
        self.reloaded = time.time()
        try:
            cfg = servers(self.role)
        except (OSError, ValueError, KeyError):
            return
        if cfg and cfg != self.cfg:
            log(f"pool {self.role}: {len(self.cfg)} -> {len(cfg)} servers")
            for u in cfg:
                self.load.setdefault(u, 0)
            self.cfg, self.urls = cfg, list(cfg)

    def cap(self, u):
        return self.cfg.get(u, {}).get("max") or self.per_server

    def capacity(self):
        return sum(self.cap(u) for u in self.urls)

    async def post(self, path: str, body: dict) -> dict:
        """Least-loaded healthy server (load relative to its cap); a server that fails at the
        transport level or with a 5xx is benched for 60 s and the request moves to another
        (requests are idempotent up to sampling)."""
        self._reload()
        async with self.cond:
            await self.cond.wait_for(lambda: self.inflight < self.capacity())
            self.inflight += 1
        try:
            for attempt in range(30):
                now = time.time()
                live = [x for x in self.urls if self.dead.get(x, 0) < now] or self.urls
                u = min(live, key=lambda x: self.load[x] / self.cap(x))
                self.load[u] += 1
                key = self.cfg.get(u, {}).get("key")
                try:
                    r = await self.client.post(u + path, json=body,
                                               headers={"Authorization": f"Bearer {key}"} if key else None)
                    r.raise_for_status()
                    return r.json()
                except (httpx.TransportError, httpx.HTTPStatusError) as e:
                    if not isinstance(e, httpx.HTTPStatusError) or e.response.status_code >= 500:
                        self.dead[u] = time.time() + 60
                    if attempt == 29:
                        raise
                    await asyncio.sleep(min(2 + 2 * attempt, 30))
                finally:
                    self.load[u] -= 1
        finally:
            async with self.cond:
                self.inflight -= 1
                self.cond.notify_all()


async def generate(pool: Pool, ids: list[int], n: int, max_tokens: int, seed: int | None = None) -> list[dict]:
    """Sample n continuations of a token-id prompt. Returns [{ids, text, finish}]."""
    body = {"model": "qwen", "prompt": ids, "n": n, "max_tokens": max_tokens, "temperature": 1.0,
            "return_token_ids": True, "skip_special_tokens": True}
    if seed is not None:
        body["seed"] = seed
    out = await pool.post("/v1/completions", body)
    return [{"ids": c.get("token_ids"), "text": c["text"], "finish": c["finish_reason"]} for c in out["choices"]]


async def next_token_logprobs(pool: Pool, ids: list[int], k: int = 20) -> dict:
    """Top-k next-token logprobs (raw model distribution) at the end of a token-id prompt."""
    body = {"model": "qwen", "prompt": ids, "max_tokens": 1, "temperature": 1.0, "logprobs": k,
            "top_k": -1, "top_p": 1.0, "return_token_ids": True, "skip_special_tokens": False,
            "return_tokens_as_token_ids": True}
    out = await pool.post("/v1/completions", body)
    return out["choices"][0]


RUBRICS = None


def rubric_first_prompt(rubric: str, messages) -> str:
    """Same text blocks as the production template, reordered so the per-behavior part
    (instructions, format, rubric) is a shared prefix that vLLM can cache across transcripts."""
    from wcref.judge import DEFAULT_TRANSCRIPT_JUDGE_TEMPLATE, RUBRIC_RESPONSE_SCHEMA, render_output_spec, render_transcript
    full = DEFAULT_TRANSCRIPT_JUDGE_TEMPLATE.substitute(transcript="@@T@@", rubric=rubric,
                                                       json_schema_section=render_output_spec(RUBRIC_RESPONSE_SCHEMA))
    head, rest = full.split("<transcript>\n@@T@@\n</transcript>\n\n", 1)
    rub, tail = rest.split("\n</rubric>\n\n", 1)
    return head + rub + "\n</rubric>\n\n<transcript>\n" + render_transcript(messages) + "\n</transcript>\n\n" + tail


async def judge(pool: Pool, behavior_id: str, user_text: str, response: str, thinking: bool = True,
                order: str = "prod") -> dict:
    """WeirdChat transcript judge. Returns {match: bool|None, explanation, n_out}."""
    from wcref.judge import build_judge_prompt, parse_json_response  # lazy: GPU envs lack docent
    from wcref.types import Message
    global RUBRICS
    if RUBRICS is None:
        RUBRICS = load_rubrics()
    msgs = [Message(role="user", content=user_text), Message(role="assistant", content=response)]
    prompt = (build_judge_prompt(RUBRICS[behavior_id], msgs) if order == "prod"
              else rubric_first_prompt(RUBRICS[behavior_id], msgs))
    body = {"model": "gemma", "messages": [{"role": "user", "content": prompt}], "temperature": 1.0,
            "max_tokens": 8192, "chat_template_kwargs": {"enable_thinking": thinking}}
    last = None
    for _ in range(3):
        out = await pool.post("/v1/chat/completions", body)
        msg = out["choices"][0]["message"]
        try:
            j = parse_json_response(msg.get("content") or "")
            return {"match": bool(j["match"]), "explanation": str(j.get("explanation", "")),
                    "n_out": out["usage"]["completion_tokens"], "n_in": out["usage"]["prompt_tokens"]}
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            last = e
    return {"match": None, "explanation": f"judge_error: {last}", "n_out": 0, "n_in": 0}


# ---------------------------------------------------------------- persistence

class JsonlStore:
    """Append-only JSONL keyed by 'key'; reruns skip finished keys."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.done[d["key"]] = d
        self.f = open(self.path, "a")

    def add(self, rec: dict):
        self.done[rec["key"]] = rec
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
