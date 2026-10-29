#!/bin/bash
# Phase 2b rate ablations: qwen_abl servers on GPUs 0-1, Gemma judges on 2-7 (servers_rate.json), then the summary.
cd /data/agastyas/weirdchat-neurons/code
export WCN_SERVERS=/data/agastyas/weirdchat-neurons/code/servers_rate.json WCN_ROOT=/data/agastyas/weirdchat-neurons
PY=/data/agastyas/weirdchat-neurons/cenv/bin/python
$PY rate.py run && $PY rate.py summary
