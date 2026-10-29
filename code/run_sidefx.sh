#!/bin/bash
# Phase 2c side effects + amplification: qwen_abl servers on GPUs 0-2, Gemma judges on 3-7 (servers_sidefx.json), then the summary.
cd /data/agastyas/weirdchat-neurons/code
export WCN_SERVERS=/data/agastyas/weirdchat-neurons/code/servers_sidefx.json WCN_ROOT=/data/agastyas/weirdchat-neurons
PY=/data/agastyas/weirdchat-neurons/cenv/bin/python
$PY sidefx.py run && $PY sidefx.py summary
