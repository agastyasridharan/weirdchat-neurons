#!/bin/bash
# Phase 3c: shared-set rates (qwen_abl on GPUs 0-2, Gemma judges on 3-7), then the summary.
cd /data/agastyas/weirdchat-neurons/code
export WCN_SERVERS=/data/agastyas/weirdchat-neurons/code/servers_sh.json WCN_ROOT=/data/agastyas/weirdchat-neurons
until curl -s -m 5 localhost:8101/v1/models | grep -q qwen; do sleep 10; done
PY=/data/agastyas/weirdchat-neurons/cenv/bin/python
$PY rate3.py run && $PY rate3.py summary
