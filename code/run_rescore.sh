#!/bin/bash
# Thinking-on re-scoring of Phase 2 candidate forks (fresh off-jump >= 0.25), then the summary.
cd /data/agastyas/weirdchat-neurons/code
export WCN_SERVERS=/data/agastyas/weirdchat-neurons/code/servers_post.json WCN_ROOT=/data/agastyas/weirdchat-neurons
PY=/data/agastyas/weirdchat-neurons/cenv/bin/python
$PY rescore_on.py judge && $PY rescore_on.py summary
