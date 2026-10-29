#!/bin/bash
# Fresh thinking-on confirmation of the 46 Phase 2 forks (Qwen on GPU 7, Gemma judges on 4-6), then the summary.
cd /data/agastyas/weirdchat-neurons/code
export WCN_SERVERS=/data/agastyas/weirdchat-neurons/code/servers_confirm.json WCN_ROOT=/data/agastyas/weirdchat-neurons
until curl -s -m 5 localhost:8107/v1/models | grep -q qwen; do sleep 10; done
PY=/data/agastyas/weirdchat-neurons/cenv/bin/python
$PY confirm_on.py judge && $PY confirm_on.py summary
