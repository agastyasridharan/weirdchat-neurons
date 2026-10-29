#!/bin/bash
# Phase 2d upstream rate run: qwen_abl server on GPU 0 (GPUs 1-2 serve the held-out fork search), Gemma judges on 3-7 (servers_up.json).
cd /data/agastyas/weirdchat-neurons/code
export WCN_SERVERS=/data/agastyas/weirdchat-neurons/code/servers_up.json WCN_ROOT=/data/agastyas/weirdchat-neurons
PY=/data/agastyas/weirdchat-neurons/cenv/bin/python
$PY uprate.py run && $PY uprate.py summary && $PY uprate.py export
