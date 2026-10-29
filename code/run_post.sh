#!/bin/bash
# After the fork search: held-out Stage A (thinking-off), then thinking-on re-judging of fork confirmations.
cd /data/agastyas/weirdchat-neurons/code
export WCN_SERVERS=/data/agastyas/weirdchat-neurons/code/servers_post.json WCN_ROOT=/data/agastyas/weirdchat-neurons
/data/agastyas/weirdchat-neurons/cenv/bin/python stage_a.py judge --off > /data/agastyas/weirdchat-neurons/logs/stage_a_judge_off.2.log 2>&1 &
/data/agastyas/weirdchat-neurons/cenv/bin/python verify_on.py 80 > /data/agastyas/weirdchat-neurons/logs/verify_on.log 2>&1
wait
