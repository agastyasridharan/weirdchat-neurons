#!/bin/bash
# Wait for discovery Stage A judging, then run the full fork search (thinking-off judge).
P=/data/agastyas/weirdchat-neurons
cd $P/code
# university proxy for remote (Runpod) judge servers; local servers bypass it
export http_proxy=http://www-cache.rz.ruhr-uni-bochum.de:80 https_proxy=http://www-cache.rz.ruhr-uni-bochum.de:80 no_proxy=localhost,127.0.0.1,::1
until WCN_ROOT=$P $P/cenv/bin/python disc_done.py off >/dev/null 2>&1; do sleep 20; done
WCN_SERVERS=$P/code/servers_fork.json WCN_ROOT=$P exec $P/cenv/bin/python forks.py --off > $P/logs/forks_off.log 2>&1
