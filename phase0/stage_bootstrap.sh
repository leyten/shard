#!/bin/bash
# stage box: shard runtime deps + the gpt-oss-120b target weights. Detached + logged so
# the launching ssh can return; poll /root/stage_ready.txt for completion.
set -e
bash /root/setup_box.sh > /root/setup.log 2>&1
pip install --break-system-packages -q cryptography >> /root/setup.log 2>&1   # wire.py transport
echo "DEPS_DONE rc=$?" >> /root/setup.log
# authenticated HF pull avoids the anon throttle (~10x faster on the 57GB shards). Token is NOT in this
# (public) repo — push ~/.hf_token to the box alongside this script; get_model.py reads HF_TOKEN from env.
export HF_TOKEN=$(cat /root/.hf_token 2>/dev/null)
python3 /root/get_model.py openai/gpt-oss-120b /root/models/gpt-oss-120b > /root/dl120.log 2>&1
R=$?
echo "120b rc=$R" > /root/stage_ready.txt
du -sh /root/models/gpt-oss-120b >> /root/stage_ready.txt 2>&1
nvidia-smi --query-gpu=memory.total --format=csv,noheader >> /root/stage_ready.txt 2>&1
