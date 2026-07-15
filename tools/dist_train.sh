#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-26119}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


# 设置显存使用阈值（单位：MiB）

pkill -9 python

pkill -9 -f train.py

nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits



PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
