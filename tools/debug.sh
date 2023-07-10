#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-$RANDOM}

PYTHONPATH="$(dirname $0)/..":"/mmlabtoolbox/mmdetection":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_debug.py $CONFIG --launcher pytorch ${@:3}
