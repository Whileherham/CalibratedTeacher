#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-$RANDOM}

PYTHONPATH="$(dirname $0)/..":"/mmlabtoolbox/mmdetection":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_tide.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
