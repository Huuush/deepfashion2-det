#!/usr/bin/env bash

set -x

#PARTITION=$1
#JOB_NAME=$2
#CONFIG=$3
#WORK_DIR=$4
#GPUS=${GPUS:-8}
#GPUS_PER_NODE=${GPUS_PER_NODE:-8}
#CPUS_PER_TASK=${CPUS_PER_TASK:-5}
#SRUN_ARGS=${SRUN_ARGS:-""}
#PY_ARGS=${@:5}

PARTITION=sl-mmdet
JOB_NAME=vfnetx-r2-101
CONFIG=/home/vmip/mmdetection/configs/vfnet/vfnet_r2_101_fpn_2x_cloth.py
WORK_DIR=/home/vmip/slurm_nfs/mmdet_work_dir/vfnetx-r2-101-mdconvc3-c5-1x
GPUS=4
GPUS_PER_NODE=2
CPUS_PER_TASK=4
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

PYTHONPATH=/home/vmip/anaconda3/bin/python:$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}
