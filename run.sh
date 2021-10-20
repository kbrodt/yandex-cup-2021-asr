#!/usr/bin/env sh


set -eu


GPU=${GPU:-0,1,2,3}
PORT=${PORT:-29501}
N_GPUS=4

IN_CHANNELS=1

OPTIM=fusedadam
LR=0.001

SCHEDULER=cosa
T_MAX=25

N_EPOCHS=376

MODEL_NAME=efficientnet-b2
CHECKPOINT=./chkps/"${MODEL_NAME}"
# LOAD="${CHECKPOINT}"/model_last.pth
ALPHA=1

# CUDA_VISIBLE_DEVICES="${GPU}" python3 \
CUDA_VISIBLE_DEVICES="${GPU}" python3 -m torch.distributed.launch --nproc_per_node="${N_GPUS}" --master_port="${PORT}" \
    ./train.py \
        --model-name "${MODEL_NAME}" \
        --in-channels "${IN_CHANNELS}" \
        --optim "${OPTIM}" \
        --learning-rate "${LR}" \
        --scheduler "${SCHEDULER}" \
        --T-max "${T_MAX}" \
        --num-epochs "${N_EPOCHS}" \
        --checkpoint-dir "${CHECKPOINT}" \
        --distributed \
        --fp16 \
        --mixup \
        --alpha "${ALPHA}" \

        # --pseudo \
        # --ft \
        # --load "${LOAD}" \
        # --resume \
