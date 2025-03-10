#!/bin/bash
set -e
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $(dirname $( dirname ${CURRENT_DIR})))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-250217

echo PYTHONPATH:$PYTHONPATH


# cmd="torchrun ${DISTRIBUTED_ARGS} hf2mcore_qwen2_vl_test.py \
#     --load ${SOURCE_CKPT_PATH} \
#     --save ${TARGET_CKPT_PATH} \
#     --target-tensor-model-parallel-size ${TP} \
#     --target-pipeline-model-parallel-size ${PP} \
#     --micro-batch-size 1 \
#     --save-interval 1 \
#     --swiglu \
#     --num-layers ${NUM_LAYERS} \
#     --hidden-size ${HIDDEN_SIZE} \
#     --ffn-hidden-size ${INTERMEDIATE_SIZE} \
#     --num-attention-heads ${NUM_ATTN_HEADS} \
#     --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
#     --seq-length 1 \
#     --no-async-tensor-model-parallel-allreduce \
#     --patch-tokenizer-type Qwen2Tokenizer \
#     --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
#     --no-bias-swiglu-fusion \
#     --no-rope-fusion \
#     --use-rotary-position-embeddings \
#     --disable-bias-linear \
#     --add-qkv-bias \
#     --normalization RMSNorm \
#     --norm-epsilon ${RMS_NORM_EPS} \
#     --use-mcore-models \
#     --attention-dropout 0.0 \
#     --hidden-dropout 0.0 \
#     --rotary-base 1000000 \
#     --spatial-merge-size 2 \
#     ${safe_options} \
#     ${te_options} \
#     ${convert_options} \
#     ${pr_options} \
#     ${cpu_options} \
#     ${tie_option} \
#     ${gqa_options} \
#     ${uneven_split_option} \
#     ${vp_options}"

cmd="torchrun ${DISTRIBUTED_ARGS} hf2mcore_qwen2_vl_test.py"

echo $cmd
eval $cmd
