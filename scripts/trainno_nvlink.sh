#!/bin/bash

MASTER_ADDR="127.0.0.1"                    
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     
NPROC_PER_NODE=2

MODEL_PATH="/data/model/Qwen2.5-VL-7B-Instruct"  
VGGT_MODEL_PATH="/data/model/VGGT-1B"
OUTPUT_DIR="/data/model/JanusVLN_Base/1"                  
CACHE_DIR="./cache"                        
mkdir -p $OUTPUT_DIR
DATASETS="train_r2r_rxr" 
# CUDA 路径
export CUDA_HOME=/data/miniconda3/envs/vln
export PATH=$CUDA_HOME/bin:$PATH


# NCCL 优化
export NCCL_P2P_LEVEL=SYS
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_BUFFSIZE=2097152
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=0

# 缓解显存碎片，避免大块分配失败
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ==========================================
# DeepSpeed ZeRO-2 + CPU Offload optimizer
# 7B 模型 optimizer states ~28GB/卡，必须 offload 到 CPU
# ==========================================
cat > /tmp/zero2_offload.json << 'EOF'
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
EOF

# ==========================================
# GPU 显存 & 利用率监控（后台，每30秒记录一次）
# ==========================================
GPU_LOG="${OUTPUT_DIR}/gpu_monitor.log"
echo "timestamp,gpu_idx,mem_used_MiB,mem_total_MiB,gpu_util_pct,temp_C,power_W" > "$GPU_LOG"
(
  while true; do
    TS=$(date +'%F_%T')
    nvidia-smi \
      --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
      --format=csv,noheader,nounits \
      | sed "s/^/${TS}, /" \
      | tr -d ' ' \
      >> "$GPU_LOG" 2>/dev/null
    sleep 30
  done
) &
MONITOR_PID=$!
echo "[GPU Monitor] PID=$MONITOR_PID, 日志 → $GPU_LOG"

# 捕获退出信号，确保 Ctrl+C 也能停止监控进程
cleanup() {
  echo "[GPU Monitor] 停止监控 (PID=$MONITOR_PID)"
  kill "$MONITOR_PID" 2>/dev/null
}
trap cleanup EXIT

# ==========================================
# 断点续练：自动检测最新 checkpoint
# ==========================================
RESUME_ARG=""
LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST_CKPT" ]; then
  echo "[Resume] 检测到检查点，从 $LATEST_CKPT 恢复训练"
  RESUME_ARG="--resume_from_checkpoint $LATEST_CKPT"
else
  echo "[Resume] 未找到检查点，从头开始训练"
fi

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/qwen_vl/train/train_qwen.py \
    --model_name_or_path $MODEL_PATH \
    --vggt_model_path $VGGT_MODEL_PATH \
    --tune_mm_llm True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --dataset_use $DATASETS \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --bf16 True \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 1e-6 \
    --model_max_length 20000 \
    --data_flatten False \
    --max_pixels $((576*28*28)) \
    --min_pixels $((16*28*28)) \
    --base_interval 2 \
    --video_max_frames 8 \
    --video_min_frames 4 \
    --video_max_frame_pixels $((1664*28*28)) \
    --video_min_frame_pixels $((256*28*28)) \
    --num_train_epochs 1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 200 \
    --save_total_limit 3 \
    --deepspeed /tmp/zero2_offload.json \
    --gradient_checkpointing \
    --dataloader_num_workers 16\
    --dataloader_pin_memory True \
    --group_by_modality_length True \
    --seed 42 \
    --max_steps 10000 \
    --report_to none \
    --reference_frame first \
    $RESUME_ARG \
    2>&1 | tee -a ${OUTPUT_DIR}/train.log
