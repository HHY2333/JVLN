#!/bin/bash
# =============================================================================
# train_with_eval.sh
# 功能：每隔 EVAL_INTERVAL 训练步后，对所有保留的 checkpoint 做快速批量评估，
#       记录最佳 checkpoint（按 val_unseen SPL），然后自动继续训练。
# 用法：cd /home/user/HuHuaiyang/JVLN && bash scripts/train_with_eval.sh
# =============================================================================

set -euo pipefail

# ─── 可调参数 ────────────────────────────────────────────────────────────────
# 训练总步数自动按“数据集样本数 / 全局 batch size × TARGET_EPOCHS”计算
TARGET_EPOCHS=1            # 训练轮数（按样本自动换算成总步数）
EVAL_INTERVAL=1000         # 每隔多少步评估一次（必须是 save_steps 200 的倍数）
QUICK_EPISODES=16         # 快速评估每个 checkpoint 的 episode 总数（跨所有 GPU）
QUICK_MAX_STEPS=50         # 快速评估每个 episode 的最大步数（缩短评估耗时）
EVAL_SPLIT="val_seen"      # 评估集（val_seen / val_unseen）
NPROC=2                    # GPU 数量（训练和评估共用）
DATALOADER_NUM_WORKERS=16   # 过大时容易把主机内存/IO 压爆，触发 SIGKILL
# 序列打包（data_flatten=True）下：每个 GPU 每步处理的样本数。
# 数据集单样本均值 ~1517 tokens，p90 ~2483 tokens，model_max_length=20000。
# batch_size=8 时打包长度 p90≈19864，截断率≈8%；batch_size=12 截断率≈44%（丢数据）。
# 若要零截断且吞吐最大，选 batch=8，等效全局 tokens/step = 8×2×20000≈320k。
TRAIN_BATCH_SIZE=8          # per_device_train_batch_size（打包模式下从 12 降至 8，避免截断）
GRAD_ACC_STEPS=2           # gradient_accumulation_steps（补回全局 batch：8×2×2GPU=32 样本/step）
# 评估视频：true=保存，false=不保存（快速评估建议关闭以节省磁盘和时间）
SAVE_VIDEO=true
SAVE_VIDEO_RATIO=1       # 保存视频的 episode 比例（仅 SAVE_VIDEO=true 时生效，0~1）

# ─── 路径配置（与 trainzero3.sh 保持一致）────────────────────────────────────
MASTER_ADDR="127.0.0.1"
MODEL_PATH="/data/model/Qwen2.5-VL-3B-Instruct"
VGGT_MODEL_PATH="/data/model/VGGT-1B"
OUTPUT_DIR="/data/model/JanusVLN_Base/3b"
CACHE_DIR="./cache"
CONFIG="config/vln_r2r.yaml"
DATASETS="train_r2r_rxr"
# checkpoint 缺失的配置文件来源目录（含 chat_template.json / configuration.json / preprocessor_config.json）
CONFIG_SRC="/data/model/JanusVLN_Base/1"

BEST_CKPT_FILE="${OUTPUT_DIR}/best_checkpoint.txt"
EVAL_SUMMARY="${OUTPUT_DIR}/quick_eval_summary.jsonl"
mkdir -p "$OUTPUT_DIR"

# ─── 自动统计数据集样本数并换算总步数 ────────────────────────────────────────
DATASET_ANN_PATH=""
if [[ "$DATASETS" == "train_r2r_rxr" ]]; then
    DATASET_ANN_PATH="/home/user/HuHuaiyang/train_r2r_rxr.json"
fi

if [[ -n "$DATASET_ANN_PATH" && -f "$DATASET_ANN_PATH" ]]; then
    DATASET_SAMPLES=$(python3 - <<PYEOF
import json
path = r"$DATASET_ANN_PATH"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(len(data) if isinstance(data, list) else 0)
PYEOF
)
else
    echo "[Main] 警告：未找到数据集标注文件，回退到 TOTAL_STEPS=30000"
    DATASET_SAMPLES=0
fi

GLOBAL_BATCH_SIZE=$(( TRAIN_BATCH_SIZE * GRAD_ACC_STEPS * NPROC ))
if [[ "$DATASET_SAMPLES" -gt 0 ]]; then
    TOTAL_STEPS=$(( (DATASET_SAMPLES * TARGET_EPOCHS + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE ))
else
    TOTAL_STEPS=30000
fi

# ─── 环境变量 ────────────────────────────────────────────────────────────────
export CUDA_HOME=/data/miniconda3/envs/vln
export PATH=$CUDA_HOME/bin:$PATH
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_LEVEL=SYS
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_BUFFSIZE=2097152
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── DeepSpeed 配置 ──────────────────────────────────────────────────────────
# 监控数据显示：显存仅用 34/30 GB（80GB 卡占比 42%/37%），余量充足。
# 去掉 offload_optimizer（CPU Adam），改为纯 GPU Adam（FusedAdam）。
# 原来每个 optimizer step 需要 4~5 秒 GPU→CPU→GPU 传输，导致 34% 双卡空闲。
# 去掉后 optimizer step 在 GPU 上完成，预计再提速 25~35%。
DS_CONFIG="/tmp/zero2_gpu_adam_autoeval.json"
cat > "$DS_CONFIG" << 'EOF'
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": { "lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto" }
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
EOF

# ─── 辅助函数 ────────────────────────────────────────────────────────────────

# 将评估所需的三个配置文件复制到 checkpoint 目录（若已存在则跳过）
copy_config_files() {
    local CKPT="$1"
    local FILES=("chat_template.json" "configuration.json" "preprocessor_config.json")
    for F in "${FILES[@]}"; do
        local SRC="${CONFIG_SRC}/${F}"
        local DST="${CKPT}/${F}"
        if [[ ! -f "$DST" ]]; then
            if [[ -f "$SRC" ]]; then
                cp "$SRC" "$DST"
                echo "[CopyConfig] 已复制 ${F} → $(basename "$CKPT")/"
            else
                echo "[CopyConfig] 警告: 源文件不存在: $SRC"
            fi
        fi
    done
}

# 解析 result.json 中最终汇总行的 SPL（如果没有汇总行，则平均 episode 级 SPL）
parse_spl() {
    local result_file="$1"
    python3 - <<PYEOF
import json, sys, os

result_file = "$result_file"
if not os.path.exists(result_file):
    print("0.0000")
    sys.exit(0)

spls = []
summary_spl = None
with open(result_file) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            res = json.loads(line)
        except:
            continue
        if "spls_all" in res:
            summary_spl = res["spls_all"]   # 优先取汇总行
        elif "scene_id" in res and "spl" in res:
            spls.append(res["spl"])

if summary_spl is not None:
    print(f"{summary_spl:.4f}")
elif spls:
    print(f"{sum(spls)/len(spls):.4f}")
else:
    print("0.0000")
PYEOF
}

# 对单个 checkpoint 做快速评估，将 SPL 写入 <ckpt>/quick_eval_done
quick_eval_ckpt() {
    local CKPT="$1"
    local CKPT_NAME
    CKPT_NAME=$(basename "$CKPT")
    local MARKER="${CKPT}/quick_eval_done"

    if [[ -f "$MARKER" ]]; then
        echo "[QuickEval] $CKPT_NAME 已评估过，跳过（SPL=$(cat "$MARKER")）"
        return
    fi

    local EVAL_OUT="${CKPT}/quick_eval_${EVAL_SPLIT}"
    mkdir -p "$EVAL_OUT"

    local PORT
    PORT=$(shuf -i 31000-39999 -n 1)

    echo "[QuickEval] $(date '+%H:%M:%S') 开始评估 $CKPT_NAME（${QUICK_EPISODES} episodes × max_steps=${QUICK_MAX_STEPS}）..."

    # 评估前确保三个配置文件存在（checkpoint 保存时不包含这些文件）
    copy_config_files "$CKPT"

    # 构建视频标志（SAVE_VIDEO=true 时才传 --save_video）
    local VIDEO_ARGS="--save_video_ratio ${SAVE_VIDEO_RATIO}"
    [[ "$SAVE_VIDEO" == "true" ]] && VIDEO_ARGS="--save_video ${VIDEO_ARGS}"

    torchrun \
        --nproc_per_node=$NPROC \
        --master_addr="$MASTER_ADDR" \
        --master_port=$PORT \
        src/evaluation.py \
        --model_path "$CKPT" \
        --habitat_config_path "$CONFIG" \
        --eval_split "$EVAL_SPLIT" \
        --output_path "$EVAL_OUT" \
        --max_episodes $QUICK_EPISODES \
        --max_steps $QUICK_MAX_STEPS \
        $VIDEO_ARGS \
        2>&1 | tee "${EVAL_OUT}/quick_eval.log"

    local SPL
    SPL=$(parse_spl "${EVAL_OUT}/result.json")
    echo "$SPL" > "$MARKER"
    echo "[QuickEval] $CKPT_NAME 完成: ${EVAL_SPLIT} SPL = $SPL"
}

# 对所有已保存的 checkpoint 批量评估，返回最佳 checkpoint 路径（写入 stdout 最后一行）
eval_all_checkpoints() {
    local BEST_SPL="0"
    local BEST_CKPT=""
    local TIMESTAMP
    TIMESTAMP=$(date '+%Y-%m-%dT%H:%M:%S')

    echo "──────────────────────────────────────────────────────────"
    echo "[BatchEval] $(date '+%H:%M:%S') 开始批量评估保留的 checkpoint ..."

    local CKPTS
    CKPTS=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V || true)

    if [[ -z "$CKPTS" ]]; then
        echo "[BatchEval] 未发现 checkpoint，跳过评估"
        return
    fi

    for CKPT in $CKPTS; do
        quick_eval_ckpt "$CKPT"
        local MARKER="${CKPT}/quick_eval_done"
        local SPL
        SPL=$(cat "$MARKER" 2>/dev/null || echo "0.0000")

        # 追加到汇总日志
        echo "{\"time\":\"${TIMESTAMP}\",\"checkpoint\":\"$(basename "$CKPT")\",\"split\":\"${EVAL_SPLIT}\",\"spl\":${SPL}}" \
            >> "$EVAL_SUMMARY"

        # 用 awk 比较浮点数
        if awk "BEGIN{exit !($SPL > $BEST_SPL)}"; then
            BEST_SPL="$SPL"
            BEST_CKPT="$CKPT"
        fi
    done

    echo "──────────────────────────────────────────────────────────"
    echo "[BatchEval] 评估结果汇总："
    for CKPT in $CKPTS; do
        local MARKER="${CKPT}/quick_eval_done"
        local SPL
        SPL=$(cat "$MARKER" 2>/dev/null || echo "N/A")
        local MARK=""
        [[ "$CKPT" == "$BEST_CKPT" ]] && MARK=" ★ BEST"
        echo "  $(basename "$CKPT")  SPL=$SPL${MARK}"
    done
    echo "──────────────────────────────────────────────────────────"

    if [[ -n "$BEST_CKPT" ]]; then
        echo "$BEST_CKPT" > "$BEST_CKPT_FILE"
        echo "[BatchEval] 当前最佳: $(basename "$BEST_CKPT")  SPL=$BEST_SPL"
        echo "[BatchEval] 已写入: $BEST_CKPT_FILE"
    fi

    # ── 清理：删除既非最佳、又非最新（续训需要）的 checkpoint ──────────────
    local LATEST_FOR_PRUNE
    LATEST_FOR_PRUNE=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    local PRUNED=0
    for CKPT in $CKPTS; do
        [[ "$CKPT" == "$BEST_CKPT" ]]        && continue   # 保留：当前最佳
        [[ "$CKPT" == "$LATEST_FOR_PRUNE" ]] && continue   # 保留：最新（续训）
        echo "[Prune] 删除多余存档: $(basename "$CKPT")"
        rm -rf "$CKPT"
        PRUNED=$(( PRUNED + 1 ))
    done
    local KEEP_MSG="$(basename "${BEST_CKPT:-（无）}")（最佳）"
    if [[ -n "$LATEST_FOR_PRUNE" && "$LATEST_FOR_PRUNE" != "$BEST_CKPT" ]]; then
        KEEP_MSG="${KEEP_MSG} / $(basename "$LATEST_FOR_PRUNE")（最新/续训）"
    fi
    echo "[Prune] 本轮删除 ${PRUNED} 个多余存档，保留: ${KEEP_MSG}"
}

# ─── 主训练循环 ──────────────────────────────────────────────────────────────
echo "========================================================"
echo "[Main] 训练 + 自动批量评估 启动"
echo "  数据集样本数 : $DATASET_SAMPLES"
echo "  目标轮数     : $TARGET_EPOCHS"
echo "  全局batch    : $GLOBAL_BATCH_SIZE (= $TRAIN_BATCH_SIZE × $GRAD_ACC_STEPS × $NPROC)"
echo "  总步数     : $TOTAL_STEPS"
echo "  评估间隔   : $EVAL_INTERVAL 步"
echo "  每次评估   : checkpoint 数=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | wc -l || true)（save_total_limit 4）"
echo "  每个评估   : $QUICK_EPISODES episodes, max_steps=$QUICK_MAX_STEPS"
echo "  评估集     : $EVAL_SPLIT"
echo "========================================================"

for STEP_THRESHOLD in $(seq $EVAL_INTERVAL $EVAL_INTERVAL $TOTAL_STEPS); do

    echo ""
    echo "###  训练阶段：目标步数 = $STEP_THRESHOLD  ###"

    # 检测已有 checkpoint 步数，如果已经超过本阶段目标则跳过训练
    LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    if [[ -n "$LATEST_CKPT" ]]; then
        LATEST_STEP=$(basename "$LATEST_CKPT" | sed 's/checkpoint-//')
        if (( LATEST_STEP >= STEP_THRESHOLD )); then
            echo "[Main] 已有 checkpoint-${LATEST_STEP} >= ${STEP_THRESHOLD}，跳过训练阶段"
        else
            echo "[Main] 从 checkpoint-${LATEST_STEP} 继续，训练至步数 ${STEP_THRESHOLD}"
            TRAIN_PORT=$(shuf -i 20000-29999 -n 1)
            torchrun \
                --nproc_per_node=$NPROC \
                --master_addr=$MASTER_ADDR \
                --master_port=$TRAIN_PORT \
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
                --per_device_train_batch_size $TRAIN_BATCH_SIZE \
                --gradient_accumulation_steps $GRAD_ACC_STEPS \
                --learning_rate 2e-5 \
                --mm_projector_lr 1e-5 \
                --vision_tower_lr 1e-6 \
                --model_max_length 20000 \
                --data_flatten True \
                --max_pixels $((576*28*28)) \
                --min_pixels $((16*28*28)) \
                --base_interval 2 \
                --video_max_frames 13 \
                --video_min_frames 4 \
                --video_max_frame_pixels $((1664*28*28)) \
                --video_min_frame_pixels $((256*28*28)) \
                --num_train_epochs 1 \
                --warmup_ratio 0.03 \
                --lr_scheduler_type cosine \
                --weight_decay 0.01 \
                --logging_steps 10 \
                --save_steps 200 \
                --save_total_limit 6 \
                --deepspeed "$DS_CONFIG" \
                --gradient_checkpointing \
                --dataloader_num_workers $DATALOADER_NUM_WORKERS \
                --dataloader_pin_memory True \
                --dataloader_prefetch_factor 4 \
                --group_by_modality_length True \
                --seed 42 \
                --max_steps $TOTAL_STEPS \
                --stop_at_step $STEP_THRESHOLD \
                --report_to none \
                --reference_frame first \
                --resume_from_checkpoint "$LATEST_CKPT" \
                2>&1 | tee -a "${OUTPUT_DIR}/train.log"
        fi
    else
        echo "[Main] 无已有 checkpoint，从头训练至步数 ${STEP_THRESHOLD}"
        TRAIN_PORT=$(shuf -i 20000-29999 -n 1)
        torchrun \
            --nproc_per_node=$NPROC \
            --master_addr=$MASTER_ADDR \
            --master_port=$TRAIN_PORT \
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
            --per_device_train_batch_size $TRAIN_BATCH_SIZE \
            --gradient_accumulation_steps $GRAD_ACC_STEPS \
            --learning_rate 2e-5 \
            --mm_projector_lr 1e-5 \
            --vision_tower_lr 1e-6 \
            --model_max_length 20000 \
            --data_flatten True \
            --max_pixels $((576*28*28)) \
            --min_pixels $((16*28*28)) \
            --base_interval 2 \
            --video_max_frames 13 \
            --video_min_frames 4 \
            --video_max_frame_pixels $((1664*28*28)) \
            --video_min_frame_pixels $((256*28*28)) \
            --num_train_epochs 1 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type cosine \
            --weight_decay 0.01 \
            --logging_steps 10 \
            --save_steps 200 \
            --save_total_limit 6 \
            --deepspeed "$DS_CONFIG" \
            --gradient_checkpointing \
            --dataloader_num_workers $DATALOADER_NUM_WORKERS \
            --dataloader_pin_memory True \
            --dataloader_prefetch_factor 4 \
            --group_by_modality_length True \
            --seed 42 \
            --max_steps $TOTAL_STEPS \
            --stop_at_step $STEP_THRESHOLD \
            --report_to none \
            --reference_frame first \
            2>&1 | tee -a "${OUTPUT_DIR}/train.log"
    fi

    # ── 评估阶段 ─────────────────────────────────────────────────────
    echo ""
    echo "### 评估阶段：step=$STEP_THRESHOLD，评估全部保留 checkpoint ###"
    eval_all_checkpoints

done

echo ""
echo "========================================================"
echo "[Main] 全部 $TOTAL_STEPS 步训练 + 评估完成！"
FINAL_BEST=$(cat "$BEST_CKPT_FILE" 2>/dev/null || echo "未记录")
echo "[Main] 最终最佳 checkpoint：$FINAL_BEST"
echo "[Main] 评估汇总日志：$EVAL_SUMMARY"
echo "========================================================"
