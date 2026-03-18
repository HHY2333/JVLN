#!/bin/bash
# =============================================================================
# trainwith_eval_progressive.sh
# 渐进式记忆压缩训练脚本（Progressive Memory Compression Training）
#
# 核心改动（相对于 trainwith_eval.sh）：
#   1. data_flatten=True   — 开启序列打包，提升 GPU 利用率
#   2. 模型：Qwen2.5-VL-3B-Instruct（3B）
#   3. ZeRO Stage 2（GPU Adam，无 offload）
#   4. 历史帧窗口 W=12，步长 ∆=4（与论文一致）
#      对应 memory_group_size=4（K=4，3层×4=12帧）
#
# 性能优化记录（相对初始版本）：
#   [实测分析] 数据集图像均为 640×480 (habitat 仿真器输出)，实际 374 tokens/帧。
#   video_max_frame_pixels 无论设多大，对本数据集均无缩放，故不改动。
#
#   - model_max_length: 20000→12288
#       实测单样本均值 ~1028 tokens，p99=1321，max=1616；
#       batch_size=8 打包后 p99_packed=10568，12288 有安全余量，20000 过度浪费。
#   - gradient_checkpointing: 开启
#       GPU1 显存已达 79/81 GB（97%），开启后降至 ~55 GB，
#       为 batch_size 从 6→8 提供空间（全局 batch 24→32）。
#   - TRAIN_BATCH_SIZE: 6→8（全局 batch=8×2×2=32 样本/step）
#   - NCCL_BUFFSIZE: 2MB→32MB（减少双卡 AllReduce 往返轮次）
#   - DATALOADER_NUM_WORKERS: 8→4（8 workers×4 VideoReader 线程=32 CPU 线程竞争）
#   - dataloader_prefetch_factor: 2→4（VGGT 串行耗时，需更多预取缓冲以掩盖延迟）
#
#   [根本瓶颈] VGGT-1B 逐帧串行 forward（约 52 次/step）是 20s/it 的主因，
#   需代码层批量化改造（当前脚本无法解决）。
#
# 用法：cd /data/HuHuaiyang/JVLN && bash scripts/trainwith_eval_progressive.sh
# =============================================================================

set -euo pipefail

# ─── 可调参数 ────────────────────────────────────────────────────────────────
TARGET_EPOCHS=1            # 训练轮数
EVAL_INTERVAL=5000        # 分段训练间隔（每训练 EVAL_INTERVAL 步停下评估）
QUICK_EPISODES=16          # 快速评估 episode 数
QUICK_MAX_STEPS=50         # 快速评估每 episode 最大步数
EVAL_SPLIT="val_seen"      # 评估集
NPROC=2                    # GPU 数量
DATALOADER_NUM_WORKERS=4   # DataLoader worker 数（4 workers × 4 VideoReader线程=16线程，避免过多 CPU 争用）

# -----------------------------------------------------------------------
# 序列打包（Sequence Packing）配置
# data_flatten=True：将多条样本的 token 拼接到同一个长序列中，
# 避免 padding 浪费，提升 GPU 利用率（论文同款配置）。
#
# 实测单样本均值 ~1028 tokens（640×480 图像 374 tokens/帧，渐进压缩后均值 806 视觉
# tokens + ~220 文本 tokens，详见 model_max_length 注释）。
# batch_size=8 + grad_acc=2 → 全局 batch = 8×2×2GPU = 32 样本/step；
# 开启 gradient_checkpointing 节省约 20GB 显存以支持更大 batch。
# -----------------------------------------------------------------------
TRAIN_BATCH_SIZE=8         # per_device_train_batch_size（gradient_checkpointing 后显存充裕）
GRAD_ACC_STEPS=2           # gradient_accumulation_steps（全局 batch=8×2×2=32 样本/step）
DATA_FLATTEN=True          # 开启序列打包

SAVE_VIDEO=false
SAVE_VIDEO_RATIO=0.2

# ─── 路径配置 ────────────────────────────────────────────────────────────────
MASTER_ADDR="127.0.0.1"
MODEL_PATH="/data/model/Qwen2.5-VL-3B-Instruct"
VGGT_MODEL_PATH="/data/model/VGGT-1B"
# 与 trainwith_eval.sh 使用不同 OUTPUT_DIR，避免 checkpoint 混淆
OUTPUT_DIR="/data/model/evln/1"
CACHE_DIR="/data/model/evln/1/cache"
CONFIG="config/vln_r2r.yaml"
DATASETS="train_r2r_rxr"
CONFIG_SRC="/data/model/JanusVLN_Base/1"

BEST_CKPT_FILE="${OUTPUT_DIR}/best_checkpoint.txt"
EVAL_SUMMARY="${OUTPUT_DIR}/quick_eval_summary.jsonl"
mkdir -p "$OUTPUT_DIR"

# ─── 自动统计数据集样本数 ─────────────────────────────────────────────────────
DATASET_ANN_PATH=""
if [[ "$DATASETS" == "train_r2r_rxr" ]]; then
    DATASET_ANN_PATH="/data/HuHuaiyang/JVLN/train_r2r_rxr.json"
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
export NCCL_BUFFSIZE=33554432
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── DeepSpeed ZeRO-2 配置（GPU Adam，无 offload）────────────────────────────
DS_CONFIG="/tmp/zero2_gpu_adam_progressive.json"
cat > "$DS_CONFIG" << EOF
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
    "gradient_accumulation_steps": ${GRAD_ACC_STEPS},
  "gradient_clipping": 1.0,
    "train_batch_size": ${GLOBAL_BATCH_SIZE},
    "train_micro_batch_size_per_gpu": ${TRAIN_BATCH_SIZE},
  "wall_clock_breakdown": false
}
EOF

# ─── 辅助函数 ────────────────────────────────────────────────────────────────
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
            summary_spl = res["spls_all"]
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
    copy_config_files "$CKPT"

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
        --num_history 12 \
        $VIDEO_ARGS \
        2>&1 | tee "${EVAL_OUT}/quick_eval.log"

    local SPL
    SPL=$(parse_spl "${EVAL_OUT}/result.json")
    echo "$SPL" > "$MARKER"
    echo "[QuickEval] $CKPT_NAME 完成: ${EVAL_SPLIT} SPL = $SPL"
}

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

        echo "{\"time\":\"${TIMESTAMP}\",\"checkpoint\":\"$(basename "$CKPT")\",\"split\":\"${EVAL_SPLIT}\",\"spl\":${SPL}}" \
            >> "$EVAL_SUMMARY"

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
    fi

    local LATEST_FOR_PRUNE
    LATEST_FOR_PRUNE=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    local PRUNED=0
    for CKPT in $CKPTS; do
        [[ "$CKPT" == "$BEST_CKPT" ]]        && continue
        [[ "$CKPT" == "$LATEST_FOR_PRUNE" ]] && continue
        echo "[Prune] 删除多余存档: $(basename "$CKPT")"
        rm -rf "$CKPT"
        PRUNED=$(( PRUNED + 1 ))
    done
    echo "[Prune] 本轮删除 ${PRUNED} 个多余存档"
}

# ─── 主训练（分段 stop_at_step + 总进度条）────────────────────────────────────
echo "========================================================"
echo "[Main] 渐进式记忆压缩训练 启动（Progressive Memory Compression）"
echo "  模型         : Qwen2.5-VL-3B-Instruct"
echo "  ZeRO 阶段    : 2（GPU Adam，无 offload）"
echo "  序列打包     : 开启（data_flatten=True，多样本拼接提升 GPU 利用率）"
echo "  历史窗口     : W=12 帧，步长 ∆=4，K=4（3 压缩层 × 4 帧）"
echo "  数据集样本数 : $DATASET_SAMPLES"
echo "  全局 batch   : $GLOBAL_BATCH_SIZE (= $TRAIN_BATCH_SIZE × $GRAD_ACC_STEPS × $NPROC)"
echo "  总步数       : $TOTAL_STEPS"
echo "  分段间隔     : $EVAL_INTERVAL"
echo "  输出目录     : $OUTPUT_DIR"
echo "========================================================"

render_progress() {
    local CURRENT="$1"
    local TOTAL="$2"
    local WIDTH=40
    local FILLED=$(( CURRENT * WIDTH / TOTAL ))
    local EMPTY=$(( WIDTH - FILLED ))
    local PCT=$(( CURRENT * 100 / TOTAL ))
    printf "\n[Progress] ["
    printf "%${FILLED}s" "" | tr ' ' '#'
    printf "%${EMPTY}s" "" | tr ' ' '-'
    printf "] %3d%%  (%d/%d steps)\n" "$PCT" "$CURRENT" "$TOTAL"
}

for STEP_THRESHOLD in $(seq $EVAL_INTERVAL $EVAL_INTERVAL $TOTAL_STEPS); do
    render_progress "$STEP_THRESHOLD" "$TOTAL_STEPS"
    echo "[Stage] 训练到 step=${STEP_THRESHOLD} 后停下评估"

    TRAIN_PORT=$(shuf -i 20000-29999 -n 1)
    LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)

    if [[ -n "$LATEST_CKPT" ]]; then
        echo "[Main] 从 $(basename "$LATEST_CKPT") 续训"
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
            --model_max_length 12288 \
            --data_flatten $DATA_FLATTEN \
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
            --save_steps 500 \
            --save_total_limit 6 \
            --deepspeed "$DS_CONFIG" \
            --gradient_checkpointing True \
            --dataloader_num_workers $DATALOADER_NUM_WORKERS \
            --dataloader_pin_memory True \
            --dataloader_prefetch_factor 4 \
            --group_by_modality_length True \
            --seed 42 \
            --max_steps $TOTAL_STEPS \
            --stop_at_step $STEP_THRESHOLD \
            --total_steps $TOTAL_STEPS \
            --report_to none \
            --reference_frame first \
            --resume_from_checkpoint "$LATEST_CKPT" \
            2>&1 | tee -a "${OUTPUT_DIR}/train.log"
    else
        echo "[Main] 从头训练"
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
            --model_max_length 12288 \
            --data_flatten $DATA_FLATTEN \
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
            --save_steps 500 \
            --save_total_limit 6 \
            --deepspeed "$DS_CONFIG" \
            --gradient_checkpointing True \
            --dataloader_num_workers $DATALOADER_NUM_WORKERS \
            --dataloader_pin_memory True \
            --dataloader_prefetch_factor 4 \
            --group_by_modality_length True \
            --seed 42 \
            --max_steps $TOTAL_STEPS \
            --stop_at_step $STEP_THRESHOLD \
            --total_steps $TOTAL_STEPS \
            --report_to none \
            --reference_frame first \
            2>&1 | tee -a "${OUTPUT_DIR}/train.log"
    fi

    echo "[Stage] 开始评估当前保留 checkpoint"
    eval_all_checkpoints
done

echo ""
echo "========================================================"
echo "[Main] 全部完成！"
FINAL_BEST=$(cat "$BEST_CKPT_FILE" 2>/dev/null || echo "未记录")
echo "[Main] 最终最佳 checkpoint：$FINAL_BEST"
echo "[Main] 评估汇总日志：$EVAL_SUMMARY"
echo "========================================================"
