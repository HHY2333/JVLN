export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))

CHECKPOINT="/data/model/JanusVLN_Base/3b_zero3/checkpoint-5000"
echo "CHECKPOINT: ${CHECKPOINT}"
OUTPUT_PATH="evaluation"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
CONFIG="config/vln_r2r.yaml"
echo "CONFIG: ${CONFIG}"

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT src/evaluation.py --model_path $CHECKPOINT --habitat_config_path $CONFIG --max_steps 200 --seed 55 --save_video --output_path $OUTPUT_PATH

