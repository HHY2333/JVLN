import os
import gzip
import json
import glob
import numpy as np
import concurrent.futures
from functools import partial
from tqdm import tqdm
import random
import argparse 

# =============================================================================
# 历史帧采样参数（History Frame Sampling Parameters）
# 论文 Efficient-VLN Section 3.3 "Temporal Sampling"：
#
# 以固定步长 ∆ 从历史中均匀采样，减少相邻帧的冗余。
# 再配合渐进式压缩策略：
#   - 最近 K 帧的 ft 以 2×2 下采样
#   - 次近 K 帧以 4×4 下采样
#   - 更早帧以 8×8 下采样
# 总 token 上界 ≈ K * S * (1/4 + 1/16 + 1/64 + ...) ≤ K/3 * S
#
# max_history_images: 最大保留的历史帧数（不含当前帧）
#   设为 K * 压缩层数 = 3 * 3 = 9（3 组，每组 3 帧），与论文中 K=3 对应
# temporal_stride:    采样步长 ∆，每 ∆ 步取一帧历史
#   ∆=2 表示每 2 步采一帧；轨迹越长，覆盖时间跨度越大
# =============================================================================
MAX_HISTORY_IMAGES = 12      # 最大历史帧数（不含当前帧），对应 K=4 × 3 压缩层 = 12（论文 window=12）
TEMPORAL_STRIDE = 4          # 采样步长 ∆：每 ∆ 步取一帧历史（论文 Section 3.3，∆=4）


def sample_history_frames(i: int, total_frames: int) -> list:
    """
    给定当前步 i，从 [0, i-1] 中按步长 ∆ 采样历史帧索引，
    并保证当前帧 i 始终是列表最后一个元素。

    采样策略（论文 Section 3.3 Temporal Sampling）：
    1. 从最近一帧历史（i-1）开始，每隔 TEMPORAL_STRIDE 步向过去采样
    2. 采样结果按时间正序排列，最后追加当前帧 i
    3. 若历史帧不足 MAX_HISTORY_IMAGES，则全取

    Args:
        i:            当前帧索引（0-indexed，即轨迹第 i 步）
        total_frames: 轨迹总帧数（仅用于边界检查）

    Returns:
        采样后的帧索引列表，格式：[hist_idx_0, hist_idx_1, ..., current_idx=i]
        其中历史帧按时间正序，当前帧在末尾。
    """
    if i == 0:
        # 第 0 步：没有历史帧，只有当前帧
        return [0]

    # 从 i-1 开始，每 TEMPORAL_STRIDE 步向过去采样，共采 MAX_HISTORY_IMAGES 帧
    # Start from i-1, sample backwards with stride TEMPORAL_STRIDE
    history_idxs = list(range(i - 1, -1, -TEMPORAL_STRIDE))   # [i-1, i-1-∆, i-1-2∆, ...]
    history_idxs = history_idxs[:MAX_HISTORY_IMAGES]           # 截断到最大历史帧数
    history_idxs = sorted(history_idxs)                        # 恢复时间正序

    # 拼接：[历史帧（时间正序）] + [当前帧]
    return history_idxs + [i]


def process_episode_scalevln(ep, img_root, act_map):
    episode_results = []
    episode_id = str(ep['id'])
    instruction = ep['instructions'][0]
    name = ep['video'].split('/')[1]
    img_dir = os.path.join(img_root, name, 'rgb')
    missing_images_count = 0  

    try:
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    except FileNotFoundError:
        print(f"Warning: Directory not found for scalevln episode {episode_id}: {img_dir}")
        return [] 

    num_images = len(img_files)
    for i in range(num_images):
        # -------------------------------------------------------------------
        # 历史帧采样（History Frame Sampling）
        # 论文 Section 3.3: 以步长 ∆=TEMPORAL_STRIDE 均匀采样历史帧，
        # 保留最近 MAX_HISTORY_IMAGES 帧，当前帧始终在末尾。
        # 比原来的 np.linspace 方案更忠实于论文：
        #   - linspace 均等分割时间轴（固定总帧数）
        #   - 步长 ∆ 以固定间隔从最近帧向过去采样（自然体现"最近帧最密集"）
        # -------------------------------------------------------------------
        idxs = sample_history_frames(i, num_images)
        sampled_imgs = [img_files[j] for j in idxs]

        original_len = len(sampled_imgs)
        sampled_imgs = [img_path for img_path in sampled_imgs if os.path.exists(img_path)]
        
        num_missing = original_len - len(sampled_imgs)
        if num_missing > 0:
            missing_images_count += num_missing

        if not sampled_imgs:
            continue 
        
        his_img_tags = "<image>" * (len(sampled_imgs) - 1)
        
        if i == num_images - 1:
            action = 'STOP'
        else:
            action = act_map[ep['actions'][i + 1]]
        
        conversations = [
            {"from": "human", "value": f"You are a visual language navigation model, and your should go to the locations to complete the given task. Compare the observation and instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location and finish the task.\n This is your historical observation:{his_img_tags}\n This is your current observation:<image>\n Your task is to {instruction}\n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."},
            {"from": "gpt", "value": action}
        ]
        
        sample_dict = {
            "id": f"{episode_id}/{os.path.basename(img_files[i])}",
            "conversations": conversations,
            "images": sampled_imgs
        }

        episode_results.append(sample_dict) 

    if missing_images_count != 0:
        print("miss:", missing_images_count)
        
    return episode_results


def process_episode_vlnce(ep, img_root):

    episode_results = []
    episode_id = str(ep['episode_id'])
    instruction = ep['instruction']['instruction_text'].strip()
    img_dir = os.path.join(img_root, episode_id)
    
    try:
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    except FileNotFoundError:
        print(f"Warning: Directory not found for VLN-CE episode {episode_id}: {img_dir}")
        return []

    for i in range(len(img_files)):

        if i <= MAX_HISTORY_IMAGES:
            idxs = list(range(i + 1))
        else:
            idxs = np.linspace(0, i, 9, dtype=int).tolist()
        sampled_imgs = [img_files[j] for j in idxs]

        his_img_tags = "<image>" * (len(sampled_imgs) - 1)
        name_parts = os.path.basename(img_files[i]).replace('.png', '').split('_')
        if len(name_parts) > 3:
            action = f"{name_parts[-2].upper()}_{name_parts[-1].upper()}"
        else:
            action = name_parts[-1].upper()
            
        conversations = [
            {"from": "human", "value": f"You are a visual language navigation model, and your should go to the locations to complete the given task. Compare the observation and instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location and finish the task.\n This is your historical observation:{his_img_tags}\n This is your current observation:<image>\n Your task is to {instruction}\n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."},
            {"from": "gpt", "value": action}
        ]
        
        sample_dict = {
            "id": f"{episode_id}/{os.path.basename(img_files[i])}",
            "conversations": conversations,
            "images": sampled_imgs
        }

        episode_results.append(sample_dict)

    return episode_results


def main():
    parser = argparse.ArgumentParser(description="Process VLN datasets for training.")
    parser.add_argument(
        '--use_extra_data', 
        action='store_true',  
        help="Include extra datasets (ScaleVLN, DAgger R2R, DAgger RxR) in the processing."
    )
    args = parser.parse_args()

    all_results = []

    # --- ScaleVLN Dataset ---
    img_root_scalevln = "data/trajectory_data/ScaleVLN/images"
    json_path_scalevln = "data/trajectory_data/ScaleVLN/annotations.json"
    act_map_scalevln = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]

    # --- DAgger Dataset ---
    img_root_dagger_r2r = "data/dagger_data/R2R/images"
    json_path_dagger_r2r = "data/dagger_data/R2R/annotations.json"
    act_map_dagger_r2r = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]


    img_root_dagger_rxr = "data/dagger_data/RxR/images"
    json_path_dagger_rxr = "data/dagger_data/RxR/annotations.json"
    act_map_dagger_rxr = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]

    # --- R2R Dataset ---
    img_root_r2r = "/data/datasets/data/trajectory_data/R2R/train"
    json_path_r2r = "/data/datasets/data/datasets/r2r/train/train.json.gz"

    # --- RxR Dataset ---
    img_root_rxr = "/data/datasets/data/trajectory_data/RxR/train"
    json_path_rxr = "/data/datasets/data/datasets/rxr/train/train_guide.json.gz"

    print("Loading JSON data...")

    if args.use_extra_data:
        with open(json_path_scalevln, 'r', encoding='utf-8') as f:
            data_scalevln = json.load(f)

        with open(json_path_dagger_r2r, 'r', encoding='utf-8') as f:
            data_dagger_r2r = json.load(f)


        with open(json_path_dagger_rxr, 'r', encoding='utf-8') as f:
            data_dagger_rxr = json.load(f)

    with gzip.open(json_path_r2r, 'rt', encoding='utf-8') as f:
        data_r2r = json.load(f)

    with gzip.open(json_path_rxr, 'rt', encoding='utf-8') as f:
        data_rxr = json.load(f)
    print("JSON data loaded.")
        
    with concurrent.futures.ProcessPoolExecutor() as executor:
        if args.use_extra_data:
            print("\nProcessing ScaleVLN dataset...")
            p_process_scalevln = partial(process_episode_scalevln, img_root=img_root_scalevln, act_map=act_map_scalevln)
            results_iterator = tqdm(executor.map(p_process_scalevln, data_scalevln), total=len(data_scalevln))
            for episode_res in results_iterator:
                all_results.extend(episode_res)
            print(f"Finished ScaleVLN. Total samples: {len(all_results)}")

            print("\nProcessing DAgger R2R dataset...")
            p_process_dagger_r2r = partial(process_episode_scalevln, img_root=img_root_dagger_r2r, act_map=act_map_dagger_r2r)
            results_iterator = tqdm(executor.map(p_process_dagger_r2r, data_dagger_r2r), total=len(data_dagger_r2r))
            for episode_res in results_iterator:
                all_results.extend(episode_res)
            print(f"Finished DAgger R2R. Total samples: {len(all_results)}")


            print("\nProcessing DAgger RxR dataset...")
            p_process_dagger_rxr = partial(process_episode_scalevln, img_root=img_root_dagger_rxr, act_map=act_map_dagger_rxr)
            results_iterator = tqdm(executor.map(p_process_dagger_rxr, data_dagger_rxr), total=len(data_dagger_rxr))
            for episode_res in results_iterator:
                all_results.extend(episode_res)
            print(f"Finished DAgger RxR. Total samples: {len(all_results)}")

        # --- Process R2R ---
        print("\nProcessing R2R dataset...")
        p_process_r2r = partial(process_episode_vlnce, img_root=img_root_r2r)
        results_iterator = tqdm(executor.map(p_process_r2r, data_r2r['episodes']), total=len(data_r2r['episodes']))
        for episode_res in results_iterator:
            all_results.extend(episode_res)
        print(f"Finished R2R. Total samples: {len(all_results)}")

        # --- Process RxR ---
        print("\nProcessing RxR dataset...")
        p_process_rxr = partial(process_episode_vlnce, img_root=img_root_rxr)
        results_iterator = tqdm(executor.map(p_process_rxr, data_rxr['episodes']), total=len(data_rxr['episodes']))
        for episode_res in results_iterator:
            all_results.extend(episode_res)
        print(f"Finished RxR. Total samples: {len(all_results)}")

    if args.use_extra_data:
        output_path = "train_r2r_rxr_extra.json"
    else:
        output_path = "train_r2r_rxr.json"

    print(f"\nAll processing finished. Saving {len(all_results)} samples to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()











