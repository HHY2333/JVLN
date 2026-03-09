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

max_history_images = 16

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

    STRIDE = 4  # 采样间隔：每隔 STRIDE 步取一帧，与评估保持一致
    num_images = len(img_files)
    for i in range(num_images):
        # 从当前帧向前以 STRIDE 为步长回溯，最多取 max_history_images+1 帧
        # range 从 i 向 0 方向步进 -STRIDE，最多取 max_history_images+1 个索引
        back_idxs = list(range(i, max(i - max_history_images * STRIDE, -1), -STRIDE))
        idxs = list(reversed(back_idxs))  # 恢复时间正序（旧→新，最后一个为当前帧 i）
        sampled_imgs = [img_files[j] for j in idxs]

        original_len = len(sampled_imgs)
        sampled_imgs = [img_path for img_path in sampled_imgs if os.path.exists(img_path)]
        
        num_missing = original_len - len(sampled_imgs)
        if num_missing > 0:
            missing_images_count += num_missing

        if not sampled_imgs:
            continue 
        
        if i == num_images - 1:
            action = 'STOP'
        else:
            action = act_map[ep['actions'][i + 1]]
        
        # Prompt 与评估对齐：单 <image> 占位当前帧，历史帧通过 images 列表传入
        # data_qwen.py 会将历史帧分离走 ProgressiveSpatialCompressor
        conversations = [
            {"from": "human", "value": f"<image> Your task is to {instruction}\n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."},
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

    STRIDE = 4  # 采样间隔：与 process_episode_scalevln 及评估保持一致
    for i in range(len(img_files)):

        back_idxs = list(range(i, max(i - max_history_images * STRIDE, -1), -STRIDE))
        idxs = list(reversed(back_idxs))
        sampled_imgs = [img_files[j] for j in idxs]

        # VLN-CE 最后一帧显式设为 STOP（原代码依赖文件名提取，最后一帧可能标错）
        if i == len(img_files) - 1:
            action = 'STOP'
        else:
            name_parts = os.path.basename(img_files[i]).replace('.png', '').split('_')
            if len(name_parts) > 3:
                action = f"{name_parts[-2].upper()}_{name_parts[-1].upper()}"
            else:
                action = name_parts[-1].upper()

        # Prompt 与评估对齐：单 <image> 占位当前帧，历史帧通过 images 列表传入
        conversations = [
            {"from": "human", "value": f"<image> Your task is to {instruction}\n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."},
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
        
    # 磁盘 I/O 密集型任务，限制并发数避免 glob 引发大量 I/O 争用导致卡死
    cpu_count = os.cpu_count() or 8
    max_workers = min(cpu_count, 32)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
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
        results_iterator = tqdm(executor.map(p_process_r2r, data_r2r['episodes'], chunksize=32), total=len(data_r2r['episodes']))
        for episode_res in results_iterator:
            all_results.extend(episode_res)
        print(f"Finished R2R. Total samples: {len(all_results)}")

        # --- Process RxR ---
        print("\nProcessing RxR dataset...")
        p_process_rxr = partial(process_episode_vlnce, img_root=img_root_rxr)
        results_iterator = tqdm(executor.map(p_process_rxr, data_rxr['episodes'], chunksize=64), total=len(data_rxr['episodes']))
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











