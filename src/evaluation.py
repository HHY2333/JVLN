import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import tqdm
import torch
import copy
import cv2
import json
import random
import argparse
import itertools
import quaternion
import transformers
import numpy as np

from typing import Any
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
from transformers.image_utils import to_numpy_array

import habitat
from habitat import logger, Env
from habitat_extensions import measures
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video, observations_to_image

from utils.dist import *
import base64
from datetime import datetime
from io import BytesIO
from qwen_vl_utils import extract_vision_info
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images
from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationForJanusVLN, get_progressive_compression_ratio
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


min_pixels: int = 28 * 28
max_pixels: int = 1605632


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device('cuda')
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors
        self.save_video_ratio = args.save_video_ratio


        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = self.split
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        self.image_processor = model.processor
        self.model = model
        self.tokenizer = model.tokenizer
        
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "MOVE_FORWARD": [1],
            "TURN_LEFT": [2],
            "TURN_RIGHT": [3]
        })

        self.num_history = args.num_history


    def config_env(self) -> Env:
        env = Env(config=self.config)
        return env


    def eval_action(self, idx) -> None:
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        
        sucs, spls, oss, ones = [], [], [], []
        done_res = []
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'),'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        ones.append(res['ne'])
        
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            
            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"scene {scene_id}")
            for episode in episodes[idx::self.env_num]:
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                print("episode start: ",episode_instruction)
                episode_id = episode.episode_id
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue

                env.current_episode = episode
                observations = env.reset()

                vis_frames = []
                step_id = 0
                
                should_save_video = self.save_video and (random.random() < self.save_video_ratio)
                if should_save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}'), exist_ok=True)
                

                rgb_list = []
                time_ids = []
                action_seq = []
                self.model.model.past_key_values_vggt = None
                # ----------------------------------------------------------------
                # 推理特征缓存重置（Inference Feature Cache Reset）
                # 每个 episode 开始时清空 fused_feature_cache 和 vggt_step，
                # 确保不同 episode 的历史特征和 VGGT 时序位置不混淆。
                # ----------------------------------------------------------------
                self.model.model.fused_feature_cache = []
                self.model.model.vggt_step = 0
                
                while not env.episode_over:
                    
                    time_ids.append(step_id)
                    rgb = observations["rgb"]
                    
                    image = Image.fromarray(rgb).convert('RGB')
                    rgb_list.append(image)
                    
                    info = env.get_metrics()
                        
                    history_len = len(rgb_list) - 1 
                    
                    # -------------------------------------------------------------------
                    # 推理时历史特征采样（Inference History Feature Sampling）
                    #
                    # 新设计：不再重新编码历史 RGB 帧，而是：
                    #   1. rgb_list  → 仅用于传给 processor 生成正确数量的 <image_pad> token
                    #   2. fused_feature_cache → 存储每步编码后的融合特征 f_t，
                    #      历史帧直接从此缓存取，传给 forward() 做渐进压缩，供 LLM 使用
                    #
                    # 采样策略：以步长 ∆=2 从最近历史帧向过去采样（与 create_data.py 一致），
                    # PIL 和特征使用完全相同的采样索引，保证两者对齐。
                    # -------------------------------------------------------------------
                    TEMPORAL_STRIDE = 4  # 论文 ∆=4，与 create_data.py 保持一致
                    cache = self.model.model.fused_feature_cache  # 已缓存的融合特征列表
                    if history_len == 0:
                        # 第一步：无历史帧
                        history_pil = []
                        history_features = []
                    elif history_len <= self.num_history:
                        # 历史帧不足，全部使用
                        history_pil = rgb_list[:history_len]
                        history_features = cache[:history_len]
                    else:
                        # 从最近历史帧以步长 ∆ 向过去采样，保持时间正序
                        hist_indices = list(range(history_len - 1, -1, -TEMPORAL_STRIDE))
                        hist_indices = sorted(hist_indices[:self.num_history])
                        history_pil = [rgb_list[idx] for idx in hist_indices]
                        history_features = [cache[idx] for idx in hist_indices]
                    images = history_pil + [rgb_list[-1]]  # 仅用于 processor 分词
                    
                    
                    action = self.model.call_model(
                        rgb_list[-1],           # 当前帧（仅此帧通过编码器）
                        episode_instruction,
                        step_id,
                        history_pil_images=history_pil,      # 历史 PIL，仅用于 processor 分词
                        history_features=history_features,   # 历史融合特征，用于 forward() 压缩
                    )[0]
                    
                    if info['top_down_map'] is not None and should_save_video:
                        frame = observations_to_image({'rgb':observations['rgb']}, info)
                        vis_frames.append(frame)
                    
                    if action in self.actions2idx:
                        action = self.actions2idx[action][0]
                    else:
                        action = 0


                    if step_id >= self.args.max_steps:
                        action = 0

                    observations = env.step(action)
                    step_id += 1

                process_bar.update(1)
                metrics = env.get_metrics()
                if should_save_video:
                    images_to_video(
                        vis_frames, os.path.join(self.output_path, f'vis_{self.epoch}'), f'{scene_id}_{episode_id}', fps=6, quality=9
                    )
                vis_frames.clear()
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])
                print(f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}")
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction
                }
                
                with open(os.path.join(self.output_path, f'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")

        env.close()
        return torch.tensor(sucs).to(self.device), torch.tensor(spls).to(self.device), torch.tensor(oss).to(self.device), torch.tensor(ones).to(self.device), torch.tensor(len(sucs)).to(self.device)     




class JanusVLN_Inference:
    def __init__(self, pretrained, device="cuda"):
        config = AutoConfig.from_pretrained(pretrained)
        self.model = Qwen2_5_VLForConditionalGenerationForJanusVLN.from_pretrained(
            pretrained,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="flash_attention_2",
            mode='evaluation'
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels, padding_side="left")
        
        self.device = device


    def call_model(
        self,
        observations,              # 当前帧 PIL image（仅编码此帧）
        task,
        step_id,
        history_pil_images: list = None,  # 历史帧 PIL list，仅供 processor 生成 <image_pad> 数量
        history_features: list = None,    # 历史融合特征 [(feat, hm, wm)]，来自 fused_feature_cache
        add_frame_index: bool = False,
        gen_kwargs: dict = {},
    ):
        # =====================================================================
        # call_model 核心设计（推理高效路径）
        #
        # 1. 构建包含所有帧（历史 PIL + 当前帧）的消息，传给 processor 生成含正确
        #    <image_pad> 数量的 input_ids（全分辨率）。
        # 2. images_vggt 只存当前帧（VGGT KV cache 已携带历史几何上下文）。
        # 3. 若有历史特征（history_features），对 input_ids 做修补：
        #    将历史帧的全分辨率 <image_pad> 块替换为压缩后的数量，
        #    同时更新 image_grid_thw 为压缩值（供 forward 内 get_rope_index 使用）。
        # 4. 裁剪 pixel_values，只保留当前帧的 patch 数据（历史帧特征来自缓存）。
        # 5. 将 history_features 传给 model.generate → forward，
        #    在 forward 内完成历史特征的渐进压缩 + 拼接 + <image_pad> 填充。
        # =====================================================================
        
        messages = []
        message = [
                {"role": "system", 
                "content": "You are a visual language navigation model, and your should go to the locations to complete the given task. Compare the observation and instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location and finish the task."
                }
            ]
        context = f"These images are your historical observations and your current observation.\n Your task is to {task} \n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size

        # -------------------------------------------------------------------
        # 构建所有帧列表（历史 PIL + 当前帧），用于 processor 分词
        # Build full frame list (history PIL + current) for processor tokenization.
        # 历史 PIL 的像素数据仅用于 processor 计算 <image_pad> 数量和 image_grid_thw，
        # 不会送入神经网络再次编码（历史特征来自 fused_feature_cache）。
        # -------------------------------------------------------------------
        current_pil = observations if isinstance(observations, Image.Image) else observations[-1]
        history_pil_images = history_pil_images or []
        all_pil_images = history_pil_images + [current_pil]  # 时间正序：早→晚（最后为当前帧）

        for i in enumerate([task]):
    
            image_content = []
            image_count = 0
            for v in all_pil_images:
                if add_frame_index:
                    image_content.append({"type": "text", "text": "Frame-{}: ".format(image_count)})
                image_content.append({"type": "image", "image": v})
                image_count += 1
            message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
            messages.append(message)

        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
        images_vggt = []
        image_inputs = []
        for message in messages:
            vision_info = extract_vision_info(message)
            cur_images_vggt = []
            n_vision = len(vision_info)
            for i, ele in enumerate(vision_info):
                if "image" in ele:
                    image = ele["image"]
                    if isinstance(image, Image.Image):
                        pass
                    elif isinstance(image, str) and "base64," in image:
                        _, base64_data = image.split("base64,", 1)
                        data = base64.b64decode(base64_data)
                        with BytesIO(data) as bio:
                            image = copy.deepcopy(Image.open(bio))
                    else:
                        raise NotImplementedError("Unsupported image type")   
                else:
                    raise NotImplementedError("Unsupported vision info type")
    
                assert isinstance(image, Image.Image), f"Unsupported image type: {type(image)}"
                image = load_and_preprocess_images([image])[0]

                # -------------------------------------------------------------------
                # images_vggt 只存当前帧（最后一帧）
                # Only append current frame (last) to images_vggt for VGGT encoding.
                #
                # 历史帧的 3D 几何上下文由 VGGT 的 KV cache (past_key_values_vggt) 保留，
                # 无需再次经过 VGGT 编码，节省大量 GPU 计算。
                # 历史帧的视觉语义上下文由 fused_feature_cache 提供，传给 forward() 做压缩。
                # -------------------------------------------------------------------
                if i == n_vision - 1:
                    cur_images_vggt.append(image)  # 仅当前帧

                _, height, width = image.shape
                if (width // patch_size) % merge_size > 0:
                    width = width - (width // patch_size) % merge_size * patch_size
                if (height // patch_size) % merge_size > 0:
                    height = height - (height // patch_size) % merge_size * patch_size
                image = image[:, :height, :width]
                image_inputs.append(image)
    
            images_vggt.append(torch.stack(cur_images_vggt))
        
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            do_rescale=False
        )
        device = self.model.device

        # =====================================================================
        # input_ids 修补 + pixel_values 裁剪（仅在有历史特征缓存时执行）
        #
        # 问题背景：
        #   processor 按全分辨率计算所有帧的 <image_pad> 数量（in input_ids）
        #   和 image_grid_thw；但 forward() 的推理路径对历史帧做了渐进压缩，
        #   输出 token 数小于全分辨率。若不修补，<image_pad> 数量与实际 token 数
        #   不一致，导致 masked_scatter 报错。
        #
        # 解决方案：
        #   1. 用与 data_qwen.py / forward() 完全相同的压缩公式，计算每帧压缩后的 token 数
        #   2. 替换 input_ids 中历史帧的全分辨率 <image_pad> 块 → 压缩大小块
        #   3. 更新 image_grid_thw 中历史帧的 grid → 压缩 grid（供 get_rope_index 使用）
        #   4. 裁剪 pixel_values，只保留当前帧的 patch（历史帧特征来自缓存，不再编码）
        # =====================================================================
        if history_features:
            image_token_id = self.model.config.image_token_id
            orig_grid_thw = inputs['image_grid_thw']   # [n_imgs, 3]，全分辨率
            n_images = len(orig_grid_thw)

            mem_group_size = getattr(self.model.config, 'memory_group_size', 3)
            mem_max_order  = getattr(self.model.config, 'memory_max_compression_order', 3)

            # 计算每帧压缩后的 token 数和 grid
            full_tok = [int(thw[0] * thw[1] * thw[2]) // merge_size ** 2 for thw in orig_grid_thw]
            comp_tok = []
            comp_grid = []
            for idx, thw in enumerate(orig_grid_thw):
                if idx == n_images - 1:          # 当前帧：不压缩
                    comp_tok.append(full_tok[idx])
                    comp_grid.append(thw)
                else:
                    pos = n_images - 2 - idx     # 0 = 最近历史帧
                    r = get_progressive_compression_ratio(pos, mem_group_size, mem_max_order)
                    T, H, W = int(thw[0]), int(thw[1]), int(thw[2])
                    h_m, w_m = H // merge_size, W // merge_size
                    h_c, w_c = max(1, h_m // r), max(1, w_m // r)
                    comp_tok.append(T * h_c * w_c)
                    comp_grid.append(
                        torch.tensor([T, h_c * merge_size, w_c * merge_size],
                                     dtype=thw.dtype, device=thw.device)
                    )

            # 修补 input_ids：用压缩块替换全分辨率块（batch_size=1）
            seq = inputs['input_ids'][0].tolist()
            patched, img_idx, i = [], 0, 0
            while i < len(seq):
                if seq[i] == image_token_id and img_idx < n_images:
                    j = i
                    while j < len(seq) and seq[j] == image_token_id:
                        j += 1
                    patched.extend([image_token_id] * comp_tok[img_idx])
                    img_idx += 1
                    i = j
                else:
                    patched.append(seq[i])
                    i += 1
            inputs['input_ids'] = torch.tensor(
                [patched], dtype=inputs['input_ids'].dtype, device=inputs['input_ids'].device
            )
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            inputs['image_grid_thw'] = torch.stack(comp_grid).to(orig_grid_thw.device)

            # 裁剪 pixel_values：只保留当前帧（最后一帧）的 patch 数据
            # n_raw_patches[i] = T_i * H_i * W_i（T/H/W 来自原始全分辨率 grid_thw）
            n_raw = [int(thw[0] * thw[1] * thw[2]) for thw in orig_grid_thw]
            cur_start = sum(n_raw[:-1])
            inputs['pixel_values'] = inputs['pixel_values'][cur_start:]

        inputs["images_vggt"] = [feat.to(device) for feat in images_vggt]
        inputs = inputs.to(device)
    
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 24
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
        
        
        pad_token_id = self.tokenizer.pad_token_id
        cont = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            # 历史融合特征（来自 fused_feature_cache），传入 forward() 做渐进压缩
            # 若无历史（第一步或无缓存），传 None，触发 forward() 的兜底逻辑
            history_features=history_features if history_features else None,
        )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return answers




   
def eval():
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/streamvln')
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=12)  # 论文窗口 W=12
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help= "Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    parser.add_argument("--save_video_ratio", type=float, default=0.05, help="0~1")
    
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='rank')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu')
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--max_steps', default=400, type=int,
                        help='max_steps')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    
    args = parser.parse_args()
    set_seed(args.seed)
    init_distributed_mode(args)
    local_rank = args.local_rank

    model = JanusVLN_Inference(args.model_path, device=f"cuda:{local_rank}")

    evaluate(model, args)



def evaluate(model, args):
    
    world_size = get_world_size()

    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        epoch=0,
        args=args
    )
    sucs, spls, oss, ones, ep_num = evaluator.eval_action(get_rank()) 
    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)
    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]
    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(ones_all, ones)
    dist.barrier()
    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    ones_all = torch.cat(ones_all, dim=0)
    result_all = {
                    "sucs_all": (sum(sucs_all)/len(sucs_all)).item(),
                    "spls_all": (sum(spls_all)/len(spls_all)).item(),
                    "oss_all": (sum(oss_all)/len(oss_all)).item(),
                    "ones_all": (sum(ones_all)/len(ones_all)).item(),
                    'length': len(sucs_all)
                }
    
    print(result_all)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))

if __name__ == "__main__":
    eval()
