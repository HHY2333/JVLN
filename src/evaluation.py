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
import time  # 用于 per-step / per-episode 耗时统计

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
from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationForJanusVLN
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ── 图像分辨率限制：必须与训练脚本 (trainzero3.sh) 保持严格一致 ──────────────────
# 训练: --max_pixels $((576*28*28)) = 451,584  --min_pixels $((16*28*28)) = 12,544
# 原始评估: max_pixels=1,605,632 / min_pixels=784，差了 3.56× / 16×
# 后果：评估时每帧生成的视觉 token 数远多于训练时，RoPE 位置编码分布偏移，
#       模型看到的特征与训练分布不一致，动作预测（尤其 STOP）变得不可靠。
# 修复：与 trainzero3.sh 保持完全一致。
min_pixels: int = 16 * 28 * 28   # = 12,544（与训练一致）
max_pixels: int = 576 * 28 * 28  # = 451,584（与训练一致）


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
        # 快速评估：计算每 rank 最多处理多少 episodes
        max_episodes = getattr(self.args, 'max_episodes', -1)
        per_rank_limit = (max_episodes + self.env_num - 1) // self.env_num if max_episodes > 0 else -1
        ep_count = 0   # 当前 rank 已处理的 episode 数
        # ── Bug2 修复（part1）：断点续跑时只读取已完成的 episode 列表供跳过判断 ────────
        # 原始代码：rank 0 会将历史结果加入 sucs/spls/oss/ones，其他 rank 不加；
        #   最终 all_gather 时 rank 0 的张量含 (历史+新) 条目，其余 rank 仅含新条目，
        #   导致 rank 0 的历史数据被重复统计、其他 rank 的历史数据被完全遗漏。
        # 修复：所有 rank 只读取 done_res（用于跳过已完成 episode），不向指标列表写入；
        #   最终指标由 rank 0 在 evaluate() 中直接读取完整 result.json 计算（见 part2）。
        if os.path.exists(os.path.join(self.output_path, 'result.json')):
            with open(os.path.join(self.output_path, 'result.json'), 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        res = json.loads(line)
                    except json.JSONDecodeError:
                        # 跳过损坏行，避免中断续跑
                        continue
                    # 仅记录 episode 级别的结果行（含 scene_id 字段）；跳过汇总统计行
                    if "scene_id" not in res:
                        continue
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
        
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            
            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"scene {scene_id}")
            for episode in episodes[idx::self.env_num]:
                # 快速评估：达到 per_rank_limit 时提前退出
                if per_rank_limit > 0 and ep_count >= per_rank_limit:
                    break

                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                print("episode start: ",episode_instruction)
                episode_id = episode.episode_id
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue

                env.current_episode = episode
                observations = env.reset()

                vis_frames = []
                step_id = 0
                ep_start = time.perf_counter()   # ① episode 开始时刻，用于计算总耗时与 s/step

                should_save_video = self.save_video and (random.random() < self.save_video_ratio)
                if should_save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}'), exist_ok=True)
                

                rgb_list = []
                time_ids = []
                action_seq = []
                # ② 帧张量缓存：{帧在 rgb_list 中的索引 → 预处理+裁剪后的 CPU Tensor}
                # 每帧只调用一次 load_and_preprocess_images，避免历史帧被反复重算
                frame_tensor_cache: dict = {}
                self.model.model.past_key_values_vggt = None
                # ③ 死循环检测：记录最近执行的动作字符串（执行前已映射为 int，这里记录映射前的原始输出）
                # 若最近 loop_window 步内动作全在 {TURN_LEFT, TURN_RIGHT} 之间交替，
                # 强制执行 MOVE_FORWARD 打破僵局，防止智能体原地旋转浪费步数。
                recent_actions: list = []
                loop_window: int = getattr(self.args, 'loop_window', 6)
                loop_force_steps: int = 0   # 剩余强制 MOVE_FORWARD 步数
                force_stopped: bool = False  # 标记是否因超过 max_steps 被迫 STOP（非模型主动）

                while not env.episode_over:
                    # ── Bug6 修复：达到最大步数后直接 STOP，不再调用模型推理，节省算力 ──
                    # 原始代码在推理后才检查 max_steps，导致最后一步白白做了一次前向计算。
                    if step_id >= self.args.max_steps:
                        observations = env.step(0)   # 0 = STOP
                        step_id += 1
                        force_stopped = True   # 标记为强制截止，后续不计入成功
                        break

                    time_ids.append(step_id)
                    rgb = observations["rgb"]

                    image = Image.fromarray(rgb).convert('RGB')
                    rgb_list.append(image)
                    # 新帧立刻预处理并缓存（PIL→归一化Tensor→裁剪），O(1) 额外开销；
                    # 后续步骤直接复用缓存，避免历史帧在每步被反复重算。
                    new_idx = len(rgb_list) - 1
                    frame_tensor_cache[new_idx] = self.model.preprocess_single_image(image)

                    info = env.get_metrics()

                    # 构建历史帧索引：间隔 STRIDE=4 向前回溯，最多取 num_history 帧历史
                    # + 当前帧（共 num_history+1 帧），与 create_data.py 采样策略完全一致。
                    STRIDE = 4
                    back_indices = list(range(new_idx, max(new_idx - self.num_history * STRIDE, -1), -STRIDE))
                    image_indices = list(reversed(back_indices))
                    images = [rgb_list[i] for i in image_indices]

                    # 将对应的预处理张量一并传入，call_model 直接复用缓存，跳过重复 decode/crop
                    precomputed_tensors = [frame_tensor_cache[i] for i in image_indices]

                    step_t0 = time.perf_counter()   # per-step 推理计时开始
                    raw_model_output = self.model.call_model(
                        images, episode_instruction, step_id,
                        precomputed_image_tensors=precomputed_tensors,
                    )[0]
                    step_elapsed = time.perf_counter() - step_t0

                    # ── Bug1 修复：对模型输出进行关键词提取，不区分大小写，支持带冗余文本的输出 ──
                    # 原始代码直接将原始字符串查 actions2idx 字典；若模型输出含多余文字、换行符
                    # 或大小写不一致（如 "I should move_forward"），则匹配失败，动作被错误置为
                    # STOP（action=0），严重低估导航性能（尤其在 val_unseen 上影响显著）。
                    # 修复：在大写化输出中按优先级顺序搜索合法关键词，首次命中则采用；
                    #       全部未命中才兜底为 STOP。
                    upper_output = raw_model_output.strip().upper()
                    action = "STOP"   # 防御性默认值，确保后续代码安全
                    for key in self.actions2idx:
                        if key in upper_output:
                            action = key
                            break

                    # ── Bug4 修复：死循环检测与强制干预（recent_actions 记录实际执行动作）────
                    # 原始代码先将原始模型输出追加到 recent_actions，再执行强制覆盖；
                    # 导致强制步骤中仍将 TURN_LEFT/TURN_RIGHT 记入检测窗口，使窗口快速
                    # 再次充满，提前重复触发死循环检测，实际强制步数被稀释。
                    # 修复：强制步骤直接跳过 recent_actions 记录，只在正常步骤中累积并检测。
                    if loop_force_steps > 0:
                        # 本步为强制 MOVE_FORWARD 阶段，消耗一次计数，不记录到检测窗口
                        action = "MOVE_FORWARD"
                        loop_force_steps -= 1
                        print(
                            f"  [rank={get_rank()} | step={step_id:03d}] "
                            f"[LOOP-BREAK] forced MOVE_FORWARD (remain={loop_force_steps})",
                            flush=True,
                        )
                    else:
                        # 正常步骤：将已归一化的动作加入检测窗口（最多保留 loop_window 条）
                        recent_actions.append(action)
                        if len(recent_actions) > loop_window:
                            recent_actions.pop(0)
                        # 最近 loop_window 步全为 TURN_LEFT/TURN_RIGHT 且两者均出现 → 死循环
                        if (
                            len(recent_actions) == loop_window
                            and all(a in ("TURN_LEFT", "TURN_RIGHT") for a in recent_actions)
                            and "TURN_LEFT" in recent_actions
                            and "TURN_RIGHT" in recent_actions
                        ):
                            action = "MOVE_FORWARD"
                            loop_force_steps = 1   # 额外强制执行 1 步
                            recent_actions.clear()   # 清空窗口，重新开始检测
                            print(
                                f"  [rank={get_rank()} | step={step_id:03d}] "
                                f"[LOOP-DETECT] TURN_LEFT/RIGHT loop over {loop_window} steps → force MOVE_FORWARD",
                                flush=True,
                            )

                    # 逐步打印：rank / step 进度 / 模型原始输出 / 最终执行动作 / 单步耗时 / 历史帧数
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S.%f')}]"
                        f"  [rank={get_rank()} | step={step_id:03d}/{self.args.max_steps}"
                        f" | hist={len(image_indices)-1:02d}] "
                        f"raw={raw_model_output.strip()!r}  action={action!r:<14s}  {step_elapsed:.2f}s/step",
                        flush=True,
                    )

                    if info['top_down_map'] is not None and should_save_video:
                        frame = observations_to_image({'rgb': observations['rgb']}, info)
                        vis_frames.append(frame)

                    # 动作字符串 → 环境整数索引（经 Bug1 修复后 action 已归一化，此处必然命中）
                    if action in self.actions2idx:
                        action = self.actions2idx[action][0]
                    else:
                        action = 0   # 防御性兜底，理论上不会触达

                    observations = env.step(action)
                    step_id += 1

                process_bar.update(1)
                metrics = env.get_metrics()
                ep_elapsed = time.perf_counter() - ep_start   # episode 总耗时

                # 强制超步截止：模型并未主动选择 STOP，不应计为成功
                # Habitat 的 success 判定基于 STOP 时的距离，强制 STOP 时可能恰好在终点
                # 附近而被误判为成功；此处将其归零，保证 success/spl 语义正确。
                if force_stopped:
                    metrics = dict(metrics)   # 转为普通 dict 以允许修改
                    metrics['success'] = 0.0
                    metrics['spl']     = 0.0
                if should_save_video:
                    images_to_video(
                        vis_frames, os.path.join(self.output_path, f'vis_{self.epoch}'), f'{scene_id}_{episode_id}', fps=6, quality=9
                    )
                vis_frames.clear()
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])
                # ⑥ episode 结束：打印详细摘要（含 s/step、累计均值）
                print(
                    f"[rank={get_rank()}] DONE {scene_id}_{episode_id} "
                    f"steps={step_id} "
                    f"{'[FORCE-STOP] ' if force_stopped else ''}"
                    f"elapsed={ep_elapsed:.1f}s ({ep_elapsed/max(step_id,1):.2f}s/step) | "
                    f"suc={metrics['success']} "
                    f"spl={metrics['spl']:.4f} "
                    f"os={metrics['oracle_success']:.4f} "
                    f"ne={metrics['distance_to_goal']:.3f}m | "
                    f"cumul: suc={np.mean(sucs):.2%} spl={np.mean(spls):.2%} ne={np.mean(ones):.3f}",
                    flush=True,
                )
                # ⑦ 实时更新进度条后缀，显示累计指标
                process_bar.set_postfix(OrderedDict([
                    ('N',   len(sucs)),
                    ('suc', f'{np.mean(sucs):.2%}'),
                    ('spl', f'{np.mean(spls):.2%}'),
                    ('ne',  f'{np.mean(ones):.3f}'),
                ]))
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "force_stopped": force_stopped,
                    "episode_instruction": episode_instruction
                }
                
                with open(os.path.join(self.output_path, f'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")
                ep_count += 1

            # 快速评估：当前场景已达上限，不再遍历后续场景
            if per_rank_limit > 0 and ep_count >= per_rank_limit:
                break

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

    def preprocess_single_image(self, pil_image: Image.Image) -> torch.Tensor:
        """
        将单张 PIL Image 预处理为裁剪后的 CPU 张量，供外部逐帧缓存使用。

        流程与 call_model 内部完全一致：
          1. load_and_preprocess_images  → 归一化到 [0,1] 的 (C,H,W) Tensor
          2. 裁剪至 patch_size * merge_size 的整数倍（VLM patch 对齐要求）

        每帧只调用一次；eval_action 在新帧进入 rgb_list 时立刻缓存，
        后续步骤直接复用，避免历史帧在每步被反复重算。

        Returns:
            torch.Tensor: shape (C, H', W')，CPU，已归一化，已裁剪。
        """
        patch_size = self.processor.image_processor.patch_size
        merge_size  = self.processor.image_processor.merge_size
        img = load_and_preprocess_images([pil_image])[0]   # (C, H, W)
        _, h, w = img.shape
        if (w // patch_size) % merge_size > 0:
            w = w - (w // patch_size) % merge_size * patch_size
        if (h // patch_size) % merge_size > 0:
            h = h - (h // patch_size) % merge_size * patch_size
        return img[:, :h, :w]   # CPU Tensor，不上 GPU

    def call_model(
        self,
        observations,
        task,
        step_id,
        add_frame_index: bool = False,
        gen_kwargs: dict = None,   # Bug5 修复：避免可变默认参数在多次调用间共享同一字典对象
        precomputed_image_tensors: list = None,
        # 预处理好的帧张量列表（与 observations 一一对应，最后一个为当前帧）。
        # 由 eval_action 中的 frame_tensor_cache 提供；若非 None，则跳过内部
        # load_and_preprocess_images + crop，直接使用缓存张量，显著减少重复计算。
    ):
        # =========================================================================
        # 分离历史帧与当前帧
        # =========================================================================
        # observations 是一个 PIL Image 列表，最后一帧为当前观测，其余为历史帧。
        # 设计约定：
        #   - 当前帧（current_obs）: 作为常规 image token 嵌入 prompt，
        #                            同时走 VGGT 空间对齐（images_vggt）。
        #   - 历史帧（history_obs）: 不出现在 prompt 的 image token 中，
        #                            而是通过 history_pixel_values 送入模型，
        #                            在 forward() 中经过渐进空间压缩后前置到
        #                            inputs_embeds，参与因果注意力。
        if isinstance(observations, (list, tuple)) and len(observations) > 1:
            history_obs = observations[:-1]   # 历史帧（PIL Image list，旧 → 新）
            current_obs = observations[-1]    # 当前帧（PIL Image）
        elif isinstance(observations, (list, tuple)) and len(observations) == 1:
            history_obs = []
            current_obs = observations[0]
        else:
            # 单张 PIL Image，无历史
            history_obs = []
            current_obs = observations

        # ── 从预处理缓存中提取当前帧/历史帧张量（如有）────────────────────────
        # precomputed_image_tensors 与 observations 顺序对应：
        #   最后一个 → 当前帧预处理张量；其余 → 历史帧预处理张量。
        # 每个张量已由 preprocess_single_image 完成归一化+裁剪，可直接使用。
        if precomputed_image_tensors is not None:
            precomputed_current_tensor   = precomputed_image_tensors[-1]          # Tensor(C,H,W) CPU
            precomputed_history_tensors  = list(precomputed_image_tensors[:-1])   # list[Tensor]
        else:
            precomputed_current_tensor  = None
            precomputed_history_tensors = None

        messages = []
        # ── Prompt 对齐修复 ─────────────────────────────────────────────────────
        # 训练阶段（data_qwen.py preprocess_qwen_2_visual）固定使用：
        #   System: "You are a helpful assistant."
        # 且多帧样本经 _get_item 裁剪后，User 文本仅保留最后一个 <image> 之后的内容：
        #   "<image>\n Your task is to {instruction}\n You should take one of the following actions:\n..."
        # 原始评估代码使用了完全不同的 System 描述和 User 文本，导致模型的
        # STOP 触发条件完全失配——模型从未见过评估格式，无法在正确时机输出 STOP。
        # 修复：将评估 prompt 与训练格式严格对齐。
        message = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        # 与训练数据保持完全一致：data_qwen.py 会将 <image>\n 替换为 <image>（去掉换行），
        # 因此训练时 <image> 后接的是空格而非 \n，评估此处也用空格对齐。
        context = f" Your task is to {task}\n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size
        # ── Bug3 修复：去除 `for i in enumerate([task]):` 多余循环 ──────────────
        # 原始代码用 enumerate([task]) 只循环一次，loop 变量 i=(0,task) 从未在循环体内使用；
        # 此处直接内联循环体，逻辑完全等价，消除迷惑性的单次伪循环。
        # 历史帧不在此处放入 message，而是通过 history_pixel_values 独立处理。
        if isinstance(current_obs, Image.Image):
            message.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": current_obs},
                    {"type": "text", "text": context},
                ]
            })
        else:
            message.append({"role": "user", "content": [{"type": "text", "text": context}]})
        messages.append(message)

        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
        images_vggt = []
        image_inputs = []
        for message in messages:
            vision_info = extract_vision_info(message)
            cur_images_vggt = []
            # 此时 vision_info 只包含当前帧（已不含历史帧）
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
    
                if precomputed_current_tensor is not None:
                    # ★ 使用缓存张量（已裁剪），跳过 PIL→Tensor 转换与 crop
                    image = precomputed_current_tensor
                else:
                    assert isinstance(image, Image.Image), f"Unsupported image type: {type(image)}"
                    image = load_and_preprocess_images([image])[0]
                    # 裁剪为 patch_size * merge_size 的整数倍
                    _, height, width = image.shape
                    if (width // patch_size) % merge_size > 0:
                        width = width - (width // patch_size) % merge_size * patch_size
                    if (height // patch_size) % merge_size > 0:
                        height = height - (height // patch_size) % merge_size * patch_size
                    image = image[:, :height, :width]

                if i == len(vision_info) - 1:
                    # 最后一帧（当前帧）送入 VGGT 做空间对齐
                    cur_images_vggt.append(image)
                image_inputs.append(image)
    
            images_vggt.append(torch.stack(cur_images_vggt))

        # =========================================================================
        # 单独预处理历史帧，得到 history_pixel_values 和 history_grid_thw
        # =========================================================================
        # 历史帧走与当前帧完全相同的预处理管线（load → crop），
        # 然后通过 image_processor 独立计算 pixel_values 和 image_grid_thw，
        # 不参与 prompt 的 image token 计数。
        history_pixel_values = None
        history_grid_thw = None
        history_frame_counts = None

        if len(history_obs) > 0:
            if precomputed_history_tensors is not None:
                # ★ 直接使用外部缓存的历史帧张量（已预处理+裁剪），
                # 无需对每一帧再次执行 load_and_preprocess_images。
                # 在 N 步 episode 中，此处节省的计算量为 O(num_history * N)。
                hist_image_inputs = precomputed_history_tensors
            else:
                hist_image_inputs = []
                for hist_pil in history_obs:
                    # 与当前帧相同的加载 + 预处理流程（无缓存时的回退路径）
                    hist_img = load_and_preprocess_images([hist_pil])[0]
                    # 裁剪为 patch_size * merge_size 的整数倍（与当前帧一致）
                    _, h_px, w_px = hist_img.shape
                    if (w_px // patch_size) % merge_size > 0:
                        w_px = w_px - (w_px // patch_size) % merge_size * patch_size
                    if (h_px // patch_size) % merge_size > 0:
                        h_px = h_px - (h_px // patch_size) % merge_size * patch_size
                    hist_img = hist_img[:, :h_px, :w_px]
                    hist_image_inputs.append(hist_img)

            # 调用 image_processor 独立处理历史帧，
            # 将预处理后的张量（不再 rescale）转为 pixel_values 和 image_grid_thw
            device = self.model.device
            hist_proc_out = self.processor.image_processor(
                images=hist_image_inputs,
                return_tensors="pt",
                do_rescale=False,   # 已由 load_and_preprocess_images 归一化
            )
            history_pixel_values = hist_proc_out["pixel_values"].to(device)
            history_grid_thw    = hist_proc_out["image_grid_thw"].to(device)
            # 单 batch 推理：所有历史帧都属于第 0 个样本
            history_frame_counts = [len(hist_image_inputs)]

        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            do_rescale=False
        )
        device = self.model.device

        inputs["images_vggt"] = [feat.to(device) for feat in images_vggt]
        # 将历史帧渐进压缩所需的输入追加到 inputs，
        # generate() 会在 prefill 阶段将其传给 forward()，step>0 时自动置 None。
        inputs["history_pixel_values"] = history_pixel_values
        inputs["history_grid_thw"]     = history_grid_thw
        inputs["history_frame_counts"] = history_frame_counts
        inputs = inputs.to(device)
    
        # Bug5 修复：gen_kwargs=None 时初始化空字典；若调用方传入了字典则浅拷贝，
        # 防止修改调用方持有的原始对象（防御性编程）。
        gen_kwargs = dict(gen_kwargs) if gen_kwargs else {}
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 24
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
        
        
        pad_token_id = self.tokenizer.pad_token_id
        # torch.inference_mode 比 no_grad 更激进：完全禁用梯度引擎并优化张量内存视图，
        # 推理阶段可节省约 5-10% 显存并轻微提速；对 generate() 的输出无任何影响。
        with torch.inference_mode():
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
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
    parser.add_argument("--num_history", type=int, default=16)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help= "Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    parser.add_argument("--save_video_ratio", type=float, default=1, help="0~1")
    
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
    parser.add_argument('--max_episodes', default=-1, type=int,
                        help='快速评估时限制总 episode 数（-1 表示全量），各 rank 均分；'
                             '例如 --max_episodes 12 + 2 GPU = 每卡评估 6 个 episode')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--loop_window", type=int, default=6,
                        help="最近 N 步全为 TURN_LEFT/TURN_RIGHT 交替时判定为死循环并强制 MOVE_FORWARD（默认 6）")

    
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
    # 打印本次新评估的 episode 统计（不含续跑前已完成的 episode）
    new_count = len(sucs_all)
    if new_count > 0:
        new_result = {
            "new_sucs":  (sum(sucs_all)  / new_count).item(),
            "new_spls":  (sum(spls_all)  / new_count).item(),
            "new_oss":   (sum(oss_all)   / new_count).item(),
            "new_ones":  (sum(ones_all)  / new_count).item(),
            "new_count": new_count,
        }
        print(f"[New episodes this run] {new_result}", flush=True)

    if get_rank() == 0:
        # ── Bug2 修复（part2）：从 result.json 读取所有 episode 结果，计算完整最终指标 ──
        # 原始代码依赖 all_gather 汇聚指标：rank 0 的张量含 (历史+新)，其余 rank 仅含新数据，
        # all_gather 后 rank 0 的历史数据被重复计算，其他 rank 的历史数据被遗漏，结果错误。
        # 修复：rank 0 读取完整 result.json（含续跑前所有 episode），统一计算最终指标。
        result_file = os.path.join(args.output_path, 'result.json')
        all_sucs, all_spls, all_oss, all_nes = [], [], [], []
        with open(result_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    res = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 仅统计 episode 级别的结果行（含 scene_id 字段），跳过汇总行
                if "scene_id" not in res:
                    continue
                all_sucs.append(res['success'])
                all_spls.append(res['spl'])
                all_oss.append(res['os'])
                all_nes.append(res['ne'])
        result_all = {
            "sucs_all": float(np.mean(all_sucs)) if all_sucs else 0.0,
            "spls_all": float(np.mean(all_spls)) if all_spls else 0.0,
            "oss_all":  float(np.mean(all_oss))  if all_oss  else 0.0,
            "ones_all": float(np.mean(all_nes))  if all_nes  else 0.0,
            "length":   len(all_sucs),
        }
        print(f"[Final complete metrics] {result_all}", flush=True)
        with open(result_file, 'a') as f:
            f.write(json.dumps(result_all) + "\n")

if __name__ == "__main__":
    eval()
