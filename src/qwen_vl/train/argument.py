import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    vggt_model_path: str = field(default="facebook/VGGT-1B/")
    lam: float = field(default=0.2)
    distill_loss_weight: float = field(default=1.0)
    reference_frame: str = field(default="last")


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    max_samples: int = field(default=-1)
    shuffle: bool = field(default=True)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    stop_at_step: int = field(
        default=-1,
        metadata={"help": "Stop training at this global step (enables staged train+eval loops in shell scripts)."},
    )
    total_steps: int = field(
        default=-1,
        metadata={"help": "真实训练总步数（用于全局进度条显示）。分段训练时 max_steps 被 stop_at_step 覆盖，"
                          "total_steps 保留原始总步数供进度回调使用。"},
    )
