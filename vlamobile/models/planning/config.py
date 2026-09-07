"""Configuration for the SmolVLM planning module."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PlannerConfig:
    """Configuration for `SmolVLMPlanner`.

    Plain dataclass (not a LeRobot policy config): the planner sits above the
    VLA policy and is not itself a policy, so no registration is needed.
    """

    # Model loading (mirrors MiniSAModel._init_vision_language_encoder).
    model_name: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    device: str | None = None  # resolved to "cuda" if available, else "cpu"
    load_in_4bit: bool = True
    torch_dtype: str = "bfloat16"

    # Generation parameters. temperature == 0 means greedy decoding
    # (deterministic), which is the default for reproducible plans.
    max_new_tokens: int = 512
    temperature: float = 0.0

    # History is truncated to this many most-recent entries (oldest dropped
    # first; the plan header is always kept) to bound prompt length.
    max_history_entries: int = 30

    # Planning limits.
    max_subtasks: int = 8

    # Image preprocessing: same processor sizing as MiniSA.
    image_longest_edge: int = 512

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.max_subtasks < 1:
            raise ValueError(f"max_subtasks must be >= 1, got {self.max_subtasks}")
        if self.max_history_entries < 1:
            raise ValueError(
                f"max_history_entries must be >= 1, got {self.max_history_entries}"
            )
