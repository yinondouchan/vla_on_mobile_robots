"""Mean/std scaling for state and action chunks, matching SmolVLA's MEAN_STD processors.

Stats are per action dimension and broadcast across the chunk. The forward
divisor is ``std + eps``. The inverse multiplies by ``std`` alone.
"""

from __future__ import annotations

import torch

EPS = 1e-8
NORM_STATS_NAME = "norm_stats.pt"


def norm_stats_from_dataset(stats: dict) -> dict[str, torch.Tensor]:
    """Pull ``observation.state`` and ``action`` mean/std from ``dataset.meta.stats``."""
    if not stats:
        raise ValueError("dataset has no stats; mean/std normalization needs meta/stats.json")

    def vector(feature: str, name: str) -> torch.Tensor:
        return torch.as_tensor(stats[feature][name], dtype=torch.float32).reshape(-1)

    return {
        "state_mean": vector("observation.state", "mean"),
        "state_std": vector("observation.state", "std"),
        "action_mean": vector("action", "mean"),
        "action_std": vector("action", "std"),
    }


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.to(dtype=x.dtype)
    std = std.to(dtype=x.dtype)
    return (x - mean) / (std + EPS)


def unnormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.to(dtype=x.dtype)
    std = std.to(dtype=x.dtype)
    return x * std + mean


def normalize_batch(batch: dict, stats: dict[str, torch.Tensor]) -> dict:
    batch["state"] = normalize(batch["state"], stats["state_mean"], stats["state_std"])
    batch["action"] = normalize(batch["action"], stats["action_mean"], stats["action_std"])
    return batch
