"""Drop-in inference wrapper around :class:`FrozenVLAModel`.

``predict`` matches :class:`vlamobile.inference.LerobotInference`: one action per
call, shaped ``(1, action_dim)``. The backbone runs when the queued chunk is
empty or the task string changes.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from .config import FrozenVLAConfig
from .model import FrozenVLAModel, _MODEL_CONFIG_NAME
from .norm import NORM_STATS_NAME, normalize, unnormalize

_DEFAULT_CAMERA_KEYS = ("robotfrontview_high", "robotfrontview")


class FrozenVLAInference:
    """Run a frozen-VLA checkpoint with the ``LerobotInference`` call contract."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        camera_keys: list[str] | None = None,
        n_action_steps: int | None = None,
    ):
        checkpoint_dir = Path(checkpoint_dir)
        cfg_dict = torch.load(
            checkpoint_dir / _MODEL_CONFIG_NAME,
            map_location="cpu",
            weights_only=True,
        )
        cfg = FrozenVLAConfig(**cfg_dict)
        stats_path = checkpoint_dir / NORM_STATS_NAME
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"missing {NORM_STATS_NAME} in {checkpoint_dir}; retrain so the checkpoint includes dataset mean/std"
            )
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        self.state_mean = stats["state_mean"]
        self.state_std = stats["state_std"]
        self.action_mean = stats["action_mean"]
        self.action_std = stats["action_std"]

        self.model = FrozenVLAModel(cfg)
        self.model.load_action_head(checkpoint_dir)
        self.model.eval()

        self.camera_keys = list(camera_keys) if camera_keys is not None else list(_DEFAULT_CAMERA_KEYS)
        self.n_action_steps = cfg.action_chunk_size if n_action_steps is None else n_action_steps
        self._queue: deque[np.ndarray] = deque()
        self._task: str | None = None

    def reset(self) -> None:
        """Drop the queued action chunk. Call when the environment is reset."""
        self._queue.clear()
        self._task = None

    def predict(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        task: str | None = None,
    ) -> np.ndarray:
        """Return the next action as ``float64`` with shape ``(1, action_dim)``."""
        if task is None:
            raise ValueError("task is required")
        if task != self._task:
            self._queue.clear()
            self._task = task
        if not self._queue:
            self._enqueue_chunk(state, images, task)
        return self._queue.popleft()

    def _enqueue_chunk(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        task: str,
    ) -> None:
        state_tensor = normalize(
            torch.from_numpy(np.asarray(state, dtype=np.float32).reshape(1, -1)),
            self.state_mean,
            self.state_std,
        )
        sample_images = [images[key] for key in self.camera_keys]
        with torch.no_grad():
            actions = self.model(
                {
                    "images": [sample_images],
                    "task": [task],
                    "state": state_tensor,
                }
            )
        chunk = unnormalize(
            actions[0, : self.n_action_steps].detach().cpu(),
            self.action_mean,
            self.action_std,
        ).numpy().astype(np.float64)
        for row in chunk:
            self._queue.append(row.reshape(1, -1))
