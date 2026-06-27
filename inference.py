from __future__ import annotations

from copy import copy

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.control_utils import predict_action


class LerobotInference:
    """Run a LeRobot policy from Hugging Face as a drop-in joystick replacement."""

    def __init__(
        self,
        policy_path: str,
        *,
        dataset_repo_id: str = "YinonDouchan/mobile_robot_lift_v1",
        robot_type: str = "custom_mobile_robot",
        grip_threshold: float = 0.5,
    ):
        self.policy_path = policy_path
        self.dataset_repo_id = dataset_repo_id
        self.robot_type = robot_type
        self.grip_threshold = grip_threshold

        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        policy_cfg.pretrained_path = policy_path

        ds_meta = LeRobotDatasetMetadata(dataset_repo_id)
        self._policy: PreTrainedPolicy = make_policy(policy_cfg, ds_meta=ds_meta)
        self._policy.eval()

        self._device = torch.device(policy_cfg.device)
        self._use_amp = policy_cfg.use_amp

        preprocessor_overrides = {
            "device_processor": {"device": str(self._device)},
        }
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_path,
            preprocessor_overrides=preprocessor_overrides,
        )

    def reset(self) -> None:
        """Reset policy internal state. Call when the environment is reset."""
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def predict(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        task: str | None = None,
    ) -> np.ndarray:
        """Run one policy step and return [forward, turn, grip, lift]."""
        observation = {
            "observation.state": np.asarray(state, dtype=np.float32),
        }
        for camera_name, image in images.items():
            observation[f"observation.images.{camera_name}"] = image

        action = predict_action(
            observation=copy(observation),
            policy=self._policy,
            device=self._device,
            preprocessor=self._preprocessor,
            postprocessor=self._postprocessor,
            use_amp=self._use_amp,
            task=task,
            robot_type=self.robot_type,
        )
        return action.squeeze(0).detach().cpu().numpy().astype(np.float64)

