from __future__ import annotations

from copy import copy

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.common.control_utils import predict_action

from vlamobile.models.planning.config import PlannerConfig
from vlamobile.models.planning.planning import Planner


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

        self._ds_meta = LeRobotDatasetMetadata(dataset_repo_id, revision="main")
        self._policy: PreTrainedPolicy = make_policy(policy_cfg, ds_meta=self._ds_meta)
        self._policy.eval()

        self._device = torch.device(policy_cfg.device)
        self._use_amp = policy_cfg.use_amp

        preprocessor_overrides = {
            "device_processor": {"device": str(self._device)},
        }
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_path,
            preprocessor_overrides=preprocessor_overrides
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

        return action.detach().cpu().numpy().astype(np.float64)


class LerobotInferenceWithPlanner:
    """Hierarchical controller: high-level planner + low-level VLA execution.

    Drop-in replacement for :class:`LerobotInference`. On each episode it
    decomposes the composite ``task`` into subtasks, periodically monitors
    progress via the planner, and conditions the VLA on
    ``planner.current_subtask()``.
    """

    def __init__(
        self,
        policy_path: str,
        *,
        dataset_repo_id: str = "YinonDouchan/mobile_robot_lift_v1",
        robot_type: str = "custom_mobile_robot",
        grip_threshold: float = 0.5,
        planner: Planner | None = None,
        planner_cfg: PlannerConfig | None = None,
        predefined_tasks: list[str] | None = None,
        monitor_every: int = 30,
    ):
        self.policy = LerobotInference(
            policy_path,
            dataset_repo_id=dataset_repo_id,
            robot_type=robot_type,
            grip_threshold=grip_threshold,
        )
        self.planner = (
            planner
            if planner is not None
            else Planner(cfg=planner_cfg, predefined_tasks=predefined_tasks)
        )
        self.predefined_tasks = predefined_tasks
        self.monitor_every = monitor_every
        self._frame_idx = 0
        self._planned_task: str | None = None

    def reset(self) -> None:
        """Reset VLA policy and planner state for a new episode."""
        self.policy.reset()
        self.planner.reset()
        self._frame_idx = 0
        self._planned_task = None

    def predict(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        task: str | None = None,
    ) -> np.ndarray:
        """Decompose/monitor as needed, then run one VLA step on the active subtask."""
        needs_plan = self.planner.state is None or (
            task is not None and task != self._planned_task
        )
        if needs_plan:
            if task is None:
                raise RuntimeError(
                    "no plan yet; pass a composite task on the first predict(...) call"
                )
            self.planner.decompose(
                task, images, predefined_tasks=self.predefined_tasks
            )
            self._planned_task = task
            self._frame_idx = 0

        if (
            self.monitor_every > 0
            and self._frame_idx > 0
            and self._frame_idx % self.monitor_every == 0
        ):
            self.planner.step(images, step=self._frame_idx)

        subtask = self.planner.current_subtask()
        action = self.policy.predict(
            state=state,
            images=images,
            task=subtask if subtask else task,
        )
        self._frame_idx += 1
        return action


