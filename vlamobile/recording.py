from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import shutil
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from vlamobile.env import DEFAULT_STATE_NAMES

DEFAULT_ACTION_NAMES = ("forward", "turn", "grip", "lift")
DEFAULT_CAMERAS = {"sideview": (480, 640)}


@dataclass
class RecorderConfig:
    repo_id: str
    root: str | Path
    fps: int = 20
    robot_type: str = "custom_mobile_robot"
    action_names: tuple[str, ...] = DEFAULT_ACTION_NAMES
    state_names: tuple[str, ...] = DEFAULT_STATE_NAMES
    cameras: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(DEFAULT_CAMERAS))
    use_videos: bool = True
    streaming_encoding: bool = True
    vcodec: str = "libsvtav1"
    resume: bool = False
    overwrite_root: bool = False


def build_features(config: RecorderConfig) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(config.state_names),),
            "names": list(config.state_names),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(config.action_names),),
            "names": list(config.action_names),
        },
        "environment_state_qpos": {
            "dtype": "float64",
            "shape": (37,),
            "names": [],
        },
        "environment_state_qvel": {
            "dtype": "float64",
            "shape": (32,),
            "names": [],
        },
    }
    for camera_name, (height, width) in config.cameras.items():
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video" if config.use_videos else "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


class LeRobotRecorder:
    """Records MuJoCo teleop rollouts into a LeRobot v3 dataset (parquet + mp4)."""

    def __init__(self, config: RecorderConfig):
        self.config = config
        self._recording = False
        self._task: str | None = None
        self._dataset = self._open_dataset()

    def _open_dataset(self) -> LeRobotDataset:
        if self.config.resume:
            return LeRobotDataset.resume(
                self.config.repo_id,
                root=self.config.root,
                streaming_encoding=self.config.streaming_encoding,
                vcodec=self.config.vcodec
            )

        if self.config.overwrite_root and self.config.root is not None and os.path.exists(self.config.root):
            shutil.rmtree(self.config.root)

        return LeRobotDataset.create(
            repo_id=self.config.repo_id,
            fps=self.config.fps,
            features=build_features(self.config),
            root=self.config.root,
            robot_type=self.config.robot_type,
            use_videos=self.config.use_videos,
            streaming_encoding=self.config.streaming_encoding,
            vcodec=self.config.vcodec,
        )

    @property
    def dataset(self) -> LeRobotDataset:
        return self._dataset

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def task(self) -> str | None:
        return self._task

    @property
    def has_pending_frames(self) -> bool:
        return self._dataset.has_pending_frames()

    def start_recording(self, task: str) -> None:
        self._task = task
        self._recording = True
        

    def stop_recording(self) -> None:
        self._recording = False

    def add_frame(
        self,
        *,
        action: np.ndarray,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        environment_state: dict[str, np.ndarray] | None = None,
    ) -> None:
        if not self._recording:
            return
        if self._task is None:
            raise RuntimeError("Episode task not set. Call start_recording(task) first.")

        frame = {
            "task": self._task,
            "action": np.asarray(action, dtype=np.float32),
            "observation.state": np.asarray(state, dtype=np.float32),
        }

        if environment_state is not None:
            frame["environment_state_qpos"] = environment_state["qpos"]
            frame["environment_state_qvel"] = environment_state["qvel"]

        for camera_name, image in images.items():
            frame[f"observation.images.{camera_name}"] = image

        self._dataset.add_frame(frame)

    def save_episode(self) -> None:
        if not self.has_pending_frames:
            return
        self._dataset.save_episode()
        self._recording = False
        self._task = None

    def discard_episode(self) -> None:
        if self.has_pending_frames:
            self._dataset.clear_episode_buffer()
        self._recording = False
        self._task = None

    def finalize(self) -> None:
        self._dataset.finalize()

    def close(self) -> None:
        self.finalize()

    def __enter__(self) -> LeRobotRecorder:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
