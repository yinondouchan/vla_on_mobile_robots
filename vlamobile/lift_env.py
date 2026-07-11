from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

def _default_model_path() -> Path:
    here = Path(__file__).resolve().parent
    for base in (here.parent, here):
        path = base / "assets/mobile_robot_lift/scene.xml"
        if path.exists():
            return path
    return here.parent / "assets/mobile_robot_lift/scene.xml"


DEFAULT_MODEL_PATH = _default_model_path()
DEFAULT_CAMERA = "sideview"
DEFAULT_IMAGE_SIZE = (480, 640)
DEFAULT_STATE_NAMES = ("base_yaw", "lift_state", "grip_r", "grip_l")

MAX_VELOCITY = 50.0
TURN_GAIN = 0.5
MAX_LIFT = 0.15
GRIP_OPEN = 0.07
GRIP_CLOSED = 0.0


def mobile_robot_state(data: mujoco.MjData) -> np.ndarray:
    """Proprioceptive state for the custom mobile robot (free-base + gripper)."""
    qpos = data.qpos
    qw, qx, qy, qz = qpos[3:7]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return np.array(
        # base_yaw, lift_state, grip1, grip2
        [yaw, qpos[13], qpos[14], qpos[15]],
        dtype=np.float32,
    )


def reset_joint_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    rng: np.random.Generator,
    x_range: tuple[float, float] = (-1.0, 1.0),
    y_range: tuple[float, float] = (-1.0, 1.0),
    q_range: tuple[float, float] = (-np.pi, np.pi),
) -> None:
    x = rng.uniform(*x_range)
    y = rng.uniform(*y_range)
    q = rng.uniform(*q_range)
    joint_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)]
    joint_qpos = data.qpos[joint_adr : joint_adr + 7]
    joint_qpos[:2] = [x, y]
    joint_qpos[3:7] = [np.cos(q / 2), 0, 0, np.sin(q / 2)]


class LiftEnv(gym.Env):
    """Basic Gymnasium environment for the mobile lift MuJoCo scene."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        render_mode: str | None = "rgb_array",
        camera_name: str = DEFAULT_CAMERA,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        include_images: bool = True,
        max_episode_steps: int = 1000,
        frame_skip: int = 20, # set default output rate to 30Hz while simulation rate stays at 600Hz
        hide_small_cube: bool = False,
        hide_medium_cube: bool = False,
        hide_large_cube: bool = False,
        hide_red_platform: bool = False,
        hide_green_platform: bool = False,
        hide_blue_platform: bool = False,
    ) -> None:
        super().__init__()

        self.model_path = Path(model_path)
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.image_height, self.image_width = image_size
        self.include_images = include_images
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.hide_small_cube = hide_small_cube
        self.hide_medium_cube = hide_medium_cube
        self.hide_large_cube = hide_large_cube
        self.hide_red_platform = hide_red_platform
        self.hide_green_platform = hide_green_platform
        self.hide_blue_platform = hide_blue_platform

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if self.camera_id < 0:
            raise ValueError(f"Camera '{self.camera_name}' not found in {self.model_path}")

        self.renderer: mujoco.Renderer | None = None
        if self.include_images or self.render_mode is not None:
            self.renderer = mujoco.Renderer(self.model, height=self.image_height, width=self.image_width)

        self._rng = np.random.default_rng()
        self._elapsed_steps = 0

        self._setup_spaces()

    def _setup_spaces(self) -> None:
        observation_spaces: dict[str, spaces.Space] = {
            "observation.state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4,),
                dtype=np.float32,
            ),
        }
        if self.include_images:
            observation_spaces[f"observation.images.{self.camera_name}"] = spaces.Box(
                low=0,
                high=255,
                shape=(self.image_height, self.image_width, 3),
                dtype=np.uint8,
            )
        self.observation_space = spaces.Dict(observation_spaces)

        self.action_space = spaces.Box(
            low=np.array([-MAX_VELOCITY, -MAX_VELOCITY, 0.0, 0.0], dtype=np.float32),
            high=np.array([MAX_VELOCITY, MAX_VELOCITY, 1.0, MAX_LIFT], dtype=np.float32),
            dtype=np.float32,
        )

        self.environment_state_space = spaces.Dict({
            "qpos": spaces.Box(
                low=-np.inf, high=np.inf, shape=self.data.qpos.shape, dtype=np.float64
            ),
            "qvel": spaces.Box(
                low=-np.inf, high=np.inf, shape=self.data.qvel.shape, dtype=np.float64
            )
        })


    def _apply_action(self, action: np.ndarray) -> None:
        forward = float(action[0]) * MAX_VELOCITY
        turn = float(action[1]) * MAX_VELOCITY * TURN_GAIN
        grip = float(np.clip(action[2], 0.0, 1.0))
        lift = float(action[3]) * MAX_LIFT
        self.data.actuator("right_motor").ctrl = forward + turn
        self.data.actuator("left_motor").ctrl = forward - turn
        self.data.actuator("gripper_lift").ctrl = lift

        grip_pos = GRIP_OPEN + (GRIP_CLOSED - GRIP_OPEN) * grip
        self.data.actuator("gripper_right_pos").ctrl = grip_pos
        self.data.actuator("gripper_left_pos").ctrl = grip_pos

    def _get_observation(self) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {
            "observation.state": mobile_robot_state(self.data),
        }
        if self.include_images:
            assert self.renderer is not None
            self.renderer.update_scene(self.data, camera=self.camera_id)
            obs[f"observation.images.{self.camera_name}"] = self.renderer.render()

        return obs

    def _compute_reward(self) -> float:
        return 0.0

    def get_env_state(self) -> dict[str, np.ndarray]:
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy()}


    def reset_poses(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:

        y_range_small = (-1.0, 0.8) if not self.hide_small_cube else (100.0, 100.0)
        y_range_medium = (-1.0, 0.8) if not self.hide_medium_cube else (100.0, 100.0)
        y_range_large = (-1.0, 0.8) if not self.hide_large_cube else (100.0, 100.0)

        reset_joint_pose(model, data, "base_joint", rng, y_range=(-2.0, -1.0))
        reset_joint_pose(model, data, "cube_joint0", rng, y_range=y_range_small)
        reset_joint_pose(model, data, "cube_joint1", rng, y_range=y_range_medium)
        reset_joint_pose(model, data, "cube_joint2", rng, y_range=y_range_large)

        if self.hide_red_platform:
            self._set_platform_pos(model, "box1", (10.0, 100.0, 0.0))
        if self.hide_green_platform:
            self._set_platform_pos(model, "box2", (20.0, 100.0, 0.0))
        if self.hide_blue_platform:
            self._set_platform_pos(model, "box3", (30.0, 100.0, 0.0))

    @staticmethod
    def _set_platform_pos(
        model: mujoco.MjModel,
        body_name: str,
        pos: tuple[float, float, float]
    ) -> None:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        model.body_pos[body_id] = pos


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        if options is not None and "initial_state" in options and options["initial_state"] is not None:
            initial_state = options["initial_state"]
            self.data.qpos[:] = initial_state["qpos"]
            self.data.qvel[:] = initial_state["qvel"]
        else:
            self.reset_poses(self.model, self.data, self._rng)
            initial_state = self.get_env_state()

        # set grippers to be initially open and lift to be initially down
        self.data.qpos[13:16] = GRIP_OPEN
            
        mujoco.mj_forward(self.model, self.data)

        self.renderer.update_scene(self.data, camera=self.camera_id)

        self._elapsed_steps = 0
        observation = self._get_observation()
        info = {"task": options.get("task") if options else None, "initial_state": initial_state}
        return observation, info

    def step(self, action: Optional[np.ndarray] = None) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        if action is not None:
            action = np.asarray(action, dtype=np.float32)
            self._apply_action(action)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._elapsed_steps += 1
        observation = self._get_observation()
        reward = self._compute_reward()
        terminated = False
        truncated = self._elapsed_steps >= self.max_episode_steps
        info: dict = {}
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.image_height, width=self.image_width)
        self.renderer.update_scene(self.data, camera=self.camera_id)
        frame = self.renderer.render()
        if self.render_mode == "human":
            raise NotImplementedError("Human rendering is not implemented. Use render_mode='rgb_array'.")
        return frame

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


gym.register(
    id="LiftMobileRobot-v0",
    entry_point="vlamobile.lift_env:LiftEnv",
    # max_episode_steps=10000,
)
