from lerobot.envs.factory import make_env
import mujoco
import mujoco.viewer
import glfw
import numpy as np
import pygame
import os
import fire
import time
from PIL import Image

from vlamobile.controls import JoystickControl
from vlamobile.inference import LerobotInference, LerobotInferenceWithPlanner
from vlamobile.models.frozen_vla.inference import FrozenVLAInference
from vlamobile.recording import LeRobotRecorder, RecorderConfig, DEFAULT_RESOLUTION
from vlamobile.tasks import tasks


class Sim:
    def __init__(
        self,
        record: bool = True,
        disable_control: bool = False,
        compare_policy: bool = False,
        env_hf_path: str = "YinonDouchan/mobile_robot_lift_env@main",
        dataset_repo_id: str = "YinonDouchan/mobile_robot_lift_v1",
        local_data_root: str = "data",
        policy=None,
        task: str = "Pick up the small cube",
        policy_task: str = "Pick up the small cube",
        framerate: float = 30.0,
        policy_wait_frames: int = 30,
        render_camera_name: str = "sideview",
    ):
        self.record = record
        self.disable_control = disable_control
        self.compare_policy = compare_policy
        self.task = task
        self.policy_task = policy_task
        self.framerate = framerate
        self.policy_wait_frames = policy_wait_frames
        self.render_camera_name = render_camera_name

        self.recorder_start_recording = False
        self.recorder_save_episode = False
        self.recorder_discard_episode = False
        self.recorder_finalize = False
        self.reset_environment = False
        self.save_observation = False
        self.current_task = None
        self.joystick_override = False
        self.policy_frame_count = 0

        self.env = make_env(
            env_hf_path, n_envs=1, use_async_envs=False, trust_remote_code=True
        )["hub_env"][0]
        self.env.reset()

        self.env_unwrapped = self.env.envs[0].unwrapped
        self.env_unwrapped.max_episode_steps = float("inf")
        self.env_model = self.env_unwrapped.model
        self.env_data = self.env_unwrapped.data

        self.lerobot_recorder = None
        if record:
            recorder_config = RecorderConfig(
                repo_id=dataset_repo_id,
                root=local_data_root,
                fps=int(framerate),
                robot_type="custom_mobile_robot",
                cameras={
                    camera: DEFAULT_RESOLUTION
                    for camera in self.env_unwrapped.camera_names
                },
            )
            if os.path.exists(recorder_config.root):
                recorder_config.resume = True
            self.lerobot_recorder = LeRobotRecorder(recorder_config)

        self.policy = policy

        self.joystick_control = JoystickControl()

        obs, _ = self.env.reset()
        self.obs = obs
        self.state = obs["observation.state"]

    def key_callback(self, keycode: int) -> None:
        if keycode == glfw.KEY_SPACE:
            self.reset_environment = True
        if keycode == glfw.KEY_Y:
            self.recorder_start_recording = True
            print("Reset poses and started recording")
        elif keycode == glfw.KEY_S:
            self.recorder_save_episode = True
            print("Saved episode")
        elif keycode == glfw.KEY_E:
            self.recorder_discard_episode = True
            print("Discarded episode")
        elif keycode == glfw.KEY_F:
            self.recorder_finalize = True
            print("Finalizing recording")
        elif keycode == glfw.KEY_O:
            self.joystick_override = not self.joystick_override
            print(
                f"Joystick override {'enabled' if self.joystick_override else 'disabled'}"
            )
        elif keycode == glfw.KEY_U:
            self.save_observation = True

    def save_observation_to_disk(self, frames: dict[str, np.ndarray], state: np.ndarray) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("saved_observations", timestamp)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "state.npy"), state)

        for camera_name, image in frames.items():
            img = np.asarray(image)
            if img.dtype != np.uint8:
                if np.nanmax(img) <= 1.0:
                    img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(out_dir, f"{camera_name}.png"))

        print(f"Saved observation to {out_dir}")

    def handle_recorder_events(self, task_fn, task_fn_kwargs=None):
        if self.reset_environment:
            self.reset_environment = False
            self.env.reset()
            if task_fn_kwargs is None:
                task_fn_kwargs = {}
            self.current_task = task_fn(**task_fn_kwargs)
            print(f"Reset poses. Current task: {self.current_task}")
        if self.recorder_start_recording:
            self.recorder_start_recording = False
            self.lerobot_recorder.start_recording(self.current_task)
            print("Started recording")
        elif self.recorder_save_episode:
            self.recorder_save_episode = False
            self.lerobot_recorder.save_episode()
            print("Saved episode")
        elif self.recorder_discard_episode:
            self.recorder_discard_episode = False
            self.lerobot_recorder.discard_episode()
            print("Discarded episode")
        elif self.recorder_finalize:
            self.recorder_finalize = False
            self.lerobot_recorder.finalize()
            print("Finalizing recording")    

    def handle_policy_events(self):
        if self.reset_environment:
            self.reset_environment = False
            self.env.reset()
            self.policy.reset()
            self.policy_frame_count = 0
        if self.joystick_override:
            self.policy.reset()

  
    def get_next_action(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        if self.record:
            action = self.joystick_control(
                self.state, frames, task=self.policy_task
            )
            self.lerobot_recorder.add_frame(
                action=action,
                state=self.state[0],
                images=frames,
                environment_state=self.env_unwrapped.get_env_state(),
            )
            self.handle_recorder_events(task_fn=tasks[self.task])

            if self.compare_policy:
                test_action = self.policy.predict(
                    state=self.state, images=frames, task=self.task
                )
                print(test_action)

            action = action[None]
        else: # policy control with optional joystick override
            if self.joystick_override:
                action = self.joystick_control(
                    self.state, frames, task=self.policy_task
                )
            elif self.policy_frame_count < self.policy_wait_frames:
                action = np.zeros((1, 4), dtype=np.float64)
            else:
                action = self.policy.predict(
                    state=self.state, images=frames, task=self.policy_task
                )

            self.policy_frame_count += 1
            self.handle_policy_events()

        return action

    def run(self):
        with mujoco.viewer.launch_passive(
            self.env_model, self.env_data, key_callback=self.key_callback
        ) as viewer:
            viewer.cam.fixedcamid = self.env_unwrapped.camera_ids[
                self.render_camera_name
            ]
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

            while viewer.is_running():
                frames = {
                    k[len("observation.images.") :]: v[0]
                    for k, v in self.obs.items()
                    if k.startswith("observation.images.")
                }

                if self.save_observation:
                    self.save_observation = False
                    self.save_observation_to_disk(frames, self.obs["observation.state"])

                action = self.get_next_action(frames)

                with viewer.lock():
                    self.obs, _, _, _, _ = self.env.step(
                        action if not self.disable_control else None
                    )
                    self.state = self.obs["observation.state"]

                viewer.sync()
                if self.framerate > 0:
                    time.sleep(1.0 / self.framerate)

        if self.record:
            self.lerobot_recorder.stop_recording()
            self.lerobot_recorder.finalize()
            self.lerobot_recorder.close()

        self.env.close()


def _camera_keys(camera_keys: str | list[str] | None) -> list[str] | None:
    if camera_keys is None:
        return None
    if isinstance(camera_keys, str):
        return [key.strip() for key in camera_keys.split(",") if key.strip()]
    return list(camera_keys)


def make_policy(
    policy_path: str,
    dataset_repo_id: str,
    policy_type: str = "lerobot",
    use_policy_planner: bool = False,
    camera_keys: str | list[str] | None = None,
    n_action_steps: int | None = None,
):
    """Build a policy with ``predict`` and ``reset``.

    ``policy_path`` is a LeRobot Hub id or local checkpoint, or a FrozenVLA
    directory containing ``frozen_vla_config.pt`` and ``action_head.pt``.
    ``dataset_repo_id`` is used only for ``policy_type="lerobot"``.
    ``camera_keys`` is a list or a comma-separated string, used only for
    ``policy_type="frozen_vla"``.
    """
    if policy_type == "lerobot":
        if use_policy_planner:
            return LerobotInferenceWithPlanner(
                policy_path=policy_path,
                dataset_repo_id=dataset_repo_id,
                robot_type="custom_mobile_robot",
            )
        return LerobotInference(
            policy_path=policy_path,
            dataset_repo_id=dataset_repo_id,
            robot_type="custom_mobile_robot",
        )
    if policy_type == "frozen_vla":
        if use_policy_planner:
            raise ValueError("use_policy_planner is only supported for policy_type='lerobot'")
        return FrozenVLAInference(
            policy_path,
            camera_keys=_camera_keys(camera_keys),
            n_action_steps=n_action_steps,
        )
    raise ValueError(f"unknown policy_type {policy_type!r}; expected 'lerobot' or 'frozen_vla'")


def main(
    record: bool = True,
    disable_control: bool = False,
    compare_policy: bool = False,
    env_hf_path: str = "YinonDouchan/mobile_robot_lift_env@main",
    dataset_repo_id="YinonDouchan/mobile_robot_lift_v1",
    local_data_root="data",
    policy_path="YinonDouchan/smolvla_mobile_robot_lift_v1",
    policy_type: str = "lerobot",
    use_policy_planner: bool = False,
    camera_keys: str | list[str] | None = None,
    n_action_steps: int | None = None,
    task="Pick up the small cube",
    policy_task="Pick up the small cube",
    framerate: float = 30.0,
    policy_wait_frames: int = 30,
    render_camera_name: str = "sideview",
):
    policy = None
    if not record or compare_policy:
        policy = make_policy(
            policy_path,
            dataset_repo_id,
            policy_type=policy_type,
            use_policy_planner=use_policy_planner,
            camera_keys=camera_keys,
            n_action_steps=n_action_steps,
        )
    sim = Sim(
        record=record,
        disable_control=disable_control,
        compare_policy=compare_policy,
        env_hf_path=env_hf_path,
        dataset_repo_id=dataset_repo_id,
        local_data_root=local_data_root,
        policy=policy,
        task=task,
        policy_task=policy_task,
        framerate=framerate,
        policy_wait_frames=policy_wait_frames,
        render_camera_name=render_camera_name,
    )
    sim.run()


if __name__ == "__main__":
    fire.Fire(main)
