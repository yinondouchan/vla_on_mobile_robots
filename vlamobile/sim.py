from typing import Callable
import gymnasium
from lerobot.envs.factory import make_env
import mujoco
import mujoco.viewer
import glfw
import numpy as np
import pygame
import os
import fire
import time

from vlamobile.inference import LerobotInference
from vlamobile.lift_env import LiftEnv
from vlamobile.recording import LeRobotRecorder, RecorderConfig, DEFAULT_RESOLUTION
from vlamobile.tasks import tasks


def main(record: bool = True,
        disable_control: bool = False,
        compare_policy: bool = False,
        env_hf_path: str = "YinonDouchan/mobile_robot_lift_env@main",
        dataset_repo_id="YinonDouchan/mobile_robot_lift_v1",
        local_data_root="data",
        policy_path="YinonDouchan/smolvla_mobile_robot_lift_v1",
        task="Pick up the small cube",
        policy_task="Pick up the small cube",
        framerate: float = 30.0,
        policy_wait_frames: int = 30,
        render_camera_name: str = "sideview"
        ):

  input_active = False
  recorder_start_recording = False
  recorder_save_episode = False
  recorder_discard_episode = False
  recorder_finalize = False
  reset_environment = False
  current_task = None
  joystick_override = False
  policy_frame_count = 0

  def key_callback(keycode: int) -> None:
    nonlocal input_active, lerobot_recorder, recorder_start_recording, \
     recorder_save_episode, recorder_discard_episode, recorder_finalize, reset_environment, joystick_override
    if keycode == glfw.KEY_SPACE:
      reset_environment = True
    if keycode == glfw.KEY_Y:
      recorder_start_recording = True
      print("Reset poses and started recording")
    elif keycode == glfw.KEY_S:
      recorder_save_episode = True
      print("Saved episode")
    elif keycode == glfw.KEY_E:
      recorder_discard_episode = True
      print("Discarded episode")
    elif keycode == glfw.KEY_F:
      recorder_finalize = True
      print("Finalizing recording")
    elif keycode == glfw.KEY_O:
      joystick_override = not joystick_override
      print(f"Joystick override {'enabled' if joystick_override else 'disabled'}")

  def handle_recorder_events(env, lerobot_recorder, task_fn, task_fn_kwargs=None):
    nonlocal recorder_start_recording, recorder_save_episode, recorder_discard_episode, recorder_finalize, reset_environment, current_task
    if reset_environment:
        reset_environment = False
        env.reset()
        if task_fn_kwargs is None:
          task_fn_kwargs = {}
        current_task = task_fn(**task_fn_kwargs)
        print(f"Reset poses. Current task: {current_task}")
    if recorder_start_recording:
        recorder_start_recording = False
        lerobot_recorder.start_recording(current_task)
        print(f"Started recording")
    elif recorder_save_episode:
        recorder_save_episode = False
        lerobot_recorder.save_episode()
        print("Saved episode")
    elif recorder_discard_episode:
        recorder_discard_episode = False
        lerobot_recorder.discard_episode()
        print("Discarded episode")
    elif recorder_finalize:
        recorder_finalize = False
        lerobot_recorder.finalize()
        print("Finalizing recording")


  def handle_policy_events(env, policy):
    nonlocal reset_environment, joystick_override, policy_frame_count
    if reset_environment:
        reset_environment = False
        env.reset()
        policy.reset()
        policy_frame_count = 0
    if joystick_override:
        policy.reset()


  env = make_env(env_hf_path, n_envs=1, use_async_envs=False, trust_remote_code=True)['hub_env'][0]
  env.reset()

  env_unwrapped = env.envs[0].unwrapped
  env_unwrapped.max_episode_steps = float('inf')
  env_model = env_unwrapped.model
  env_data = env_unwrapped.data

  if record:
    recorder_config = RecorderConfig(
        repo_id=dataset_repo_id,
        root=local_data_root,
        fps=int(framerate),
        robot_type="custom_mobile_robot",
        cameras={camera: DEFAULT_RESOLUTION for camera in env_unwrapped.camera_names},
    )

    if os.path.exists(recorder_config.root):
      recorder_config.resume = True

    lerobot_recorder = LeRobotRecorder(recorder_config)  # You may want to pass recorder_config as needed
  if not record or compare_policy:
    policy_control = PolicyControl(policy_path, dataset_repo_id)

  pygame.init()
  pygame.joystick.init()

  if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
  else:
    joystick = None
  
  joystick_control = JoystickControl(joystick)

  obs, _ = env.reset()
  state = obs['observation.state']

  with mujoco.viewer.launch_passive(env_model, env_data, key_callback=key_callback) as viewer:
      viewer.cam.fixedcamid = env_unwrapped.camera_ids[render_camera_name]
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

      while viewer.is_running():
          frames  = {k[len("observation.images."):]: v[0] for k, v in obs.items() if k.startswith("observation.images.")}

          if record:
            action = joystick_control(state, frames, task=policy_task)
            lerobot_recorder.add_frame(action=action, state=state[0], images=frames,
             environment_state=env_unwrapped.get_env_state())
            handle_recorder_events(env, lerobot_recorder=lerobot_recorder, task_fn=tasks[task])

            if compare_policy:
              test_action = policy_control(state=state, frames=frames, task=task)
              print(test_action)

            action = action[None]
          else:
            if joystick_override:
              action = joystick_control(state, frames)
            elif policy_frame_count < policy_wait_frames:
              action = np.zeros((1, 4), dtype=np.float64)
            else:
              action = policy_control(state=state, frames=frames, task=policy_task)

            policy_frame_count += 1
            handle_policy_events(env, policy_control)

          input_active = False

          # Step the simulation forward
          with viewer.lock():
            obs, _, _, _, info = env.step(action if not disable_control else None)
            state = obs['observation.state']
              
          # Render the frame and sync with the viewer
          viewer.sync()
          time.sleep(1.0 / framerate)  # Sleep to target 30 FPS (approximately)
     
  if record:
    lerobot_recorder.stop_recording()
    lerobot_recorder.finalize()
    lerobot_recorder.close()

  env.close()


class JoystickControl:
    def __init__(self, joystick: pygame.joystick.Joystick):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() > 0:
          joystick = pygame.joystick.Joystick(0)
          joystick.init()
        else:
          joystick = None

        self.forward = 0.0
        self.turn = 0.0
        self.lift = 0.0
        self.grip = 0.0

    def __call__(self, state, frames, task: str):
        for event in pygame.event.get():
          if event.type == pygame.JOYAXISMOTION:
              if event.axis == 4:
                self.forward = -event.value
              elif event.axis == 3:
                self.turn = -event.value
              elif event.axis == 1:
                self.lift = event.value
              elif event.axis == 2:
                # Left trigger: typically -1 (released) .. +1 (pressed) -> [0, 1]
                self.grip = (event.value + 1.0) * 0.5
                # print(f"Axis {event.axis} moved to {event.value:.2f}")
                
            # elif event.type == pygame.JOYBUTTONUP:
            #     print(f"Button {event.button} released")
                
            # elif event.type == pygame.JOYHATMOTION:
            #     print(f"Hat/D-pad {event.hat} moved to {event.value}")

        return np.array([
            self.forward,
            self.turn,
            self.grip,
            self.lift
        ], dtype=np.float64)


    def reset(self):
      self.forward = 0.0
      self.turn = 0.0
      self.lift = 0.0
      self.grip = 0.0


class PolicyControl:
    def __init__(self, policy_path, dataset_repo_id):
        self.policy = LerobotInference(
          policy_path=policy_path,
          dataset_repo_id=dataset_repo_id,
          robot_type="custom_mobile_robot",
        )
        self.policy.reset()

    def __call__(self, state, frames, task: str):
        action = self.policy.predict(state=state, images=frames, task=task)
        return action

    def reset(self):
        self.policy.reset()


if __name__ == "__main__":
  fire.Fire(main)