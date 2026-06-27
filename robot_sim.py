import mujoco
import mujoco.viewer
import time
import glfw
import numpy as np
import pygame
import os
import fire

from recording import RecorderConfig, LeRobotRecorder
from inference import LerobotInference
from tasks import generate_pick_and_place_task
from env import LiftEnv


def main(record: bool = True,
        disable_control: bool = False,
        compare_policy: bool = False,
        dataset_repo_id="YinonDouchan/mobile_robot_lift_v1",
        local_data_root="data",
        policy_path="YinonDouchan/smolvla_mobile_robot_lift_v1",
        task="Pick up the small cube",
        ):

  forward = 0.0
  turn = 0.0
  lift = 0.0
  grip = False
  input_active = False
  recorder_start_recording = False
  recorder_save_episode = False
  recorder_discard_episode = False
  recorder_finalize = False
  reset_environment = False
  current_task = None

  def key_callback(keycode: int) -> None:
    nonlocal input_active, lerobot_recorder, recorder_start_recording, \
     recorder_save_episode, recorder_discard_episode, recorder_finalize, reset_environment
    if keycode == glfw.KEY_SPACE:
      reset_environment = True
    if keycode == glfw.KEY_R:
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


  def handle_recorder_events(env, lerobot_recorder):
    nonlocal recorder_start_recording, recorder_save_episode, recorder_discard_episode, recorder_finalize, reset_environment, current_task
    if reset_environment:
        reset_environment = False
        env.reset()
        # current_task = generate_pick_and_place_task()
        current_task = "Lift gripper up and then lift it down"
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
    nonlocal recorder_start_recording, recorder_save_episode, recorder_discard_episode, recorder_finalize, reset_environment, current_task
    if reset_environment:
        reset_environment = False
        env.reset()
        policy.reset()


  def read_joystick(joystick):
    nonlocal forward, turn, grip, lift

    if joystick is None:
      return

    # axes = [joystick.get_axis(idx) for idx in range(6)]
    # print(axes)
    for event in pygame.event.get():
        if event.type == pygame.JOYAXISMOTION:
          if event.axis == 4:
            forward = -event.value
          elif event.axis == 3:
            turn = -event.value
          elif event.axis == 1:
            lift = event.value
            # print(f"Axis {event.axis} moved to {event.value:.2f}")
        elif event.type == pygame.JOYBUTTONDOWN:
          if event.button == 4:
            grip = not grip
            
        # elif event.type == pygame.JOYBUTTONUP:
        #     print(f"Button {event.button} released")
            
        # elif event.type == pygame.JOYHATMOTION:
        #     print(f"Hat/D-pad {event.hat} moved to {event.value}")

  env = LiftEnv(model_path='simple_scene.xml')
  env.reset()

  if record:
    recorder_config = RecorderConfig(
        repo_id=dataset_repo_id,
        root=local_data_root,
        fps=int(1.0 / env.model.opt.timestep),
        robot_type="custom_mobile_robot",
    )

    if os.path.exists(recorder_config.root):
      recorder_config.resume = True

    lerobot_recorder = LeRobotRecorder(recorder_config)  # You may want to pass recorder_config as needed

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() > 0:
      joystick = pygame.joystick.Joystick(0)
      joystick.init()
    else:
      joystick = None
      # raise IOError("No joystick found")
  if not record or compare_policy:
    # Initialize the policy
    policy = LerobotInference(
        policy_path=policy_path,
        dataset_repo_id=dataset_repo_id,
        robot_type="custom_mobile_robot",
    )
    policy.reset()

  obs, _ = env.reset()
  state = obs['observation.state']


  # 2. Open the visualizer and run the simulation
  with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
      viewer.cam.fixedcamid = env.camera_id
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

      while viewer.is_running():
          frame = env.render()

          if record:
            read_joystick(joystick)
            action = np.array([
                forward,
                turn,
                grip,
                lift
            ], dtype=np.float64)
            lerobot_recorder.add_frame(action=action, state=state, images={"sideview": frame})
            handle_recorder_events(env, lerobot_recorder=lerobot_recorder)

            if compare_policy:
              test_action = policy.predict(state=state, images={"sideview": frame}, task=task)
              print(test_action)
          else:
            action = policy.predict(state=state, images={"sideview": frame}, task=task)
            action[0] = np.clip(action[0], -1.0, 1.0)
            action[1] = np.clip(action[1], -1.0, 1.0)
            action[2] = np.clip(action[2], 0.0, 1.0)
            action[3] = np.clip(action[3], 0.0, 1.0)
            handle_policy_events(env, policy)

          input_active = False

          # Step the simulation forward
          with viewer.lock():
            obs, _, _, _, _ = env.step(action if not disable_control else None)
            state = obs['observation.state']
              
          # Render the frame and sync with the viewer
          viewer.sync()
     
  if record:
    lerobot_recorder.stop_recording()
    lerobot_recorder.finalize()
    lerobot_recorder.close()

  env.close()



if __name__ == "__main__":
  fire.Fire(main)