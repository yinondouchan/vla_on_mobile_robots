import mujoco
import mujoco.viewer
import time
import glfw
import numpy as np
import pygame
import os
import fire

from recording import RecorderConfig, LeRobotRecorder, mobile_robot_state
from inference import LerobotInference
from tasks import TASKS, generate_pick_and_place_task


INPUT_DECAY_FORWARD = 0.95
INPUT_DECAY_TURN = 0.97
INPUT_DECAY_LIFT = 0.95

JOYSTICK_LIFT_GAIN = 0.002
MAX_VELOCITY = 50.
TURN_GAIN = 0.5
    

def reset_joint_pose(model, data, joint_name, x_range=(-1, 1), y_range=(-1, 1), q_range=(-np.pi, np.pi)):
  x = np.random.uniform(x_range[0], x_range[1])
  y = np.random.uniform(y_range[0], y_range[1])
  q = np.random.uniform(q_range[0], q_range[1])
  joint_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)]
  joint_qpos = data.qpos[joint_adr:joint_adr + 7]
  joint_qpos[:2] = [x, y]
  joint_qpos[3:7] = [np.cos(q/2), 0, 0, np.sin(q/2)]


def reset_poses(model, data):
  reset_joint_pose(model, data, "base_joint", y_range=(-2, -1))
  reset_joint_pose(model, data, "cube_joint0", y_range=(-1, 0.8))
  reset_joint_pose(model, data, "cube_joint1", y_range=(-1, 0.8))
  reset_joint_pose(model, data, "cube_joint2", y_range=(-1, 0.8))


def update_actuators(data, viewer, action: np.ndarray):
  forward, turn, grip, lift = action

  with viewer.lock():
      data.actuator('right_motor').ctrl = forward
      data.actuator('left_motor').ctrl = forward
      data.actuator('right_motor').ctrl += turn
      data.actuator('left_motor').ctrl -= turn
      data.actuator('gripper_lift').ctrl = lift

      if not grip:
          data.actuator('gripper_right_pos').ctrl = 0.07
          data.actuator('gripper_left_pos').ctrl = 0.07
      else:
          data.actuator('gripper_right_pos').ctrl = 0.0
          data.actuator('gripper_left_pos').ctrl = 0.0


def main(record: bool = True,
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
    nonlocal input_active, model, data, lerobot_recorder, recorder_start_recording, \
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


  def handle_recorder_events(model, data, lerobot_recorder):
    nonlocal recorder_start_recording, recorder_save_episode, recorder_discard_episode, recorder_finalize, reset_environment, current_task
    if reset_environment:
        reset_environment = False
        reset_poses(model, data)
        current_task = generate_pick_and_place_task()
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


  def read_joystick(joystick):
    nonlocal forward, turn, grip, lift
    # axes = [joystick.get_axis(idx) for idx in range(6)]
    # print(axes)
    for event in pygame.event.get():
        if event.type == pygame.JOYAXISMOTION:
          if event.axis == 4:
            forward = -event.value * MAX_VELOCITY
          elif event.axis == 3:
            turn = -event.value * MAX_VELOCITY * TURN_GAIN
          elif event.axis == 1:
            lift = event.value * 0.15
            # print(f"Axis {event.axis} moved to {event.value:.2f}")

          lift = max(0.0, min(0.15, lift))
            
        elif event.type == pygame.JOYBUTTONDOWN:
          if event.button == 4:
            grip = not grip
            
        # elif event.type == pygame.JOYBUTTONUP:
        #     print(f"Button {event.button} released")
            
        # elif event.type == pygame.JOYHATMOTION:
        #     print(f"Hat/D-pad {event.hat} moved to {event.value}")


  # 1. Load the model from your XML file
  model = mujoco.MjModel.from_xml_path('simple_scene.xml')
  data = mujoco.MjData(model)
  
  renderer = mujoco.Renderer(model, height=480, width=640)

  if record:
    recorder_config = RecorderConfig(
        repo_id=dataset_repo_id,
        root=local_data_root,
        fps=int(1.0 / model.opt.timestep),
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
      raise IOError("No joystick found")
  else:
    # Initialize the policy
    policy = LerobotInference(
        policy_path=policy_path,
        dataset_repo_id=dataset_repo_id,
        task=task,
        robot_type="custom_mobile_robot",
    )

  reset_poses(model, data)


  # 2. Open the visualizer and run the simulation
  with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
      camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "sideview")
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
      viewer.cam.fixedcamid = camera_id

      while viewer.is_running():
          renderer.update_scene(data, camera=camera_id)
          frame = renderer.render()
          state = mobile_robot_state(data)
          if record:
            action = np.array([
                forward,
                turn,
                grip,
                lift
            ], dtype=np.float64)
            lerobot_recorder.add_frame(action=action, state=state, images={"sideview": frame})
            read_joystick(joystick)

            handle_recorder_events(
              model=model,
              data=data,
              lerobot_recorder=lerobot_recorder,
            )
          else:
            action = policy.predict(state=state, images={"sideview": frame})
            forward = action[0]
            turn = action[1]
            grip = action[2]
            lift = action[3]

          update_actuators(data, viewer,action)
          input_active = False

          # Step the simulation forward
          with viewer.lock():
            mujoco.mj_step(model, data)
              
          # Render the frame and sync with the viewer
          viewer.sync()
          time.sleep(model.opt.timestep)
     
  if record:
    lerobot_recorder.stop_recording()
    lerobot_recorder.finalize()
    lerobot_recorder.close()



if __name__ == "__main__":
  fire.Fire(main)