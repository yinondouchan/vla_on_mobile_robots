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


INPUT_DECAY_FORWARD = 0.95
INPUT_DECAY_TURN = 0.97
INPUT_DECAY_LIFT = 0.95

JOYSTICK_LIFT_GAIN = 0.002
MAX_VELOCITY = 50.
TURN_GAIN = 0.5

tasks = [
  "Pick up the small cube",
  "Pick up the medium cube",
  "Pick up the large cube",
  "Lift the small cube three inches above the surface",
  "Touch the red platform with the claw",
  "Move the claw directly above the green platform",
  "Open the claw while holding nothing over the blue platform",
  "Touch the top face of the large cube",
  "Nudge the medium cube to the left",
  "Hover the claw exactly two inches above the small cube",
  "Place the small cube on the red platform",         
  "Place the medium cube on the blue platform",
  "Place the large cube on the green platform",
  "Move the small cube from its current position to the green platform",
  "Pick up the largest object and hold it over the red platform",
  "Find the smallest cube and place it on the blue platform",
  "Move the medium cube onto the platform that is not red or blue",
  "Place the large cube on the red platform, then release it",
  "Clear the blue platform by moving whatever is on it to the floor",
  "Move the small cube to the same platform where the large cube is sitting",
  "Stack the small cube on top of the medium cube.",
  "Stack the medium cube on top of the large cube.",
  "Build a three-cube tower with the largest at the bottom and smallest at the top.",
  "Place the large cube on top of the small cube (testing stability handling).",
  "Place the medium cube directly to the right of the red platform.",
  "Put the small cube on the green platform, next to the medium cube.",
  "Line up all three cubes in a straight row next to the blue platform.",
  "Place the small cube between the medium cube and the large cube.",
  "Move the large cube so it is touching the side of the red platform.",
  "Stack two cubes of any size onto the blue platform.",
  # "If the small cube is on the red platform, move it to the blue platform.",
  # "Move the medium cube to the red platform, then move the large cube to the green platform.",
  # "Stack the small cube on the medium cube, then move the e ntire stack to the blue platform.",
  # "Swap the positions of the small cube and the large cube.",
  # "Unstack the tower and place each cube on a different colored platform.",
  # "Put the small cube on the green platform only if the blue platform is empty.",
  # "Move all cubes off of the platforms and onto the floor.",
  # "Take the topmost cube off the stack and place it on the red platform.",
  # "Move the large cube to the blue platform without knocking over the small cube.",
  # "Place the medium cube on the platform that matches the color of the smallest cube's current platform.",
  # "Organize the environment so that every platform has exactly one cube on it.",
  # "Sort the cubes onto the platforms by size: smallest on red, medium on green, largest on blue.",
  # "Clear the red platform completely.",
  # "Reset the environment so that no cubes are touching each other or any platforms.",
  # "Hide the small cube by placing the large cube directly over or in front of it from the camera's view.",
  # "Create a stable two-cube stack on the green platform, leaving the large cube on the floor.",
  # "Move all objects as far away from the blue platform as possible.",
  # "Build the tallest tower possible on the red platform.",
  # "Reverse the order of the current cube stack.",
  # "Put all cubes on the same platform, ordered from smallest at the bottom to largest on top.",
]
    

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

  def key_callback(keycode: int) -> None:
    nonlocal input_active, model, data, lerobot_recorder, recorder_start_recording, recorder_save_episode, recorder_discard_episode, recorder_finalize
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


  def handle_recorder_events(model, data, tasks, lerobot_recorder):
    nonlocal recorder_start_recording, recorder_save_episode, recorder_discard_episode, recorder_finalize
    
    if recorder_start_recording:
        recorder_start_recording = False
        reset_poses(model, data)
        current_task = tasks[np.random.randint(len(tasks))]
        lerobot_recorder.start_recording(current_task)
        print(f"Reset poses and started recording: {current_task}")
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
    nonlocal forward, turn, grip, lift, input_active
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
 
    
  def update_actuators(data):
    nonlocal forward, turn, grip, lift, input_active

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

    input_active = False

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
              tasks=tasks,
              lerobot_recorder=lerobot_recorder,
            )
          else:
            action = policy.predict(state=state, images={"sideview": frame})
            forward = action[0]
            turn = action[1]
            grip = action[2]
            lift = action[3]

          update_actuators(data)

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