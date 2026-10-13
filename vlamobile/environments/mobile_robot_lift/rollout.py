from contextlib import nullcontext
from lerobot.envs.factory import make_env
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import fire
import mujoco
import mujoco.viewer
import random
import os
import pandas as pd
import json
from tqdm import tqdm
from vlamobile.environments.mobile_robot_lift.lift_env import (
    is_red_cube_lifted,
    is_red_cube_on_blue_platform,
    is_red_cube_on_green_platform,
    is_red_cube_on_red_platform,
)
from vlamobile.environments.mobile_robot_lift.tasks import pick_and_place_single_cube_multi_platform
from vlamobile.inference import LerobotInference
from vlamobile.recording import DEFAULT_RESOLUTION, LeRobotRecorder, RecorderConfig


def _init_recorder(dataset_repo_id, local_data_root, framerate, env_unwrapped):

    recorder_config = RecorderConfig(
        repo_id=dataset_repo_id,
        root=local_data_root,
        fps=int(framerate),
        robot_type="custom_mobile_robot",
        cameras={camera: DEFAULT_RESOLUTION for camera in env_unwrapped.camera_names},
    )

    if os.path.exists(recorder_config.root):
      recorder_config.resume = True

    return LeRobotRecorder(recorder_config)  # You may want to pass recorder_config as needed


def _add_success_to_dataset(output_dataset_local_root):
    df = pd.read_parquet("data_pick_and_place_single_multi_egocentric_rollout/data/chunk-000/file-000.parquet")
    df_success = pd.read_parquet("data_pick_and_place_single_multi_egocentric_rollout/task_success.parquet")

    df_merged = pd.merge(df, df_success, on="episode_index", how="outer")

    df_merged.to_parquet("data_pick_and_place_single_multi_egocentric_rollout/data/chunk-000/file-000.parquet", index=False)

    with open("data_pick_and_place_single_multi_egocentric_rollout/meta/info.json", "w") as f:
        json.dump(task_meta, f)


def generate_rollouts(policy_path, env_hf_path, dataset_repo_id, output_dataset_local_root, output_dataset_repo_id=None, max_steps=2000, num_runs=100, seed=None, policy_wait_frames: int = 30, visualize=False):
    env = make_env(env_hf_path, n_envs=1, use_async_envs=False, trust_remote_code=True)['hub_env'][0]
    env.reset(seed=seed)

    env_unwrapped = env.envs[0].unwrapped
    env_unwrapped.max_episode_steps = float('inf')
    env_model = env_unwrapped.model
    env_data = env_unwrapped.data
    obs, _ = env.reset(seed=seed)
    state = obs['observation.state']

    policy = LerobotInference(
        policy_path=policy_path,
        dataset_repo_id=dataset_repo_id,
        robot_type="custom_mobile_robot",
    )
    policy.reset()

    recorder = _init_recorder(output_dataset_repo_id, output_dataset_local_root, 30, env_unwrapped)

    random.seed(seed)
    tasks = [pick_and_place_single_cube_multi_platform() for _ in range(num_runs)]

    task_successes = []

    for run in tqdm(range(num_runs)):
        task = tasks[run]

        with mujoco.viewer.launch_passive(env_model, env_data) if visualize else nullcontext() as viewer:
            if visualize:
                viewer.cam.fixedcamid = env_unwrapped.camera_ids["robotfrontview_high"] if hasattr(env_unwrapped, "camera_ids") and "robotfrontview_high" in env_unwrapped.camera_ids else 0
                # viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPLEFT, "Task:", task)
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

            for step in tqdm(range(max_steps)):
                frames  = {k[len("observation.images."):]: v[0] for k, v in obs.items() if k.startswith("observation.images.")}

                if step == policy_wait_frames:
                    recorder.start_recording(task)

                if step < policy_wait_frames:
                    action = np.zeros((1, 4), dtype=np.float64)
                else:
                    action = policy.predict(state, frames, task)
                    recorder.add_frame(action=action[0], state=state[0], images=frames, environment_state=env_unwrapped.get_env_state())

                # Step the simulation forward
                with viewer.lock() if visualize else nullcontext():
                    obs, _, _, _, info = env.step(action)
                    state = obs['observation.state']
                if visualize:
                    viewer.sync()

                task_goals = {
                    "is_cube_lifted": is_red_cube_lifted(env_model, env_data),
                    "is_cube_on_red_platform": is_red_cube_on_red_platform(env_model, env_data),
                    "is_cube_on_green_platform": is_red_cube_on_green_platform(env_model, env_data),
                    "is_cube_on_blue_platform": is_red_cube_on_blue_platform(env_model, env_data)
                }

                if not task_goals['is_cube_lifted'] and (task_goals['is_cube_on_red_platform'] or task_goals['is_cube_on_green_platform'] or task_goals['is_cube_on_blue_platform']):
                    break

            success = (("red" in task and task_goals['is_cube_on_red_platform'] and not task_goals['is_cube_on_green_platform'] and not task_goals['is_cube_on_blue_platform'])
                       or ("green" in task and task_goals['is_cube_on_green_platform'] and not task_goals['is_cube_on_red_platform'] and not task_goals['is_cube_on_blue_platform'])
                       or ("blue" in task and task_goals['is_cube_on_blue_platform'] and not task_goals['is_cube_on_red_platform'] and not task_goals['is_cube_on_green_platform']))   

            task_successes.append({'episode_index': run, 'success': success})         

        env.reset()
        policy.reset()
        recorder.save_episode()

    recorder.finalize()
    recorder.close()

    pd.DataFrame(task_successes).to_parquet(os.path.join(output_dataset_local_root, "task_success.parquet"), index=False)


if __name__ == "__main__":
    fire.Fire(generate_rollouts)