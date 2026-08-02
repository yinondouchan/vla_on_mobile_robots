from contextlib import nullcontext
from lerobot.envs.factory import make_env
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import fire
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import random
import json
import os
from tqdm import tqdm
from vlamobile.lift_env import is_red_cube_lifted, is_red_cube_on_red_platform, is_red_cube_on_green_platform, is_red_cube_on_blue_platform
from  vlamobile.tasks import pick_and_place_single_cube_multi_platform
from vlamobile.lift_env import DEFAULT_MODEL_PATH, LiftEnv, DEFAULT_STATE_NAMES
from vlamobile.inference import LerobotInference
from vlamobile.recording import DEFAULT_ACTION_NAMES


def plot_alignment(policy_actions, gt_actions, gt_states):
    # Convert evaluation history to numpy arrays
    policy_actions = np.array(policy_actions)

    action_dim = policy_actions.shape[1] if len(policy_actions.shape) > 1 else 1
    state_dim = gt_states.shape[1] if len(gt_states.shape) > 1 else 1

    # Plot policy actions vs ground truth actions for each action dimension
    fig, axs = plt.subplots(action_dim, 1, figsize=(10, 3 * action_dim), sharex=True)

    if action_dim == 1:
        axs = [axs]

    for i in range(action_dim):
        axs[i].plot(policy_actions[:, i], label=f'Policy Action')
        axs[i].plot(gt_actions[:, i], label=f'GT Action')
        axs[i].set_title(f'{DEFAULT_ACTION_NAMES[i]}')
        axs[i].set_xlabel(f'Step')
        axs[i].legend()
        axs[i].grid(True)

    plt.suptitle('Policy Actions vs Ground Truth Actions')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Plot policy states vs ground truth states for each state dimension
    fig, axs = plt.subplots(state_dim, 1, figsize=(10, 3 * state_dim), sharex=True)

    if state_dim == 1:
        axs = [axs]

    for i in range(state_dim):
        axs[i].plot(gt_states[:, i])
        axs[i].set_title(f'{DEFAULT_STATE_NAMES[i]}')
        axs[i].set_xlabel(f'Step')
        axs[i].legend()
        axs[i].grid(True)

    plt.suptitle('Ground Truth States')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


class Evaluation:
    def compare_gt_to_rollout(self, dataset_repo_id, env_hf_path, policy_path, episode_idx=0):
        # Load your custom dataset containing the expert demonstrations
        dataset = LeRobotDataset(dataset_repo_id, revision="main")

        # Extract ground-truth actions and states for a baseline comparison episode
        episode_idx = 0
        dataset_df = dataset.hf_dataset.to_pandas()
        episode_df = dataset_df[dataset_df["episode_index"] == episode_idx]
        initial_state = {
            "qpos": episode_df["environment_state_qpos"].iloc[0],
            "qvel": episode_df["environment_state_qvel"].iloc[0]
        }

        gt_actions = np.vstack(episode_df["action"])
        gt_states = np.vstack(episode_df["observation.state"])

        # env = make_env(env_hf_path, trust_remote_code=True)['hub_env'][0]
        env = LiftEnv(model_path=DEFAULT_MODEL_PATH)
        policy = LerobotInference(
            policy_path=policy_path,
            dataset_repo_id=dataset_repo_id,
            robot_type="custom_mobile_robot"
        )
        policy.reset()

        # 1. Reset simulator to the exact initialization of the ground truth episode
        obs, _ = env.reset(options={"initial_state": initial_state}) 

        policy_actions = []
        policy_states = []

        # 2. Rollout the policy over the same horizon as the ground truth episode
        horizon = len(gt_actions)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.fixedcamid = env.camera_id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

            for step in tqdm(range(horizon)):
                frame = env.render()
                # Prepare inputs for policy
                state = obs["observation.state"] if isinstance(obs, dict) else obs
                action = policy.predict(state=state, images={"sideview": frame}, task=dataset.meta.tasks[dataset.meta.tasks == episode_df['task_index'].iloc[0]].index[0])
                action[0] = np.clip(action[0], -1.0, 1.0)
                action[1] = np.clip(action[1], -1.0, 1.0)
                action[2] = np.clip(action[2], 0.0, 1.0)
                action[3] = np.clip(action[3], 0.0, 1.0)

                policy_actions.append(action)
                policy_states.append(state)

                # Step the environment
                obs, reward, done, truncated, info = env.step(action)

                # Render the frame and sync with the viewer
                viewer.sync()

        plot_alignment(policy_actions, gt_actions, policy_states, gt_states)


    def eval_policy_output(self, dataset_repo_id, policy_path, episode_idx):
        # Load your custom dataset containing the expert demonstrations
        dataset = LeRobotDataset(dataset_repo_id, revision="main")

        # Extract ground-truth actions and states for a baseline comparison episode
        dataset_df = dataset.hf_dataset.to_pandas()
        episode_df = dataset_df[dataset_df["episode_index"] == episode_idx]


        gt_actions = np.vstack(episode_df["action"])
        gt_states = np.vstack(episode_df["observation.state"])

        policy = LerobotInference(
            policy_path=policy_path,
            dataset_repo_id=dataset_repo_id,
            robot_type="custom_mobile_robot"
        )
        policy.reset()

        policy_actions = []

        horizon = len(gt_actions)
        for step in tqdm(range(horizon)):
            frame_idx = episode_df["frame_index"].iloc[step]
            frame = dataset[frame_idx]["observation.images.sideview"]
            frame = (frame.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
            state = dataset[frame_idx]["observation.state"]
            action = policy.predict(state=state, images={"sideview": frame}, task=dataset.meta.tasks.index[episode_df['task_index'].iloc[0]])
            policy_actions.append(action[0])

        policy_actions = np.array(policy_actions)
        plot_alignment(policy_actions, gt_actions, gt_states)


    def eval_pick_and_place_single_multi(self,policy_path, env_hf_path, dataset_repo_id, max_steps=2000, num_runs=100, seed=42, policy_wait_frames: int = 30, visualize=False, output_path=None):
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

        task_goals = {
            "is_cube_lifted": is_red_cube_lifted(env_model, env_data),
            "is_cube_on_red_platform": is_red_cube_on_red_platform(env_model, env_data),
            "is_cube_on_green_platform": is_red_cube_on_green_platform(env_model, env_data),
            "is_cube_on_blue_platform": is_red_cube_on_blue_platform(env_model, env_data)
        }

        random.seed(seed)
        tasks = [pick_and_place_single_cube_multi_platform() for _ in range(num_runs)]

        task_results = []

        for run in tqdm(range(num_runs)):
            task = tasks[run]

            with mujoco.viewer.launch_passive(env_model, env_data) if visualize else nullcontext() as viewer:
                if visualize:
                    viewer.cam.fixedcamid = env_unwrapped.camera_ids["robotfrontview_high"] if hasattr(env_unwrapped, "camera_ids") and "robotfrontview_high" in env_unwrapped.camera_ids else 0
                    # viewer.add_overlay(mujoco.mjtGridPos.mjGRID_TOPLEFT, "Task:", task)
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

                run_task_goals = []

                for step in tqdm(range(max_steps)):
                    frames  = {k[len("observation.images."):]: v[0] for k, v in obs.items() if k.startswith("observation.images.")}
                    if step < policy_wait_frames:
                        action = np.zeros((1, 4), dtype=np.float64)
                    else:
                        action = policy.predict(state, frames, task)

                    # Step the simulation forward
                    with viewer.lock() if visualize else nullcontext():
                        obs, _, _, _, info = env.step(action)
                        state = obs['observation.state']
                    if visualize:
                        viewer.sync()
                    # time.sleep(1.0 / 30.0)  # Sleep to target 30 FPS (or set to your framerate)

                    task_goals = {
                        "is_cube_lifted": is_red_cube_lifted(env_model, env_data),
                        "is_cube_on_red_platform": is_red_cube_on_red_platform(env_model, env_data),
                        "is_cube_on_green_platform": is_red_cube_on_green_platform(env_model, env_data),
                        "is_cube_on_blue_platform": is_red_cube_on_blue_platform(env_model, env_data)
                    }

                    run_task_goals.append(task_goals)

                    if not task_goals['is_cube_lifted'] and (task_goals['is_cube_on_red_platform'] or task_goals['is_cube_on_green_platform'] or task_goals['is_cube_on_blue_platform']):
                        break

                task_results.append({
                    "task": task,
                    "success": ("red" in task and task_goals['is_cube_on_red_platform'] and not task_goals['is_cube_on_green_platform'] and not task_goals['is_cube_on_blue_platform'])
                                or ("green" in task and task_goals['is_cube_on_green_platform'] and not task_goals['is_cube_on_red_platform'] and not task_goals['is_cube_on_blue_platform'])
                                or ("blue" in task and task_goals['is_cube_on_blue_platform'] and not task_goals['is_cube_on_red_platform'] and not task_goals['is_cube_on_green_platform']),
                    "final_platform": "red" if task_goals['is_cube_on_red_platform']
                                        else "green" if task_goals['is_cube_on_green_platform']
                                        else "blue" if task_goals['is_cube_on_blue_platform'] else None,
                    "cube_was_lifted": sum([goal['is_cube_lifted'] for goal in run_task_goals]) > 15,
                    "num_steps": step + 1
                })

            env.reset()
            policy.reset()

        success_rate = sum([result['success'] for result in task_results]) / len(task_results)
        lift_success_rate = sum([result['cube_was_lifted'] for result in task_results]) / len(task_results)
        print(f"Success rate: {success_rate}")
        print(f"Lift success rate: {lift_success_rate}")
        print("Average number of steps (successful runs only): ", np.mean([result['num_steps'] for result in task_results if result['success']]))

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                run_params = {
                    "num_runs": len(task_results),
                    "success_rate": success_rate,
                    "lift_success_rate": lift_success_rate,
                    "average_steps_successful": float(np.mean([result['num_steps'] for result in task_results if result['success']])) if any(result['success'] for result in task_results) else None,
                    "policy_path": policy_path,
                    "env_hf_path": env_hf_path,
                    "dataset_repo_id": dataset_repo_id,
                    "max_steps": max_steps,
                    "policy_wait_frames": policy_wait_frames,
                    "seed": seed,
                    "task_results": task_results
                }
                
                json.dump(run_params, f, indent=4)

if __name__ == "__main__":
    fire.Fire(Evaluation)