from lerobot.envs.factory import make_env
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import fire
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
from tqdm import tqdm

from env import LiftEnv
from inference import LerobotInference


def plot_alignment(policy_actions, gt_actions, policy_states, gt_states):
    # Convert evaluation history to numpy arrays
    policy_actions = np.array(policy_actions)
    policy_states = np.array(policy_states)

    action_dim = policy_actions.shape[1] if len(policy_actions.shape) > 1 else 1
    state_dim = policy_states.shape[1] if len(policy_states.shape) > 1 else 1

    # Plot policy actions vs ground truth actions for each action dimension
    fig, axs = plt.subplots(action_dim, 1, figsize=(10, 3 * action_dim), sharex=True)

    if action_dim == 1:
        axs = [axs]

    for i in range(action_dim):
        axs[i].plot(policy_actions[:, i], label=f'Action {i}')
        axs[i].plot(gt_actions[:, i], label=f'GT Action {i}')
        axs[i].set_xlabel(f'Action {i}')
        axs[i].legend()
        axs[i].grid(True)

    plt.suptitle('Policy Actions vs Ground Truth Actions')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Plot policy states vs ground truth states for each state dimension
    fig, axs = plt.subplots(state_dim, 1, figsize=(10, 3 * state_dim), sharex=True)

    if state_dim == 1:
        axs = [axs]

    for i in range(state_dim):
        axs[i].plot(policy_states[:, i], label=f'State {i}')
        axs[i].plot(gt_states[:, i], label=f'GT State {i}')
        axs[i].set_xlabel(f'Action {i}')
        axs[i].legend()
        axs[i].grid(True)

    plt.suptitle('Policy States vs Ground Truth States')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


def main(dataset_repo_id, env_hf_path, policy_path):
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
    env = LiftEnv(model_path='simple_scene.xml')
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


def eval_policy_output(dataset_repo_id, policy_path, episode_idx):
    # Load your custom dataset containing the expert demonstrations
    dataset = LeRobotDataset(dataset_repo_id, revision="main")

    # Extract ground-truth actions and states for a baseline comparison episode
    episode_idx = episode_idx
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
        action = policy.predict(state=state, images={"sideview": frame}, task=dataset.meta.tasks[dataset.meta.tasks == episode_df['task_index'].iloc[0]].index[0])
        policy_actions.append(action[0])

    policy_actions = np.array(policy_actions)
    plot_alignment(policy_actions, gt_actions, gt_states, gt_states)

if __name__ == "__main__":
    fire.Fire(eval_policy_output)