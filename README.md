# VLA On a Mobile Robot

A MuJoCo simulation playground for training and evaluating vision-language-action (VLA) policies on a custom mobile lift robot. Record demonstrations with a joystick, fine-tune [SmolVLA](https://huggingface.co/lerobot/smolvla_base) via [LeRobot](https://github.com/huggingface/lerobot), and run policies in the interactive simulator.

## Overview

This repo provides an end-to-end loop:

1. **Record** (robot_sim.py, record=True) — Teleoperate with a gamepad and save LeRobot-format datasets locally, ready to be uploaded to huggingface as a dataset to be used by the LeRobot ecosystem.
2. **Train** — Fine-tune SmolVLA (LoRA or full) on recorded data using the lerobot-train script.
3. **Evaluate** — Run policies in sim, compare against ground-truth actions, and inspect alignment.
4. **Run Policy** (robot_sim.py, record=False) — Run trained policy in simulator.

The custom Gymnasium environment lives in `vlamobile/env.py` and is published to Hugging Face for remote loading during training and eval (via the upload_env.py script). Currently, the published environment can be found in huggingface in YinonDouchan/mobile_robot_lift_env.

## Project structure

```
vla_playground/
├── assets/
│   └── mobile_robot_lift/      # MuJoCo robot + scene
│       ├── robot.xml           # Robot model
│       ├── scene.xml           # Scene (floor, cubes, platforms, cameras)
│       ├── meshes/             # STL meshes (box, cup, …)
│       └── textures/           # PNG textures (floor, wood, ceramic, …)
├── vlamobile/
│   ├── env.py                  # make_env() for multi cube multi platform setting
│   ├── env_single_cube_single_platform.py   # make_env() for single cube single platform setting
│   ├── env_single_cube_multi_platform.py   # make_env() for single cube multi platform setting
│   ├── lift_env.py             # Main environment implementation
│   ├── robot_sim.py            # Interactive MuJoCo sim: record + policy inference
│   ├── recording.py            # LeRobot dataset recorder utilities
│   ├── inference.py            # LeRobot policy inference utilities
│   ├── tasks.py                # Natural-language task generators
│   ├── eval_alignment.py       # Policy vs dataset action comparison plots
│   ├── upload_env.py           # Push env + assets to Hugging Face
│   └── tag_dataset.py          # Tag datasets with LeRobot v3.0 revision
├── pyproject.toml              # Project & dependency configuration
└── README.md                   # This file
```

## Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended for training and inference)
- [MuJoCo](https://mujoco.org/)
- [LeRobot](https://github.com/huggingface/lerobot)
- A gamepad (optional, for teleoperation during recording)
- Hugging Face account (`huggingface-cli login`)

## Setup

```bash
git clone <repo-url>
cd vla_playground

python -m venv .venv
source .venv/bin/activate

# TODO: document exact install command, e.g.
# pip install lerobot mujoco pygame fire matplotlib huggingface_hub

huggingface-cli login
```

Run scripts from the **repository root** so imports and asset paths resolve correctly:

```bash
export PYTHONPATH="${PWD}"
# or prefer module invocation:
python -m vlamobile.robot_sim --help
```

## Usage

### Upload the environment to Hugging Face

After changing `vlamobile/env.py` or scene assets:

```bash
python vlamobile/upload_env.py <huggingface repo name>
```

### Record demonstrations

```bash
python -m vlamobile.robot_sim \
  --record=True \
  --dataset_repo_id=YinonDouchan/mobile_robot_lift_v1 \
  --local_data_root=data \
  --task=pick_and_place
```

After recording, push the dataset to Hugging Face:

```bash
hf upload --repo-type=dataset <dataset_name> <local_data_root> --commit-message <commit_message>
```

After dataset was pushed, tag it for LeRobot:

```bash
python -m vlamobile.tag_dataset YinonDouchan/mobile_robot_lift_v1
```

### Train a policy

```bash
lerobot-train \
  --job_name=smolvla_mobile_robot_lift_v1 \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=YinonDouchan/smolvla_mobile_robot_lift_v1 \
  --dataset.repo_id=YinonDouchan/mobile_robot_lift_v1 \
  --output_dir=checkpoints/smolvla_mobile_robot_lift_v1 \
  # optional - use PEFT
  --peft.method_type=LORA \
  --peft.r=64
```

### Run policy in simulation

```bash
python -m vlamobile.robot_sim \
  --record=False \
  --compare_policy=True \
  --policy_path=YinonDouchan/smolvla_mobile_robot_lift_v1 \
  --task="Pick up the small cube"
```

### Evaluate action alignment

```bash
python -m vlamobile.eval_alignment \
  --dataset_repo_id=YinonDouchan/mobile_robot_lift_v1 \
  --env_hf_path=YinonDouchan/mobile_robot_lift_env \
  --policy_path=YinonDouchan/smolvla_mobile_robot_lift_v1 \
  --episode_idx=0
```

## Controls

### Keyboard (recording mode)


| Key     | Action                  |
| ------- | ----------------------- |
| `Space` | Reset environment       |
| `R`     | Start recording episode |
| `S`     | Save episode            |
| `E`     | Discard episode         |
| `F`     | Finalize dataset        |


### Keyboard (policy mode)

| Key     | Action                  |
| ------- | ----------------------- |
| `Space` | Reset environment       |
| `O`     | Toggle joystick override|



### Gamepad (default axes)


| Input                              | Action             |
| ---------------------------------- | ------------------ |
| Axis 4 (right joystick up/down)    | Forward / backward |
| Axis 3 (right joystick left/right) | Turn               |
| Axis 1 (left joystick down)        | Lift               |
| Axis 2 (left trigger)              | Grip (continuous)  |


## Environment

`LiftEnv` exposes:

- **Observations:** `observation.state` (4D proprioception: yaw, lift, grip_r, grip_l), `observation.images.sideview` (480×640 RGB)
- **Actions:** `[forward, turn, grip, lift]` — velocity-style base control, continuous grip `[0, 1]`, lift height

Task strings are natural-language instructions passed to the recorder and policy (see `vlamobile/tasks.py` for generators like `pick_and_place`, `turn_to_cube`, `turn_right`).

## HuggingFace artifacts

### Environemnts


| Artifact    | Repo                                 | Description                |
| ----------- | ------------------------------------ | ------------------- |
| Environment | `YinonDouchan/mobile_robot_lift_env` | multi-cube multi-platform |
| Environment | `YinonDouchan/mobile_robot_lift_env_single_cube_single_platform` | single-cube single-platform |


### Datasets and Trained policies


| Dataset Name   | Dataset Repo                                            | Trained Policy Repo                                 | Description                                              |
| -------------- | ------------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------- |
| Pick and place, single robot and platform | `YinonDouchan/smolvla_mobile_robot_lift_pick_and_place_single` | `YinonDouchan/mobile_robot_lift_pick_and_place`     | Pick the cube and place it on the platform |


## Development notes

- Scripts under `vlamobile/` use absolute imports (`from vlamobile...`). Run from the repo root with `PYTHONPATH` set, or use `python -m vlamobile.<script>`.

## Research notes

- Simulation timestep is set to 30 FPS
- Made side view camera a little closer so objects will appear larger
- Policy starts several frames from sim start to let simulation stabilize. This prevents policy from observing out of distribution observations.
- Adding a first person front camera to the robot was a game changer
- Obtained decent performance on pick and place for single robot and platform setting.
  - When cube is to the side of the robot or in the far edges of the environment, the robot sometimes misses it. It's a matter of polishing the dataset.
  - When creating demonstrations of picking up a cube, make sure the cube is inside the gripper enough - this prevents the policy from missing the cube with the gripper.

## TODO

- Test `pyproject.toml` dependencies

## License

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) — training, datasets, and policy infrastructure
- [MuJoCo](https://mujoco.org/) — physics simulation

