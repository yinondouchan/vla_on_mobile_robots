# VLA Playground

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
├── assets/mobile_robot_lift/   # MuJoCo robot + scene meshes and textures
├── vlamobile/
│   ├── env.py                  # LiftEnv + make_env() for HF hub
│   ├── robot_sim.py            # Interactive sim: record + policy inference
│   ├── recording.py            # LeRobot dataset recorder
│   ├── inference.py            # LerobotInference policy wrapper
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


### Gamepad (default axes)


| Input                              | Action             |
| ---------------------------------- | ------------------ |
| Axis 4 (right joystick up/down)    | Forward / backward |
| Axis 3 (right joystick left/right) | Turn               |
| Axis 1 (left joystick down)        | Lift               |
| Button 4 (upper left bumper)       | Toggle grip        |


## Environment

`LiftEnv` exposes:

- **Observations:** `observation.state` (6D proprioception), `observation.images.sideview` (480×640 RGB)
- **Actions:** `[forward, turn, grip, lift]` — velocity-style base control, binary grip, lift height

Task strings are natural-language instructions passed to the recorder and policy (see `vlamobile/tasks.py` for generators like `pick_and_place`, `turn_to_cube`, `turn_right`).

## Hugging Face artifacts


| Artifact    | Repo                                 | Type                |
| ----------- | ------------------------------------ | ------------------- |
| Environment | `YinonDouchan/mobile_robot_lift_env` | model (remote code) |



| Dataset Name   | Dataset Repo | Trained Policy Repo | Description |
| -------------- | --------------- | ---------------- | ------------------------ |
| Pick and place | `YinonDouchan/smolvla_mobile_robot_lift_v1`         | `YinonDouchan/mobile_robot_lift_v1`                | Pick one of the cubes and put it on one of the platforms |
| Turn to cube   | `YinonDouchan/smolvla_mobile_robot_lift_dummy_task` | `YinonDouchan/mobile_robot_lift_dummy_task`        | Turn towards one of the cubes                            |
| Turn right     | `YinonDouchan/mobile_robot_lift_turn_task`          | `YinonDouchan/smolvla_mobile_robot_lift_turn_task` | Turn to the right                                        |


## Development notes

- Scripts under `vlamobile/` use absolute imports (`from vlamobile...`). Run from the repo root with `PYTHONPATH` set, or use `python -m vlamobile.<script>`.

## TODO

- Test `pyproject.toml` dependencies

## License

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) — training, datasets, and policy infrastructure
- [MuJoCo](https://mujoco.org/) — physics simulation

