# Frozen VLA

A vision-language-action policy that loads a Hugging Face pretrained model and trains a lightweight action-chunk head on its output features. The pretrained weights stay frozen. Training updates only the head.

Camera views and the language task go into the pretrained model. The head reads that model's output features, together with proprioception, and predicts a chunk of future actions for the mobile lift robot. Recorded LeRobot datasets from the rest of this repo are the training data.

Pretrained weights are not stored in this folder. They are loaded from the Hugging Face Hub or from a local path.

## Contents

```
frozen_vla/
├── model.py    # pretrained model loader and the action-chunk head
├── config.py   # model id, quantization, action size, and chunk length
└── train.py    # training entry point for the head
```

### `model.py`

Loads a Hugging Face pretrained vision-language model and defines the action-chunk head attached to its output features. Only the `huggingface` backend is implemented.

The debug launch in `.vscode/launch.json` (`frozen_vla/model.py`) runs this module with:

- `--backend=huggingface`
- `--model_id_or_path=Qwen/Qwen3.5-4B`
- `--load_in_4bit=true`

The head reads the last non-padding token of the frozen model, concatenates proprioception, and maps that vector to an action chunk of shape `(chunk_size, action_dim)`.

### `config.py`

`FrozenVLAConfig`: backend (`huggingface`), model id or local path, optional GGUF file, 4-bit loading, proprioception size, action dimension, and action-chunk length.

### `train.py`

Training entry point using Hugging Face `Trainer` (same pattern as `minisa/train.py`). It loads a LeRobot dataset with future action chunks, keeps the pretrained model frozen, and trains the action head with MSE loss.

Example:

```bash
python -m vlamobile.models.frozen_vla.train \
  --dataset.repo_id=YinonDouchan/mobile_robot_lift_pick_and_place_single_multi_egocentric \
  --training_args.output_dir=./checkpoints/frozen_vla_mobile_robot_lift \
  --training_args.per_device_train_batch_size=4 \
  --training_args.learning_rate=1e-4 \
  --training_args.num_train_epochs=1 \
  --training_args.label_names='["action"]'
```
