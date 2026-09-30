from dataclasses import dataclass, field
from pathlib import Path

import draccus
import numpy as np
import torch
import torch.nn.functional as F
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import LeRobotDataset, LeRobotDatasetMetadata
from transformers import Trainer, TrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME

from .config import FrozenVLAConfig
from .model import FrozenVLAModel


def frozen_vla_collate_fn(features: list[dict], camera_keys: list[str]) -> dict:
    return {
        "images": [
            [item[f"observation.images.{camera}"] for camera in camera_keys]
            for item in features
        ],
        "task": [item["task"] for item in features],
        "state": torch.stack([item["observation.state"] for item in features]),
        "action": torch.stack([item["action"] for item in features]),
    }


def frozen_vla_compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)
    return {"action_mse": float(np.mean((predictions - labels) ** 2))}


class FrozenVLATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        targets = inputs["action"]
        predictions = model(inputs)
        loss = F.mse_loss(predictions, targets)
        # Must return a dict: Trainer does `outputs[1:]` on non-dict outputs, which
        # drops the first batch item from a raw prediction tensor.
        return (loss, {"predictions": predictions}) if return_outputs else loss

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        output_dir = Path(output_dir if output_dir is not None else self.args.output_dir)
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        model.save_pretrained(output_dir)
        torch.save(self.args, output_dir / TRAINING_ARGS_NAME)


@dataclass
class TrainConfig:
    dataset: DatasetConfig
    training_args: TrainingArguments
    model: FrozenVLAConfig
    num_eval_episodes: int = 10
    camera_keys: list[str] = field(
        default_factory=lambda: ["robotfrontview_high", "robotfrontview"]
    )


@draccus.wrap()
def main(cfg: TrainConfig):
    cfg.training_args.remove_unused_columns = False
    meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    n_episodes = meta.total_episodes

    train_episodes = list(range(n_episodes - cfg.num_eval_episodes))
    eval_episodes = list(range(n_episodes - cfg.num_eval_episodes, n_episodes))

    delta_timestamps = {
        "action": [i / meta.fps for i in range(cfg.model.action_chunk_size)],
    }

    model = FrozenVLAModel(cfg.model)

    train_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=train_episodes,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        delta_timestamps=delta_timestamps,
    )
    eval_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=eval_episodes,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        delta_timestamps=delta_timestamps,
    )

    def collate_fn(features: list[dict]) -> dict:
        return frozen_vla_collate_fn(features, cfg.camera_keys)

    trainer = FrozenVLATrainer(
        model=model,
        args=cfg.training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=frozen_vla_compute_metrics,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
