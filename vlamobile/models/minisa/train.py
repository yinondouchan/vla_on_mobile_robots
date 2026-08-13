from dataclasses import dataclass
from transformers import TrainingArguments, Trainer
from .model import MiniSAModel
from .config import MiniSAConfig
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import LeRobotDataset, LeRobotDatasetMetadata, resolve_delta_timestamps

import draccus
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch
import lerobot.policies.factory # noqa: F401 import to register policies
import numpy as np


image_transforms = transforms.Compose([
    transforms.Resize((512, 512))
])


def minisa_collate_fn(features: list[dict]) -> dict:
    img1 = [item["observation.images.robotfrontview_high"] for item in features]
    img2 = [item["observation.images.robotfrontview"] for item in features]

    return {
        "img1": torch.stack(img1),
        "img2": torch.stack(img2),
        "task": [item["task"] for item in features],
        "state": torch.stack([item["observation.state"] for item in features]),
        "action": torch.stack([item["action"] for item in features]),
        "success": torch.tensor([(1.0 if "success" not in item or item["success"] else 0.0) for item in features]),
        "progress": torch.stack([item["progress"] for item in features]),
    }


def minisa_compute_metrics(eval_pred):
    outputs, labels = eval_pred
    logits = outputs[0][:, 0]
    progress = outputs[0][:, 1]
    labels_success, labels_progress = labels
    preds = (1 / (1 + np.exp(-logits))).reshape(-1)
    preds_binary = (preds > 0.5).astype(np.float32)
    labels_success = labels_success.reshape(-1).astype(np.float32)

    tp = np.sum((preds_binary == 1) & (labels_success == 1))
    tn = np.sum((preds_binary == 0) & (labels_success == 0))
    fp = np.sum((preds_binary == 1) & (labels_success == 0))
    fn = np.sum((preds_binary == 0) & (labels_success == 1))
    pred_mean = np.mean(preds)
    pred_std = np.std(preds)
    hist, bin_edges = np.histogram(preds, bins=20, range=(0, 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0


    metrics = {
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "pred_hist": [(f"{bin_edge:.2f}", count) for count, bin_edge in zip(hist, bin_edges)]
    }

    preds_progress_successful = progress[labels_success == 1]
    labels_progress_successful = labels_progress[labels_success == 1]

    if len(preds_progress_successful) > 0:
        metrics["progress_mse_successful"] = np.mean((preds_progress_successful - labels_progress_successful) ** 2)
    else:
        metrics["progress_mse_successful"] = 0.0

    return metrics


class MiniSATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_success_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_progress_fn = torch.nn.MSELoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs)
        inputs_success = inputs["success"][:, None].to(outputs.dtype)
        inputs_progress = inputs["progress"][:, None].to(outputs.dtype)
        loss_success = self.loss_success_fn(outputs[:, 0].unsqueeze(1), inputs_success)
        loss_progress = torch.mean((outputs[:, 1].unsqueeze(1) - inputs_progress) ** 2 * inputs_success) # only compute progress loss for successful episodes
        loss = loss_success + loss_progress
        return (loss, {"logits": outputs, "loss_success": loss_success, "loss_progress": loss_progress}) if return_outputs else loss


@dataclass
class TrainConfig:
    dataset: DatasetConfig
    training_args: TrainingArguments
    model: MiniSAConfig
    num_eval_episodes: int = 10
    policy_path: str | None = None


@draccus.wrap()
def main(cfg: TrainConfig):
    cfg.training_args.remove_unused_columns = False
    meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    n_episodes = meta.total_episodes   # no frames/videos loaded

    train_episodes = list(range(n_episodes - cfg.num_eval_episodes))
    eval_episodes = list(range(n_episodes - cfg.num_eval_episodes, n_episodes))

    model = MiniSAModel(cfg.model)

    if cfg.policy_path is not None:
        policy_config = PreTrainedConfig.from_pretrained(cfg.policy_path)
        delta_timestamps = resolve_delta_timestamps(policy_config, meta)
    else:
        delta_timestamps = None

    train_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=train_episodes,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        image_transforms=image_transforms,
        delta_timestamps=delta_timestamps
    )

    eval_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=eval_episodes,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        image_transforms=image_transforms,
        delta_timestamps=delta_timestamps
    )

    trainer = MiniSATrainer(
        model=model,
        args=cfg.training_args,
        train_dataset=train_dataset,
        data_collator=minisa_collate_fn,
        eval_dataset=eval_dataset,
        compute_metrics=minisa_compute_metrics,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()