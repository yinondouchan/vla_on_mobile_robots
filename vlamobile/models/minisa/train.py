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
        "success": torch.stack([item["success"].to(int) for item in features]),
    }


class MiniSATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs)
        loss = self.loss_fn(outputs, inputs["success"][:, None].float())
        return (loss, outputs) if return_outputs else loss


@dataclass
class TrainConfig:
    dataset: DatasetConfig
    training_args: TrainingArguments
    model: MiniSAConfig | None = None
    num_eval_episodes: int = 10
    policy_path: str | None = None


@parser.wrap()
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

    if cfg.model is None:
        cfg.model = MiniSAConfig()


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
        # optional: data_collator, compute_metrics, optimizers=...
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()