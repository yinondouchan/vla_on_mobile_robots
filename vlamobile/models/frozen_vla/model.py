"""Frozen Hugging Face VLM with a lightweight action-chunk head."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import draccus
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from .config import FrozenVLAConfig

_ACTION_HEAD_WEIGHTS_NAME = "action_head.pt"
_MODEL_CONFIG_NAME = "frozen_vla_config.pt"

_MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "mm_token_type_ids",
)


def _to_pil(image: Image.Image | np.ndarray | torch.Tensor) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.numpy()
    array = np.ascontiguousarray(image)
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _image_lists(images) -> list[list]:
    if isinstance(images, torch.Tensor) and images.ndim == 4:
        return [[images[i]] for i in range(images.shape[0])]
    if isinstance(images, np.ndarray) and images.ndim == 4:
        return [[images[i]] for i in range(images.shape[0])]
    if isinstance(images[0], (list, tuple)):
        return list(images)
    return [[image] for image in images]


class FrozenVLAModel(nn.Module):
    """Frozen pretrained model and a small MLP action-chunk head.

    ``batch`` keys:
    - ``images``: one image, or a list of images, per sample (PIL, HWC numpy, or CHW torch)
    - ``task``: instruction string, or one string per sample
    - ``state``: proprioception, shape ``(B, proprio_dim)``

    Returns an action chunk of shape ``(B, action_chunk_size, action_dim)``.
    Pass ``return_hidden_states=True`` to also get the full per-layer sequence
    hidden states from the frozen backbone.
    """

    def __init__(self, cfg: FrozenVLAConfig) -> None:
        super().__init__()
        if cfg.backend != "huggingface":
            raise NotImplementedError(f"backend {cfg.backend!r} is not implemented")
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(cfg.model_id_or_path)
        model_kwargs: dict = {"torch_dtype": "auto", "device_map": "auto"}
        if cfg.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.backbone = AutoModelForImageTextToText.from_pretrained(cfg.model_id_or_path, **model_kwargs)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_size = self.backbone.config.text_config.hidden_size
        feature_dim = hidden_size * len(cfg.feature_layer_indices)
        self.action_head = nn.Sequential(
            nn.Linear(feature_dim + cfg.proprio_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, cfg.action_dim * cfg.action_chunk_size),
        ).to(self.backbone.device)

    def train(self, mode: bool = True) -> FrozenVLAModel:
        super().train(mode)
        self.backbone.eval()
        return self

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save only the action head and config. Backbone reloads from ``model_id_or_path``."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.action_head.state_dict(), save_directory / _ACTION_HEAD_WEIGHTS_NAME)
        torch.save(asdict(self.cfg), save_directory / _MODEL_CONFIG_NAME)

    def load_action_head(self, load_directory: str | Path) -> None:
        state_dict = torch.load(Path(load_directory) / _ACTION_HEAD_WEIGHTS_NAME, map_location="cpu", weights_only=True)
        self.action_head.load_state_dict(state_dict)

    def forward(
        self,
        batch: dict,
        *,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        features, hidden_states = self._backbone_features(batch["images"], batch["task"])
        state = batch["state"].to(device=features.device, dtype=self.action_head[0].weight.dtype)
        features = features.to(dtype=state.dtype)
        actions = self.action_head(torch.cat([features, state], dim=-1))
        actions = actions.view(-1, self.cfg.action_chunk_size, self.cfg.action_dim)
        if return_hidden_states:
            return actions, hidden_states
        return actions

    @staticmethod
    def _pool_sequence(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            return hidden[:, -1]
        last_index = attention_mask.cumsum(dim=1).argmax(dim=1)
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_index, last_index]

    def _backbone_features(
        self,
        images,
        tasks: str | Sequence[str],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        image_lists = _image_lists(images)
        if isinstance(tasks, str):
            tasks = [tasks] * len(image_lists)
        conversations = []
        for sample_images, task in zip(image_lists, tasks, strict=True):
            content = [{"type": "image", "image": _to_pil(image)} for image in sample_images]
            content.append({"type": "text", "text": task})
            conversations.append([{"role": "user", "content": content}])

        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.backbone.device)
        model_inputs = {key: inputs[key] for key in _MODEL_INPUT_KEYS if key in inputs}

        with torch.no_grad():
            outputs = self.backbone.model(
                **model_inputs,
                use_cache=False,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states
        mask = model_inputs.get("attention_mask")
        pooled_layers = [
            self._pool_sequence(hidden_states[layer_idx], mask)
            for layer_idx in self.cfg.feature_layer_indices
        ]
        features = pooled_layers[0] if len(pooled_layers) == 1 else torch.cat(pooled_layers, dim=-1)
        return features, hidden_states


@draccus.wrap()
def main(cfg: FrozenVLAConfig) -> None:
    model = FrozenVLAModel(cfg)
    actions = model(
        {
            "images": [torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8)],
            "task": ["pick up the cube"],
            "state": torch.zeros(1, cfg.proprio_dim),
        }
    )
    print(tuple(actions.shape))


if __name__ == "__main__":
    main()
