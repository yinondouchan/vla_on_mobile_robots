"""
MiniSAModel: DINOv2 vision-language-action regressor. Gets task text, observations, and action, and returns a scalar.
This model can either be used as as a Q-critic or a reward model.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from .config import MiniSAConfig


DEFAULT_MAX_TASK_LEN = 64
DEFAULT_IMAGE_SIZE = 224


class MiniSAModel(nn.Module):
    """Lightweight success classifier for the mobile lift robot.

    Inputs
    ------
    img1, img2 : (B, 3, H, W) or (B, H, W, 3)
        Two camera views (order should be consistent). Values may be uint8
        in [0, 255] or float in [0, 1]; resized to 224×224 internally.
    state : (B, N)
    action : (B, M)
    task : str | Sequence[str]
        Language instruction(s), one per batch element (or a single string broadcast).

    Output
    ------
    logits : (B,)  raw success logits (sigmoid → P(success) = expected binary reward).
    """

    def __init__(
        self,
        cfg: MiniSAConfig,
    ) -> None:
        super().__init__()
        self.max_task_len = cfg.max_task_len
        self.image_size = cfg.image_size
        self.freeze_backbone = cfg.freeze_backbone
        self.freeze_text_encoder = cfg.freeze_text_encoder
        self.device = cfg.device
        self.vision_language_encoder, self.processor = self._init_vision_language_encoder(cfg.vision_language_model_name, cfg.device)

        backbone_out_dim = self.vision_language_encoder.config.text_config.hidden_size
        if cfg.freeze_backbone:
            for param in self.vision_language_encoder.parameters():
                param.requires_grad = False
            self.vision_language_encoder.eval()

        self.state_enc = nn.Sequential(
            nn.Linear(cfg.proprio_dim, cfg.state_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(cfg.state_hidden, cfg.state_hidden)
        ).to(self.device, dtype=self.vision_language_encoder.dtype)

        self.action_enc = nn.Sequential(
            nn.Linear(cfg.action_dim * cfg.action_chunk_size, cfg.action_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(cfg.action_hidden, cfg.action_hidden)
        ).to(self.device, dtype=self.vision_language_encoder.dtype)

        fusion_in = backbone_out_dim + cfg.state_hidden + cfg.action_hidden
        self.head = nn.Sequential(
            nn.Linear(fusion_in, cfg.fusion_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden, cfg.fusion_hidden // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(cfg.fusion_hidden // 2, 1),
        ).to(self.device, dtype=self.vision_language_encoder.dtype)

    def _init_vision_language_encoder(self, vision_language_model_name: str, device: str) -> None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
        )

        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct",
         use_fast=True, size={"longest_edge": 512}, max_image_size={"longest_edge": 512})
        model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM2-256M-Instruct",
            torch_dtype=torch.bfloat16,
            # _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
            quantization_config=quantization_config
        ).to(device)
        return model, processor

    def train(self, mode: bool = True) -> MiniSAModel:
        super().train(mode)
        # Keep frozen encoders in eval (dropout / BN disabled).
        if self.freeze_backbone:
            self.vision_language_encoder.eval()
        return self

    def reset(self):
        """ This model is stateless, so this does nothing."""

    def get_optim_params(self) -> dict:
        """Return parameters to pass to the optimizer (e.g. with per-group lr/wd)."""
        return {"params": self.parameters()}

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return the full action chunk (B, chunk_size, action_dim) for the current observation."""
        return self.predict_proba(self.forward(batch))

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return a single action for the current timestep (called every step at inference)."""
        return self.predict_action_chunk(batch)


    def _prepare_conversations(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img1.permute(1, 2, 0)},
                        {"type": "image", "image": img2.permute(1, 2, 0)},
                        {"type": "text", "text": task}
                    ]
                }
            ]
            for task, img1, img2 in zip(batch["task"], batch["img1"], batch["img2"])
        ]

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        conversations = self._prepare_conversations(batch)
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            do_image_splitting=False,
            do_rescale=False
        ).to(self.device)
        with torch.no_grad():
            outputs = self.vision_language_encoder(**inputs, output_hidden_states=True)
        
        # TODO: For the sake of simplicity, we perform late fusion with state and action encoding.
        # to be implemented - inject state and action embeddings to VLM prefix
        vl_mean_pool = torch.mean(outputs.hidden_states[-1], dim=1)
        state = batch["state"].view(batch["state"].size(0), -1).to(next(self.state_enc.parameters()).dtype)
        action = batch["action"].view(batch["action"].size(0), -1).to(next(self.action_enc.parameters()).dtype)
        h_state = self.state_enc(state)
        h_action = self.action_enc(action)
        fused = torch.cat([vl_mean_pool, h_state, h_action], dim=-1)
        return self.head(fused)

    @torch.no_grad()
    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Return P(success) in [0, 1]."""
        was_training = self.training
        self.eval()
        try:
            return torch.sigmoid(logits)
        finally:
            self.train(was_training)

    def success_loss(self, logits: torch.Tensor, success: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy with logits. ``success`` is float/bool in {0, 1}."""
        return F.binary_cross_entropy_with_logits(logits, success.float().view_as(logits))


if __name__ == "__main__":
    cfg = MiniSAConfig()
    model = MiniSAModel(cfg)
    batch_size = 4
    model.forward({"img1": torch.rand(batch_size, 3, 512, 512).to(cfg.device), "img2": torch.rand(batch_size, 3, 512, 512).to(cfg.device), "task": ["lift the object" for _ in range(batch_size)],
     "state": torch.randn(batch_size, 4).to(cfg.device), "action": torch.randn(batch_size, 4).to(cfg.device)})
