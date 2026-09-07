"""
MiniSAModel: DINOv2 vision-language-action regressor. Gets task text, observations, and action, and returns a scalar.
This model can either be used as as a Q-critic or a reward model.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from .config import MiniSAConfig


DEFAULT_MAX_TASK_LEN = 64
DEFAULT_IMAGE_SIZE = 224


class MiniSAModel(nn.Module):
    """Lightweight progress regressor for the mobile lift robot.

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
    progress : (B, 1)  predicted episode progress in [0, 1].
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
        self.forward_mode = cfg.forward_mode

        backbone_out_dim = self.vision_language_encoder.config.text_config.hidden_size
        if cfg.freeze_backbone:
            for param in self.vision_language_encoder.parameters():
                param.requires_grad = False
            self.vision_language_encoder.eval()

        self.state_enc = nn.Sequential(
            nn.Linear(cfg.proprio_dim, backbone_out_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(backbone_out_dim, backbone_out_dim)
        ).to(self.device, dtype=self.vision_language_encoder.dtype)

        self.action_enc = nn.Sequential(
            nn.Linear(cfg.action_dim * cfg.action_chunk_size, backbone_out_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(backbone_out_dim, backbone_out_dim)
        ).to(self.device, dtype=self.vision_language_encoder.dtype)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=backbone_out_dim,
            num_heads=4,
            batch_first=True
        ).to(self.device, dtype=self.vision_language_encoder.dtype)

        self.pool_query = nn.Parameter(torch.randn(1, 1, backbone_out_dim)).to(self.device, dtype=self.vision_language_encoder.dtype)

        if self.forward_mode == "late_fusion":
            fusion_in = backbone_out_dim * 2
        else:
            fusion_in = backbone_out_dim

        self.progress_head = nn.Sequential(
            nn.Linear(fusion_in, cfg.fusion_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden, cfg.fusion_hidden // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(cfg.fusion_hidden // 2, 1),
            nn.Sigmoid()
        ).to(self.device)

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
        if self.forward_mode == "early_fusion":
            return self.forward_early_fusion(batch)
        elif self.forward_mode == "late_fusion":
            return self.forward_late_fusion(batch)
        else:
            raise ValueError(f"Invalid forward mode: {self.forward_mode}")

    def forward_early_fusion(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
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
        input_ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        pixel_attn = inputs.get("pixel_attention_mask")

        text_embeds = self.vision_language_encoder.get_input_embeddings()(input_ids)
        image_feats = self.vision_language_encoder.get_image_features(
            pixel_values, pixel_attn, return_dict=True
        ).pooler_output
        merged = self.vision_language_encoder.model.inputs_merger(input_ids, text_embeds, image_feats)
        extra_attention_mask = torch.ones(merged.shape[0], 2, device=attn.device, dtype=attn.dtype)  # state + action tokens
        attention_mask = torch.cat([attn, extra_attention_mask], dim=1)

        state = batch["state"].view(batch["state"].size(0), -1).to(next(self.state_enc.parameters()).dtype)
        action = batch["action"].view(batch["action"].size(0), -1).to(next(self.action_enc.parameters()).dtype)
        state_emb = self.state_enc(state).unsqueeze(1)
        action_emb = self.action_enc(action).unsqueeze(1)
        inputs_embeds = torch.cat([merged, state_emb, action_emb], dim=1)
        
        with torch.no_grad():
            outputs = self.vision_language_encoder.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        cross_attn_output, _ = self.cross_attn(self.pool_query.expand(inputs_embeds.shape[0], -1, -1),
             outputs.hidden_states[-1], outputs.hidden_states[-1], key_padding_mask=~attention_mask.bool())
        
        return self.progress_head(cross_attn_output[:, 0].to(next(self.progress_head.parameters()).dtype))

    def forward_late_fusion(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
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
        
        vl_output = outputs.hidden_states[-1]
        state = batch["state"].view(batch["state"].size(0), -1).to(next(self.state_enc.parameters()).dtype)
        action = batch["action"].view(batch["action"].size(0), -1).to(next(self.action_enc.parameters()).dtype)
        h_state = self.state_enc(state)
        h_action = self.action_enc(action)
        cross_attn_input = torch.stack([h_state, h_action], dim=1)
        cross_attn_output, _ = self.cross_attn(cross_attn_input, vl_output, vl_output)
        cross_attn_output = cross_attn_output.reshape(cross_attn_output.shape[0], -1)
        return self.progress_head(cross_attn_output.to(next(self.progress_head.parameters()).dtype))

    @torch.no_grad()
    def predict_proba(self, progress: torch.Tensor) -> torch.Tensor:
        """Return predicted progress in [0, 1]."""
        was_training = self.training
        self.eval()
        try:
            return progress
        finally:
            self.train(was_training)


if __name__ == "__main__":
    cfg = MiniSAConfig()
    model = MiniSAModel(cfg)
    batch_size = 4
    model.forward({"img1": torch.rand(batch_size, 3, 512, 512).to(cfg.device), "img2": torch.rand(batch_size, 3, 512, 512).to(cfg.device), "task": ["lift the object" for _ in range(batch_size)],
     "state": torch.randn(batch_size, 4).to(cfg.device), "action": torch.randn(batch_size, 4).to(cfg.device)})
