"""Thin SmolVLM2 chat wrapper for the planning module.

Loading mirrors ``MiniSAModel._init_vision_language_encoder`` (AutoProcessor +
AutoModelForImageTextToText + 4-bit BitsAndBytesConfig), but exposes a plain
generative chat interface instead of hidden states:

- ``chat(images, system, user) -> str`` — one multimodal turn, returns the
  generated text.
- ``chat_json(images, system, user, fallback) -> dict`` — same, but parses the
  reply as JSON with a tolerant extractor (strips code fences, finds the first
  balanced ``{...}``), retries once with a stricter reminder on parse failure,
  and finally returns ``fallback`` so callers always get a usable dict (e.g.
  "continue the current subtask").

Only the current frames are ever passed in; no images are stored between calls.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from .config import PlannerConfig

logger = logging.getLogger(__name__)

_JSON_RETRY_REMINDER = (
    "Your previous reply could not be parsed as JSON. "
    "Answer again with ONLY a single valid JSON object and no other text."
)


def extract_json(text: str) -> dict:
    """Extract the first JSON object from a model reply.

    Tolerates markdown code fences and surrounding prose: scans for the first
    balanced ``{...}`` block (ignoring braces inside strings) and parses it.

    Raises
    ------
    ValueError
        If no parseable JSON object is found.
    """
    # Strip code fences so ``` / ```json markers never end up inside the scan.
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    cleaned = "\n".join(lines)

    start = cleaned.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(cleaned)):
            char = cleaned[i]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # malformed; try the next opening brace
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = cleaned.find("{", start + 1)
    raise ValueError(f"no JSON object found in model reply: {text!r}")


class SmolVLMChat:
    """Quantized SmolVLM2 instruct model behind a minimal chat interface."""

    def __init__(self, cfg: PlannerConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else PlannerConfig()
        self.device = self.cfg.device
        self.model, self.processor = self._init_model()

    def _init_model(self) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
        cfg = self.cfg
        torch_dtype = getattr(torch, cfg.torch_dtype)
        processor = AutoProcessor.from_pretrained(
            cfg.model_name,
            use_fast=True,
            size={"longest_edge": cfg.image_longest_edge},
            max_image_size={"longest_edge": cfg.image_longest_edge},
        )
        if cfg.load_in_4bit:
            model = AutoModelForImageTextToText.from_pretrained(
                cfg.model_name,
                torch_dtype=torch_dtype,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=cfg.torch_dtype,
                ),
                device_map=cfg.device,
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                cfg.model_name,
                torch_dtype=torch_dtype,
            ).to(cfg.device)
        model.eval()
        return model, processor

    @torch.no_grad()
    def chat(self, images: list[np.ndarray], system: str, user: str) -> str:
        """Run one multimodal chat turn and return the generated text.

        Parameters
        ----------
        images : list of (H, W, 3) uint8 arrays
            Current camera frames only (e.g. the values of the ``frames``
            dict produced in ``sim.py``).
        system, user : str
            System and user prompt texts. Images are attached to the user turn.
        """
        conversation = []
        if system:
            conversation.append(
                {"role": "system", "content": [{"type": "text", "text": system}]}
            )
        user_content: list[dict] = [
            {"type": "image", "image": np.ascontiguousarray(img)} for img in images
        ]
        user_content.append({"type": "text", "text": user})
        conversation.append({"role": "user", "content": user_content})

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            do_image_splitting=False,
        ).to(self.model.device)

        generate_kwargs: dict = {"max_new_tokens": self.cfg.max_new_tokens}
        if self.cfg.temperature > 0:
            generate_kwargs.update(do_sample=True, temperature=self.cfg.temperature)
        else:
            generate_kwargs["do_sample"] = False

        output_ids = self.model.generate(**inputs, **generate_kwargs)
        # Decode only the newly generated tokens (drop the prompt).
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    def chat_json(
        self,
        images: list[np.ndarray],
        system: str,
        user: str,
        fallback: dict,
    ) -> dict:
        """``chat`` + tolerant JSON parsing, one retry, then ``fallback``.

        Never raises on model misbehavior: if neither the first reply nor the
        retry contains a parseable JSON object, returns ``fallback`` (the
        caller supplies a safe default such as continuing the current subtask).
        """
        prompt = user
        for attempt in range(2):
            reply = self.chat(images, system, prompt)
            try:
                return extract_json(reply)
            except ValueError:
                logger.warning(
                    "planning VLM reply was not valid JSON (attempt %d): %r",
                    attempt + 1,
                    reply,
                )
                prompt = f"{user}\n\n{_JSON_RETRY_REMINDER}"
        logger.warning("planning VLM failed to produce JSON twice; using fallback %r", fallback)
        return dict(fallback)
