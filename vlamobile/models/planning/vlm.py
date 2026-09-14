"""Chat wrappers for the planning module.

``HuggingFaceChat`` loads a quantized instruct VLM locally
(``AutoProcessor`` + ``AutoModelForImageTextToText`` + 4-bit BitsAndBytes)
and exposes a multimodal generative chat interface:

- ``chat(messages) -> str`` — one multimodal turn from a chat-template
  message list, returns the generated text.
- ``chat_json(messages, fallback) -> dict`` — same, but parses the reply as
  JSON with a tolerant extractor (strips code fences, finds the first
  balanced ``{...}``), retries once with a stricter reminder on parse
  failure, and finally returns ``fallback`` so callers always get a usable
  dict (e.g. "continue the current subtask").

Only the current frames are ever passed in; no images are stored between calls.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

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


def _to_pil(image: np.ndarray | Image.Image) -> Image.Image:
    """Convert an (H, W, 3) uint8 array (or pass through a PIL image) to RGB."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.ascontiguousarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shallow-copy messages, converting any numpy image parts to PIL."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(message)
            continue
        new_content: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "image" and "image" in part:
                new_content.append({**part, "image": _to_pil(part["image"])})
            else:
                new_content.append(part)
        normalized.append({**message, "content": new_content})
    return normalized


def _with_json_retry_reminder(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy ``messages`` and append the JSON retry reminder to the last user text."""
    copied: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            copied.append({**message, "content": [dict(part) for part in content]})
        else:
            copied.append(dict(message))

    for message in reversed(copied):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            for part in reversed(content):
                if part.get("type") == "text":
                    part["text"] = f"{part['text']}\n\n{_JSON_RETRY_REMINDER}"
                    return copied
            content.append({"type": "text", "text": _JSON_RETRY_REMINDER})
            return copied
        if isinstance(content, str):
            message["content"] = f"{content}\n\n{_JSON_RETRY_REMINDER}"
            return copied

    copied.append(
        {"role": "user", "content": [{"type": "text", "text": _JSON_RETRY_REMINDER}]}
    )
    return copied


class HuggingFaceChat:
    """Local HuggingFace VLM chat client (same API as the planning notebooks).

    Loads ``AutoProcessor`` + ``AutoModelForImageTextToText`` with optional
    4-bit quantization and exposes ``chat`` / ``chat_json`` for the planner.
    """

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3.5-4B",
        temperature: float = 1.0,
        top_p: float = 0.95,
        enable_thinking: bool = False,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
    ) -> None:
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

        self.model_name = model
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking

        self.processor = AutoProcessor.from_pretrained(model)
        model_kwargs: dict = {
            "torch_dtype": "auto",
            "device_map": "auto",
        }
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                # This prevents the final classification layer from dropping precision,
                # protecting output logic while keeping weights compressed.
                llm_int8_skip_modules=["lm_head"] 
            )
            
        self.model = AutoModelForImageTextToText.from_pretrained(model, **model_kwargs)
        self.model.eval()

    @torch.no_grad()
    def chat(self, messages: list[dict[str, Any]], max_new_tokens: int = 1024) -> str:
        """Run one multimodal chat turn and return the generated text.

        Parameters
        ----------
        messages : list of dict
            Chat-template messages. Image content parts may carry either a
            PIL image or an (H, W, 3) uint8 numpy array under ``"image"``.
        max_new_tokens : int
            The maximum number of new tokens to generate.
        return_logprobs : bool
            Whether to return the log probabilities of the generated tokens.
        """
        messages = _normalize_messages(messages)

        model_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.enable_thinking,
        ).to(self.model.device)

        generate_kwargs: dict = {"max_new_tokens": max_new_tokens}
        if self.temperature > 0:
            generate_kwargs.update(
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        else:
            generate_kwargs["do_sample"] = False

        generated_ids = self.model.generate(**model_inputs, **generate_kwargs, return_dict_in_generate=True, output_scores=True)
        return self.processor.decode(generated_ids.sequences[0][len(model_inputs.input_ids[0]) :].tolist(), skip_special_tokens=True).strip()

    def get_token_probability(self, messages: list[dict[str, Any]], token_str: str, max_new_tokens: int = 1) -> float:
        """
        Same as `chat` but returns the probability of the first generated token.
        token_str : str
            The string representation of the token to get the probability of.

        Returns
        -------
        prob : float
            Probability (not log-prob) of the first generated token.
        """
        messages = _normalize_messages(messages)
        model_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.enable_thinking,
        ).to(self.model.device)

        generate_kwargs: dict = {"max_new_tokens": max_new_tokens}
        if self.temperature > 0:
            generate_kwargs.update(
                do_sample=False,
                temperature=self.temperature,
                top_p=1.0,
            )
        else:
            generate_kwargs["do_sample"] = False

        output = self.model.generate(
            **model_inputs, 
            **generate_kwargs, 
            return_dict_in_generate=True,
            output_scores=True
        )
        # Grab logits/scores for first generated token
        scores = output.scores
        if not scores:
            return 0.0

        import torch.nn.functional as F

        logits = scores[0][0]  # scores[0]: first step, [0]: batch index
        probs = F.softmax(logits, dim=-1)
        token_id = self.processor.tokenizer.encode(token_str, add_special_tokens=False)[0]
        # first_gen_token_id = output.sequences[0][len(model_inputs.input_ids[0])]
        return probs[token_id].item()

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        fallback: dict,
    ) -> dict:
        """``chat`` + tolerant JSON parsing, one retry, then ``fallback``.

        Never raises on model misbehavior: if neither the first reply nor the
        retry contains a parseable JSON object, returns ``fallback``.
        """
        attempt_messages = messages
        for attempt in range(2):
            reply = self.chat(attempt_messages)
            try:
                return extract_json(reply)
            except ValueError:
                logger.warning(
                    "HuggingFace chat reply was not valid JSON (attempt %d): %r",
                    attempt + 1,
                    reply,
                )
                attempt_messages = _with_json_retry_reminder(messages)
        logger.warning(
            "HuggingFace chat failed to produce JSON twice; using fallback %r",
            fallback,
        )
        return dict(fallback)
