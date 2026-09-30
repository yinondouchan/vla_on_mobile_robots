from dataclasses import dataclass, field


@dataclass
class FrozenVLAConfig:
    """Frozen Hugging Face model plus a lightweight action-chunk head."""

    backend: str = "huggingface"
    model_id_or_path: str = "Qwen/Qwen3.5-4B"
    gguf_file: str | None = None
    load_in_4bit: bool = True

    proprio_dim: int = 4
    action_dim: int = 4
    action_chunk_size: int = 50
    # Layer indices into ``outputs.hidden_states`` for the action head (-1 = last layer).
    feature_layer_indices: list[int] = field(default_factory=lambda: [-1])
