from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig


class MiniSAConfig(PreTrainedConfig):
    """Configuration class for MyPolicy.

    """
    proprio_dim: int = 4
    action_dim: int = 4
    action_chunk_size: int = 50
    pretrained: bool = True
    vision_language_model_name: str = "google/siglip2-base-patch16-224"
    vision_dim: int = 224
    state_hidden: int = 64
    action_hidden: int = 512
    text_dim: int = 128
    fusion_hidden: int = 256
    dropout: float = 0.1
    freeze_backbone: bool = True
    freeze_text_encoder: bool = True
    max_task_len: int = 10
    image_size: int = 224

    horizon: int = 1
    n_action_steps: int = 1

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.horizon:
            raise ValueError("n_action_steps cannot exceed horizon")

    def validate_features(self) -> None:
        """Validate input/output feature compatibility.

        Call this explicitly from your policy's __init__ — the base class does not.
        """
        if not self.image_features:
            raise ValueError("MiniSA requires at least one image feature.")
        if self.action_feature is None:
            raise ValueError("MiniSA requires 'action' in output_features.")

    def get_scheduler_preset(self):
        """Return a LRSchedulerConfig from lerobot.optim, or None."""
        return None


    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)


    def observation_delta_indices(self) -> list[int] | None:
        """Relative timestep offsets the dataset loader provides per observation.

        Return `None` for single-frame policies. For temporal policies that consume
        multiple past or future frames, return a list of offsets, e.g. `[-20, -10, 0, 10]` for
        3 past frames at stride 10 and 1 future frame at stride 10.
        """
        return None


    def action_delta_indices(self) -> list[int]:
        """Relative timestep offsets for the action chunk the dataset loader returns."""
        return list(range(self.horizon))


    def reward_delta_indices(self) -> None:
        return None