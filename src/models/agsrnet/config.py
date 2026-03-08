# Standard Python library imports
from dataclasses import dataclass, field

# Local imports
from src.utils.model_args import BaseModelArgs


@dataclass
class AGSRArgs(BaseModelArgs):
    lr: float = 1e-4
    epochs: int = 1
    lmbda: float = 16  # self-reconstruction loss weight
    lr_dim: int = 160
    hr_dim: int = 320  # padded 268 + 26*2
    padding: int = 26
    K: int = 2  # super-resolution factor
    ks: list[float] = field(default_factory=lambda: [0.9, 0.7, 0.6, 0.5])
    hidden_dim: int = 320
    mean_dense: float = 0.0
    std_dense: float = 0.01
    mean_gaussian: float = 0.0
    std_gaussian: float = 0.0
    batch_size: int = 1
    weight_decay: float = 0.0
