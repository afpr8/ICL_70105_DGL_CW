# Standard Python library imports
from dataclasses import dataclass

# Local imports
from src.utils.model_args import BaseModelArgs

@dataclass
class NaiveGCNArgs(BaseModelArgs):
    lr: float = 1e-4
    epochs: int = 200
    batch_size: int = 1
    weight_decay: float = 0.0

    lr_dim: int = 160
    hr_dim: int = 320  # padded 268 + 26*2
    padding: int = 26
    hidden_dim: int = 320
