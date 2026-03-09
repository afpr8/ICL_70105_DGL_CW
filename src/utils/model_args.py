# A model args dataclass base class for re-usability

from dataclasses import dataclass

@dataclass
class BaseModelArgs:
    lr: float = 1e-4
    batch_size: int = 1
    epochs: int = 1
    weight_decay: float = 0
    padding: int = 26
    lr_dim: int = 160
    hr_dim: int = 320
