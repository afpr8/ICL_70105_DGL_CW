# A model args dataclass base class for re-usability

from dataclasses import dataclass

@dataclass
class BaseModelArgs:
    lr: float
    batch_size: int
    epochs: int
    weight_decay: float
