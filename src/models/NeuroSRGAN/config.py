# Configuration for NeuroSRGAN

# Standard Python library imports
from dataclasses import dataclass, field
from typing import Literal

# Local imports
from src.utils.model_args import BaseModelArgs


@dataclass
class NeuroSRGANArgs(BaseModelArgs):
    lr: float = 1e-4
    epochs: int = 200
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

    # NeuroSRGAN-specific hyperparameters
    variant: Literal['full', 'community_only', 'topology_only', 'baseline'] = 'full'
    K_communities: int = 7      # number of Louvain communities
    rank: int = 16              # low-rank correction dimension for CommunityAwareSRLayer
    threshold_pct: float = 80.0 # percentile threshold for Louvain input graph

    # Topology discriminator weighting
    # topo_scale: multiplicative factor applied to the 3 topology scalars before
    #   concatenation, compensating for their dilution by the 320^2 matrix values
    # topo_init_scale: multiplier on the default weight init (std_dense) for the
    #   3 topology input rows of dense_1, so those neurons start more sensitive
    topo_scale: float = 100.0
    topo_init_scale: float = 10.0

    # GAN stability controls
    # gen_loss_weight: scales the adversarial gen_loss relative to mse_loss,
    #   preventing the discriminator signal from overwhelming reconstruction
    # grad_clip: max gradient norm for both generator and discriminator;
    #   guards against sudden large updates causing mode collapse
    gen_loss_weight: float = 0.1
    grad_clip: float = 1.0
