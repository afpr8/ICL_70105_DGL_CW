# NeuroSRGAN model and associated neural network components

# Standard library imports
from typing import Literal

# Third party imports
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path as sp_shortest_path

# Local imports
from .layers import CommunityAwareSRLayer, GSRLayer, GraphConvolution
from .ops import GraphUnet
from .preprocessing import normalize_adj_torch


class NeuroSRGAN(torch.nn.Module):
    """
    NeuroSRGAN — Neurodynamics-Informed Spectral Super-Resolution GAN

    Reconstructs a high-resolution brain connectivity graph from a
    low-resolution adjacency matrix. Extends AGSRNet with two novel
    contributions selectable via the variant parameter:

        'full'          — CommunityAwareSRLayer + TopologyAwareDiscriminator
        'community_only'— CommunityAwareSRLayer + StandardDiscriminator
        'topology_only' — GSRLayer + TopologyAwareDiscriminator
        'baseline'      — GSRLayer + StandardDiscriminator (≈ AGSRNet)

    Params:
        args: Configuration object (NeuroSRGANArgs) containing model hyperparameters
    """
    def __init__(self, args) -> None:
        super().__init__()

        self.lr_dim = args.lr_dim
        self.hr_dim = args.hr_dim
        self.hidden_dim = args.hidden_dim
        self.variant: Literal['full', 'community_only', 'topology_only', 'baseline'] = (
            getattr(args, 'variant', 'full')
        )

        self.ks = args.ks

        # SR layer: community-aware for full/community_only, plain GSR otherwise
        if self.variant in ('full', 'community_only'):
            self.layer = CommunityAwareSRLayer(
                self.hr_dim,
                k=args.K_communities,
                rank=args.rank
            )
        else:
            self.layer = GSRLayer(self.hr_dim)

        self.net = GraphUnet(self.ks, self.lr_dim, self.hr_dim)

        self.gc1 = GraphConvolution(
            self.hr_dim, self.hidden_dim, dropout=0.0, act=F.relu
        )
        self.gc2 = GraphConvolution(
            self.hidden_dim, self.hr_dim, dropout=0.0, act=F.relu
        )

    def forward(
            self,
            lr: torch.Tensor,
            masks: list[torch.Tensor] | None = None
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of NeuroSRGAN

        Params:
            lr: Low-resolution adjacency matrix of shape (lr_dim, lr_dim)
            masks: List of K binary community masks, each of shape (hr_dim, hr_dim).
                Required when variant is 'full' or 'community_only'; ignored otherwise
        Returns:
            z: Reconstructed high-resolution adjacency matrix (hr_dim, hr_dim)
            net_outs: Output of the Graph U-Net
            start_gcn_outs: Initial graph convolution outputs
            outputs: Intermediate adjacency matrix produced by the SR layer
        """
        device = lr.device

        I = torch.eye(self.lr_dim, device=device, dtype=lr.dtype)
        A = normalize_adj_torch(lr).to(device=device, dtype=torch.float32)

        net_outs, start_gcn_outs = self.net(A, I)

        if isinstance(self.layer, CommunityAwareSRLayer):
            outputs, Z = self.layer(A, net_outs, masks)
        else:
            outputs, Z = self.layer(A, net_outs)

        hidden1 = self.gc1(Z, outputs)
        hidden2 = self.gc2(hidden1, outputs)

        z = hidden2
        z = (z + z.T) / 2
        z.fill_diagonal_(1)

        return torch.abs(z), net_outs, start_gcn_outs, outputs


class Dense(torch.nn.Module):
    """
    Fully connected layer implemented using explicit weight multiplication

    Params:
        in_features: Number of input features
        out_features: Number of output features
        args: Configuration object containing initialization parameters
    """
    def __init__(self, in_features: int, out_features: int, args) -> None:
        super().__init__()
        self.weights = torch.nn.Parameter(torch.empty(in_features, out_features))
        torch.nn.init.normal_(
            self.weights,
            mean=args.mean_dense,
            std=args.std_dense
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation

        Params:
            x: Input tensor of shape (*, in_features)
        Returns:
            Output tensor of shape (*, out_features)
        """
        return x @ self.weights


def to_networkx(A: torch.Tensor, threshold_pct: float = 80.0) -> nx.Graph:
    """
    Convert a padded HR adjacency tensor to a weighted NetworkX graph,
    stripping the zero-padded border (26 pixels each side: 320 → 268).
    Entries below the threshold_pct-th percentile of off-diagonal values
    are zeroed out to keep the graph sparse enough for topology computation.

    Params:
        A: HR adjacency matrix of shape (hr_dim, hr_dim), e.g. (320, 320)
        threshold_pct: Percentile threshold for sparsifying the adjacency matrix
    Returns:
        Weighted undirected NetworkX graph on the 268-node core
    """
    A_core = A[26:294, 26:294].detach().cpu().numpy()
    mask = ~np.eye(A_core.shape[0], dtype=bool)
    threshold = np.percentile(A_core[mask], threshold_pct)
    A_sparse = np.where(A_core >= threshold, A_core, 0.0)
    return nx.from_numpy_array(A_sparse)


def compute_topo_features(G: nx.Graph) -> tuple[float, float, float]:
    """
    Compute a compact topological fingerprint (SWI, GE, Q) for a graph

    Metrics:
        SWI: Small-world index = (C / C_rand) / (L / L_rand)
        GE:  Global efficiency = mean of 1/d(i,j) for i≠j
        Q:   Modularity via label propagation community detection

    Shortest paths are computed with scipy's C implementation for speed.

    Params:
        G: Weighted undirected NetworkX graph
    Returns:
        Tuple (swi, ge, q) of float topology scalars
    """
    A = nx.to_numpy_array(G)
    N = len(A)

    # Convert strengths to distances (1/weight) for weighted shortest paths.
    # Zero-weight entries are left as 0 (treated as absent by scipy).
    with np.errstate(divide='ignore', invalid='ignore'):
        A_dist = np.where(A > 0, 1.0 / A, 0.0)

    # All-pairs shortest paths via scipy (faster than NetworkX)
    dist = sp_shortest_path(A_dist, directed=False, unweighted=False)

    # Global efficiency: mean of 1/d[i,j] for i≠j, treating inf as 0
    with np.errstate(divide='ignore'):
        inv_dist = 1.0 / dist
    inv_dist[~np.isfinite(inv_dist)] = 0.0
    np.fill_diagonal(inv_dist, 0.0)
    ge = inv_dist.sum() / (N * (N - 1))

    # Small-world index: (C / C_rand) / (L / L_rand)
    finite = dist[np.isfinite(dist) & (dist > 0)]
    L = np.mean(finite) if len(finite) > 0 else 1.0
    C = nx.approximation.average_clustering(G, trials=500, seed=42)
    # Use binary degree (edge count) for the Erdős–Rényi random graph baselines,
    # not weighted strength which would inflate k_avg and break C_rand / L_rand.
    k_avg = np.mean([d for _, d in G.degree(weight=None)])
    if k_avg < 1:
        swi = 1.0
    else:
        C_rand = k_avg / N
        L_rand = np.log(N) / np.log(k_avg)
        swi = (C / C_rand) / (L / L_rand) if (C_rand > 0 and L_rand > 0 and L > 0) else 1.0

    # Modularity Q via label propagation
    G_simple = nx.Graph(G)
    G_simple.remove_edges_from(nx.selfloop_edges(G_simple))
    if G_simple.number_of_edges() == 0:
        q = 0.0
    else:
        communities = nx.community.label_propagation_communities(G_simple)
        q = nx.community.modularity(G_simple, communities)

    return swi, ge, q


class StandardDiscriminator(torch.nn.Module):
    """
    Standard discriminator operating row-by-row on the hr_dim×hr_dim matrix.
    Used in 'community_only' and 'baseline' variants (matches AGSRNet behaviour)

    Params:
        args: Configuration object containing model hyperparameters
    """
    def __init__(self, args) -> None:
        super().__init__()

        hr_dim = args.hr_dim

        self.dense_1 = Dense(hr_dim, hr_dim, args)
        self.relu_1 = torch.nn.ReLU(inplace=False)

        self.dense_2 = Dense(hr_dim, hr_dim, args)
        self.relu_2 = torch.nn.ReLU(inplace=False)

        self.dense_3 = Dense(hr_dim, 1, args)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the standard discriminator

        Params:
            inputs: Input adjacency matrix of shape (hr_dim, hr_dim)
        Returns:
            Real/fake probability per row, shape (hr_dim, 1)
        """
        dc_den1 = self.relu_1(self.dense_1(inputs))
        dc_den2 = self.relu_2(self.dense_2(dc_den1))

        output = self.dense_3(dc_den2)
        output = self.sigmoid(output)

        return torch.abs(output)


class TopologyAwareDiscriminator(torch.nn.Module):
    """
    Topology-aware discriminator [Novel contribution 2]

    Augments the standard discriminator with a compact topological fingerprint
    τ(A) = [SWI, GE, Q] computed from the predicted HR adjacency matrix:

        D(A) = Sigmoid(MLP(concat([flatten(A_hr), τ(A)])))

    The three topology scalars capture small-world structure (SWI),
    global integration efficiency (GE), and community segregation (Q).
    Used in 'full' and 'topology_only' variants.

    Params:
        args: Configuration object containing model hyperparameters
    """
    def __init__(self, args) -> None:
        super().__init__()

        hr_dim = args.hr_dim
        self.threshold_pct: float = getattr(args, 'threshold_pct', 80.0)
        self.topo_scale: float = getattr(args, 'topo_scale', 100.0)

        self.dense_1 = Dense(hr_dim * hr_dim + 3, hr_dim, args)
        # Boost initial weights for the 3 topology input rows so those neurons
        # start more sensitive to topology features than the default std_dense init
        topo_init_scale: float = getattr(args, 'topo_init_scale', 10.0)
        with torch.no_grad():
            self.dense_1.weights[-3:].mul_(topo_init_scale)

        self.relu_1 = torch.nn.ReLU(inplace=False)

        self.dense_2 = Dense(hr_dim, hr_dim, args)
        self.relu_2 = torch.nn.ReLU(inplace=False)

        self.dense_3 = Dense(hr_dim, 1, args)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(
            self,
            inputs: torch.Tensor,
            topo: torch.Tensor | None = None
        ) -> torch.Tensor:
        """
        Forward pass of the topology-aware discriminator

        Params:
            inputs: Input adjacency matrix of shape (hr_dim, hr_dim)
            topo: Pre-computed topology features [SWI, GE, Q] of shape (3,).
                If None, computed fresh from inputs via to_networkx /
                compute_topo_features (wrapped in torch.no_grad)
        Returns:
            Real/fake probability scalar, shape (1, 1)
        """
        if topo is None:
            with torch.no_grad():
                G = to_networkx(inputs, threshold_pct=self.threshold_pct)
                swi, ge, q = compute_topo_features(G)
                topo = torch.tensor([swi, ge, q], dtype=torch.float32)

        x = torch.cat(
            [inputs.flatten(), topo.to(inputs.device) * self.topo_scale]
        ).unsqueeze(0)  # (1, hr_dim² + 3)

        dc_den1 = self.relu_1(self.dense_1(x))
        dc_den2 = self.relu_2(self.dense_2(dc_den1))

        output = self.dense_3(dc_den2)
        output = self.sigmoid(output)

        return torch.abs(output)  # (1, 1)


def gaussian_noise_layer(input_layer: torch.Tensor, args) -> torch.Tensor:
    """
    Add Gaussian noise to an adjacency matrix and enforce symmetry

    Params:
        input_layer: Input adjacency matrix
        args: Configuration object containing noise parameters
    Returns:
        Noisy symmetric adjacency matrix with unit diagonal
    """
    noise = torch.empty_like(input_layer).normal_(
        mean=args.mean_gaussian, std=args.std_gaussian
    )
    z = torch.abs(input_layer + noise)

    z = (z + z.T) / 2
    z.fill_diagonal_(1)

    return z
