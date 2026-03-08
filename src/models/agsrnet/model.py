# AGSRNet model and associated neural network components

# Third party imports
import torch
import torch.nn.functional as F

# Local imports
from .layers import GSRLayer, GraphConvolution
from .ops import GraphUnet
from .preprocessing import normalize_adj_torch


class AGSRNet(torch.nn.Module):
    """
    Adversarial Graph Super-Resolution Network (AGSRNet)
    This model reconstructs a high-resolution graph from a low-resolution
        adjacency matrix using a combination of graph U-Net, spectral
        reconstruction, and graph convolution layers

    Params:
        args: Configuration object containing model hyperparameters
    """
    def __init__(self, args) -> None:
        super().__init__()

        self.lr_dim = args.lr_dim
        self.hr_dim = args.hr_dim
        self.hidden_dim = args.hidden_dim

        self.ks = args.ks

        # Core layers
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
            lr: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of AGSRNet

        Params:
            lr: Low-resolution adjacency matrix of shape (lr_dim, lr_dim)
        Returns:
            z: Reconstructed high-resolution adjacency matrix
            net_outs: Output of the Graph U-Net
            start_gcn_outs: Initial graph convolution outputs
            outputs: Intermediate adjacency matrix produced by GSRLayer
        """
        device = lr.device

        I = torch.eye(self.lr_dim, device=device, dtype=lr.dtype)
        
        A = normalize_adj_torch(lr).to(
            device=device, dtype=torch.float32
        )

        net_outs, start_gcn_outs = self.net(A, I)
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
            x: Input tensor of shape (n_nodes, in_features)
        Returns:
            Output tensor of shape (n_nodes, out_features)
        """
        return x @ self.weights


class Discriminator(torch.nn.Module):
    """
    Discriminator network used in adversarial training
    The discriminator attempts to distinguish between real and
        generated high-resolution adjacency matrices

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
        Forward pass of the discriminator

        Params:
            inputs: Input adjacency matrix

        Returns:
            Probability that the input matrix is real
        """
        dc_den1 = self.relu_1(self.dense_1(inputs))
        dc_den2 = self.relu_2(self.dense_2(dc_den1))

        output = self.dense_3(dc_den2)
        output = self.sigmoid(output)

        return torch.abs(output)


def gaussian_noise_layer(input_layer: torch.Tensor, args) -> torch.Tensor:
    """
    Add Gaussian noise to an adjacency matrix and enforce symmetry

    Params:
        input_layer: Input adjacency matrix
        args: Configuration object containing noise parameters
    Returns:
        Noisy symmetric adjacency matrix
    """
    noise = torch.empty_like(input_layer).normal_(
        mean=args.mean_gaussian, std=args.std_gaussian
    )
    z = torch.abs(input_layer + noise)

    z = (z + z.T) / 2
    z.fill_diagonal_(1)

    return z
