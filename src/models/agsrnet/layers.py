# The architectural layers for the AGSRNet model

# Third party imports
import torch
import torch.nn.functional as F


def glorot_init(input_dim: int, output_dim: int) -> torch.Tensor:
    """
    Initialize a tensor using Glorot/Xavier uniform initialization

    Params:
        input_dim: Number of input features
    output_dim: Number of output features
    Returns:
        Tensor of shape (input_dim, output_dim) initialized with
            Glorot uniform distribution
    """
    init_range = (6.0 / (input_dim + output_dim)) ** 0.5

    return torch.empty(input_dim, output_dim).uniform_(-init_range, init_range)

def symmetrize_with_identity(mat: torch.Tensor) -> torch.Tensor:
    """
    Make a matrix symmetric and set the diagonal elements to 1

    Params:
        mat: Square matrix
    Returns
        Symmetrized matrix with unit diagonal
    """
    mat = (mat + mat.T) / 2
    mat.fill_diagonal_(1)

    return mat


class GSRLayer(torch.nn.Module):
    """
    Graph Spectral Reconstruction (GSR) layer
    This layer reconstructs a higher-resolution adjacency matrix from a
        lower-resolution adjacency matrix using spectral decomposition and
        learned transformation weights
    """
    def __init__(self, hr_dim: int) -> None:
        """
        Params:
            hr_dim: Dimension of the high-resolution graph.
        """
        super().__init__()
        self.hr_dim = hr_dim
        self.weights = torch.nn.Parameter(glorot_init(hr_dim, hr_dim))

    def forward(
            self, A: torch.Tensor, X: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the GSR transformation

        Params
            A: Low-resolution adjacency matrix of shape (lr_dim, lr_dim)
            X: Feature matrix associated with the graph nodes
        Returns
            adj: Reconstructed adjacency-like matrix
            X_out: Symmetrized adjacency representation
        """
        device = X.device

        lr_dim = A.shape[0]
        # Small workaround because eigh is not implemented on MPU
        _, U_lr = torch.linalg.eigh(A.cpu(), UPLO='U')
        U_lr = U_lr.to(device)

        # U_lr = torch.abs(U_lr)
        eye_mat = torch.eye(lr_dim, device=device)
        s_d = torch.cat((eye_mat, eye_mat), dim=0)

        a = self.weights @ s_d
        b = a @ U_lr.T
        f_d = b @ X
        adj = torch.abs(f_d)
        adj.fill_diagonal_(1)

        X_out = adj @ adj.T
        X_out = symmetrize_with_identity(X_out)

        return adj, torch.abs(X_out)


class GraphConvolution(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) layer
    Implements the propagation rule described in:
        Kipf & Welling (2016)
        https://arxiv.org/abs/1609.02907

    The layer computes:
        H = activation(A @ (dropout(X) @ W))
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout: float,
            act=F.relu
        ) -> None:
        """
        Params:
            in_features: Number of input feature channels
            out_features: Number of output feature channels
            dropout: Dropout probability
            act: Activation function applied to the output
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = torch.nn.Parameter(
            torch.empty(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Apply graph convolution

        Params:
            x: Input feature matrix of shape (num_nodes, in_features)
            adj: Adjacency matrix of shape (num_nodes, num_nodes)
        Returns 
            Output feature matrix of shape (num_nodes, out_features)
        """
        x = F.dropout(x, self.dropout, self.training)
        support = x @ self.weight
        output = adj @ support

        return self.act(output)
