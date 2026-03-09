# A naive GCN model and associated neural network components

# Third party imports
import torch
import torch.nn.functional as F

# Local imports
from src.models.agsrnet.layers import GraphConvolution


class NaiveGCN(torch.nn.Module):
    """
    A simple 2-layer GCN baseline for Graph Super-Resolution.
    """
    def __init__(self, args):
        super().__init__()
        self.lr_dim = args.lr_dim   # e.g., 160
        self.hr_dim = args.hr_dim   # e.g., 268
        self.hidden_dim = getattr(args, 'hidden_dim', 64)

        # Layer 1: Expand LR features to a hidden representation
        self.gc1 = GraphConvolution(
            in_features=self.lr_dim, 
            out_features=self.hidden_dim, 
            dropout=0.1, 
            act=F.relu
        )

        # Layer 2: Project hidden features to HR dimension
        # Note: In a naive setup, we often output a feature matrix 
        # that we then use to reconstruct the adjacency matrix.
        self.gc2 = GraphConvolution(
            in_features=self.hidden_dim, 
            out_features=self.hr_dim, 
            dropout=0.0, 
            act=lambda x: x # Identity for the final layer
        )

    def forward(self, lr: torch.Tensor):
        device = lr.device
        # 1. Preprocessing: GCNs need an identity matrix as initial features
        # if no other node features exist.
        features = torch.eye(self.lr_dim, device=device)
        
        # 2. Normalize Adjacency (If not already done in your dataset)
        # adj = normalize_adj_torch(lr) 
        adj = lr 

        # 3. Forward pass through GCN layers
        h = self.gc1(features, adj)
        z = self.gc2(h, adj) # Shape: (lr_dim, hr_dim)

        # 4. Reconstruct the HR Adjacency Matrix
        # A simple naive way is the inner product of the learned embeddings
        # (Self-attention style reconstruction)
        hr_out = torch.mm(z.t(), z)
        
        # 5. Post-processing: Symmetry and Diagonal
        hr_out = (hr_out + hr_out.t()) / 2
        hr_out.fill_diagonal_(1.0)

        return torch.abs(hr_out)
