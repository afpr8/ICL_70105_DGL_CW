
import torch

class GraphUnpool(torch.nn.Module):
    """
    Simple graph unpooling layer that restores node features to their original
        positions
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(
            self, A: torch.Tensor, X: torch.Tensor, idx: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Unpool node features

        Params:
            A: Adjacency matrix, shape [N, N]
            X: Node features, shape [k, F]
            idx: Indices of nodes selected during pooling, shape [k]
        Returns:
            Unpooled adjacency matrix (unchanged) and features of shape [N, F]
        """
        new_X = torch.zeros(
            [A.shape[0], X.shape[1]],
            device=X.device,
            dtype=X.dtype
        )
        new_X[idx] = X

        return A, new_X


class GraphPool(torch.nn.Module):
    """
    Simple top-k node pooling layer.
    """
    def __init__(self, k: float, in_dim: int) -> None:
        super().__init__()
        self.k = k
        self.proj = torch.nn.Linear(in_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, A, X):
        """
        Pool nodes based on learned scores

        Params:
            A: Adjacency matrix [N, N]
            X: Node features [N, F]

        Returns:
            new_A: Pooled adjacency matrix [k_num, k_num]
            new_X: Pooled features [k_num, F]
            idx: Indices of selected nodes [k_num]
        """
        device = X.device
        num_nodes = A.shape[0]

        # Compute scores for each node
        scores = self.sigmoid(torch.abs(self.proj(X)) / 100) # [num_nodes, 1]
        k_num = max(1, int(self.k * num_nodes))

        # Select top-k nodes
        values, idx = torch.topk(scores.squeeze(-1), k_num) # [k_num]
        values = values.unsqueeze(-1) # [k_num,1]

        # Scale features by score
        new_X = X[idx, :] * values # [k_num, feature_dim]

        # Reduce adjacency to selected nodes
        new_A = A[idx, :]
        new_A = new_A[:, idx]

        return new_A.to(device), new_X.to(device), idx.to(device)


class GCN(torch.nn.Module):
    """
    Graph convolution layer
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        """
        Params:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, out_dim)
        self.drop = torch.nn.Dropout(p=dropout)

    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Apply graph convolution

        Params:
            A: Adjacency matrix [N, N]
            X: Node features [N, F_in]
        Returns:
            Updated node features [N, F_out]
        """
        X = self.drop(X)
        X = A @ X
        X = self.proj(X)

        return X


class GraphUnet(torch.nn.Module):
    """
    Graph U-Net architecture for hierarchical graph representation
    """
    def __init__(
            self,
            ks: list[float],
            in_dim: int,
            out_dim: int,
            hidden_dim: int = 320
        ) -> None:
        """
        Params:
            ks: Pooling fractions at each level
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            hidden_dim: Hidden feature dimension for internal GCN layers
        """
        super().__init__()
        self.ks = ks
        self.l_n = len(ks)

        self.start_gcn = GCN(in_dim, hidden_dim)
        self.bottom_gcn = GCN(hidden_dim, hidden_dim)
        self.end_gcn = GCN(2 * hidden_dim, out_dim)

        self.down_gcns = torch.nn.ModuleList(
            [GCN(hidden_dim, hidden_dim) for _ in range(self.l_n)]
        )
        self.up_gcns = torch.nn.ModuleList(
            [GCN(hidden_dim, hidden_dim) for _ in range(self.l_n)]
        )
        self.pools = torch.nn.ModuleList(
            [GraphPool(ks[i], hidden_dim) for i in range(self.l_n)]
        )
        self.unpools = torch.nn.ModuleList([GraphUnpool() for _ in range(self.l_n)])

    def forward(
            self, A: torch.Tensor, X: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Graph U-Net

        Params:
            A: Input adjacency matrix [N, N]
            X: Input node features [N, F_in]
        Returns:
            X_out: Output node features [N, out_dim]
            start_gcn_outs: Node features after initial GCN [N, hidden_dim]
        """
        adj_ms = []
        indices_list = []
        down_outs = []

        X = self.start_gcn(A, X)
        start_gcn_outs = X
        org_X = X

        # Downsampling
        for i in range(self.l_n):
            X = self.down_gcns[i](A, X)
            down_outs.append(X)
            adj_ms.append(A.clone())
            A, X, idx = self.pools[i](A, X)
            indices_list.append(idx)

        X = self.bottom_gcn(A, X)

        # Upsampling
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            A, idx = adj_ms[up_idx], indices_list[up_idx]
            A, X = self.unpools[i](A, X, idx)
            X = self.up_gcns[i](A, X)
            X = X.add(down_outs[up_idx])

        X = torch.cat([X, org_X], 1)
        X = self.end_gcn(A, X)

        return X, start_gcn_outs
