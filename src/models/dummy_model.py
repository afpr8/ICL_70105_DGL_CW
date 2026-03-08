# A super simple benchmark model to use as a baseline for evaluating our GNNs

import torch
import torch.nn.functional as F


class DummyModel(torch.nn.Module):
    """
    Baseline upsampling model.
    This model:
        1. Bilinearly upsamples input matrices to 268×268.
        2. Applies a learnable scalar scaling factor.
        3. Enforces symmetry.
        4. Applies ReLU activation.

    Input:
        Tensor of shape (B, H_in, W_in)
    Output:
        Tensor of shape (B, target_size, target_size)
    """
    def __init__(self, target_size: int = 268) -> None:
        super().__init__()
        self.target_size = target_size
        self.alpha = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Params:
            x: Input tensor of shape (B, H, W)
        Returns:
            output: Output tensor of shape (B, 268, 268)
        """
        if x.dim() != 3:
            raise ValueError("Input must have shape (B, H, W)")

        x = F.interpolate(
            x.unsqueeze(1),
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        x = self.alpha * x
        x = (x + x.transpose(1, 2)) / 2

        return torch.relu(x)
