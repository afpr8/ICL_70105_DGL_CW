
# Standard library imports
from typing import Any

# Third party imports
import torch
from torch.utils.data import Dataset, DataLoader

# Local imports
from src.utils.core_utils import get_device

DEVICE, PIN_MEMORY = get_device()


def predict(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int = 1
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate predictions for a dataset using a trained model

    Params:
        model: Trained PyTorch model
        dataset: Dataset returning (input, target)
        batch_size: Batch size for inference
    Returns:
        List of (prediction, ground_truth) tensor tuples
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=PIN_MEMORY
    )

    model.eval()
    preds = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            try:
                # Try normal batched inference
                output = get_prediction(model(x))
                preds.append((output.detach(), y.detach()))

            except RuntimeError:
                # Fallback: process sample-by-sample
                batch_preds = []
                for i in range(x.shape[0]):
                    xi = x[i]
                    out = get_prediction(model(xi))
                    batch_preds.append(out)

                output = torch.stack(batch_preds)
                preds.append((output.detach(), y.detach()))

    return preds


def get_prediction(output: Any) -> torch.Tensor:
    """
    Extract the primary prediction tensor from a model output
    Supported output formats:
    - ``torch.Tensor`` → returned directly
    - ``tuple`` or ``list`` → first element returned
    - object with ``prediction`` attribute → returned

    Params:
        output: Output returned by a model forward pass
    Returns:
        prediction: The primary prediction tensor
    Raises
        ValueError: If the output format is unsupported
    """
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        return output[0]

    if hasattr(output, "prediction"):
        return output.prediction

    raise ValueError("Unsupported model output type")
