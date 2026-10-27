import torch
from torch import nn


class FeedForward(nn.Module):
    supported_activations = ["relu", "gelu"]

    def __init__(
        self, d_model: int, d_ff: int, dropout: float = 0.0, activation: str = "relu"
    ):
        super().__init__()
        if isinstance(d_model, bool) or not isinstance(d_model, int):
            raise TypeError("d_model must be an integer.")

        if isinstance(d_ff, bool) or not isinstance(d_ff, int):
            raise TypeError("d_ff must be an integer.")

        if not isinstance(activation, str):
            raise TypeError("activation must be a string.")

        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        if d_ff <= 0:
            raise ValueError("d_ff must be positive.")

        if activation not in self.supported_activations:
            raise ValueError(f"activation must be one of {self.supported_activations}.")

        self.d_model = d_model

        activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, D), got {tuple(x.shape)}.")

        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Expected input feature dimension {self.d_model}, got {x.size(-1)}."
            )

        return self.fc(x)
