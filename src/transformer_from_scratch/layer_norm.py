import math

import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        if isinstance(d_model, bool) or not isinstance(d_model, int):
            raise TypeError("d_model must be an integer.")

        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            raise TypeError("eps must be a real number.")

        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be positive and finite.")

        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, D), got {tuple(x.shape)}.")

        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Expected input feature dimension {self.d_model}, got {x.size(-1)}."
            )

        if not x.is_floating_point():
            raise TypeError("Input x must have a floating-point dtype.")

        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, correction=0)

        normalized_x = (x - mean) / torch.sqrt(variance + self.eps)

        return self.weight * normalized_x + self.bias
