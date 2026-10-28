import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        if isinstance(d_model, bool) or not isinstance(d_model, int):
            raise TypeError("d_model must be an integer.")

        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        self.d_model = d_model

        even_dims = torch.arange(d_model) // 2 * 2
        denominator = 10000 ** (even_dims / d_model)

        self.register_buffer("denominator", denominator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, D), got {tuple(x.shape)}.")

        _, T, D = x.shape

        if D != self.d_model:
            raise ValueError(
                f"Expected input feature dimension {self.d_model}, got {D}"
            )

        if not x.is_floating_point():
            raise TypeError("Input x must have a floating-point dtype.")

        position = torch.arange(
            T, device=x.device, dtype=self.denominator.dtype
        ).unsqueeze(1)

        pe = position / self.denominator

        pe[..., ::2] = torch.sin(pe[..., ::2])
        pe[..., 1::2] = torch.cos(pe[..., 1::2])

        pe = pe.to(dtype=x.dtype)

        return x + pe.unsqueeze(0)
