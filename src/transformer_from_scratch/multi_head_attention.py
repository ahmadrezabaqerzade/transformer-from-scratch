import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()

        if isinstance(d_model, bool) or not isinstance(d_model, int):
            raise TypeError("d_model must be an integer.")

        if isinstance(nhead, bool) or not isinstance(nhead, int):
            raise TypeError("nhead must be an integer.")

        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        if nhead <= 0:
            raise ValueError("nhead must be positive.")

        if nhead > d_model or d_model % nhead != 0:
            raise ValueError(
                "nhead must not exceed d_model, and d_model must be divisible by nhead."
            )

        self.d_model = d_model
        self.nhead = nhead
        self.dh = d_model // nhead

        self.key_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.final_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, D), got {tuple(x.shape)}.")

        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Expected input feature dimension {self.d_model}, got {x.size(-1)}."
            )

        B, T, D = x.shape

        key = self.split_heads(self.key_proj(x), B, T)
        query = self.split_heads(self.query_proj(x), B, T)
        value = self.split_heads(self.value_proj(x), B, T)

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.dh)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        heads_output = weights @ value

        output = heads_output.permute(0, 2, 1, 3).reshape(B, T, D)
        output = self.final_proj(output)

        return output

    def split_heads(
        self, x: torch.Tensor, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        x = x.reshape(batch_size, seq_len, self.nhead, self.dh)
        return x.permute(0, 2, 1, 3)
