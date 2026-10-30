import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
    ) -> None:
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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        self._validate_input(query, "query")

        if (key is None) != (value is None):
            raise ValueError(
                "key and value must either both be provided or both be None."
            )

        # Self-Attention
        if key is None and value is None:
            key = query
            value = query

        # Cross-Attention
        else:
            self._validate_input(key, "key")
            self._validate_input(value, "value")

            if query.size(0) != key.size(0):
                raise ValueError("query and key must have the same batch size.")

            if key.size(0) != value.size(0):
                raise ValueError("key and value must have the same batch size.")

            if key.size(1) != value.size(1):
                raise ValueError("key and value must have the same sequence length.")

            if query.device != key.device or key.device != value.device:
                raise ValueError("query, key, and value must be on the same device.")

        # Attention mask validation
        if attn_mask is not None:
            if not isinstance(attn_mask, torch.Tensor):
                raise TypeError("attn_mask must be a torch.Tensor.")

            if attn_mask.dtype != torch.bool:
                raise TypeError("attn_mask must have dtype torch.bool.")

            expected_shape = (query.size(1), key.size(1))

            if attn_mask.shape != expected_shape:
                raise ValueError(
                    f"Expected attn_mask shape {expected_shape}, "
                    f"got {tuple(attn_mask.shape)}."
                )

            if attn_mask.device != query.device:
                raise ValueError("attn_mask must be on the same device as query.")

            if attn_mask.all(dim=-1).any():
                raise ValueError("attn_mask must not contain fully masked rows.")

        # Linear projections
        B, T, D = query.shape

        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # Split into attention heads
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # Scaled Dot-Product Attention
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.dh)

        # Apply attention mask
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        heads_output = weights @ value

        # Concatenate heads
        output = heads_output.permute(0, 2, 1, 3).reshape(B, T, D)

        # Final linear projection
        output = self.final_proj(output)

        return output

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        x = x.reshape(
            batch_size,
            seq_len,
            self.nhead,
            self.dh,
        )

        return x.permute(0, 2, 1, 3)

    def _validate_input(
        self,
        x: torch.Tensor,
        name: str,
    ) -> None:

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")

        if x.ndim != 3:
            raise ValueError(f"{name} must have shape (B, T, D), got {tuple(x.shape)}.")

        if x.size(-1) != self.d_model:
            raise ValueError(
                f"{name} must have feature dimension {self.d_model}, got {x.size(-1)}."
            )
