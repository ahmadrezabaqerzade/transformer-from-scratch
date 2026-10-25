import torch
from torch import nn
import math 

class SelfAttention(nn.Module):
    """
    Compute scaled dot-product self-attention for batched inputs.
    """

    def __init__(self, d_model: int):
        super().__init__()
        
        if isinstance(d_model, bool) or not isinstance(d_model, int):
            raise TypeError("d_model must be an integer.")

        if d_model <= 0:
            raise ValueError("d_model must be a positive integer.")

        self.d_model = d_model

        self.key_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention to input tensors of shape (B, T, D).
        """
        
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (B, T, D), got {tuple(x.shape)}.")

        if x.size(-1) != self.d_model:
            raise ValueError(f"Expected input feature dimension {self.d_model}, "
                             f"got {x.size(-1)}.")

        key = self.key_proj(x)
        query = self.query_proj(x)
        value = self.value_proj(x)

        scores = query @ key.transpose(-2, -1)
        scores = scores/math.sqrt(key.size(-1))
        
        weights = torch.softmax(scores, dim=-1)
        output = weights @ value
        return output
