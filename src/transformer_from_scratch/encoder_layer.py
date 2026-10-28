import torch
from torch import nn

from .feed_forward import FeedForward
from .layer_norm import LayerNorm
from .multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        eps: float = 1e-5,
    ):
        super().__init__()

        self.mha = MultiHeadAttention(d_model=d_model, nhead=nhead, dropout=dropout)

        self.feed_forward = FeedForward(
            d_model=d_model, d_ff=d_ff, activation=activation
        )

        self.norm1 = LayerNorm(d_model=d_model, eps=eps)
        self.norm2 = LayerNorm(d_model=d_model, eps=eps)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_output = self.mha(x)
        attention_output = self.dropout(attention_output)

        normalized_x = self.norm1(x + attention_output)

        feedforward_output = self.feed_forward(normalized_x)
        feedforward_output = self.dropout(feedforward_output)

        final_output = self.norm2(normalized_x + feedforward_output)

        return final_output
