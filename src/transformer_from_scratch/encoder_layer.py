from torch import Tensor, nn

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
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.d_model = d_model

        self.mha = MultiHeadAttention(d_model=d_model, nhead=nhead, dropout=dropout)

        self.feed_forward = FeedForward(
            d_model=d_model, d_ff=d_ff, activation=activation, dropout=dropout
        )

        self.norm1 = LayerNorm(d_model=d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model=d_model, eps=layer_norm_eps)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:

        attention_output = self.mha(
            query=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        attention_output = self.dropout(attention_output)

        normalized_src = self.norm1(src + attention_output)

        feedforward_output = self.feed_forward(normalized_src)
        feedforward_output = self.dropout(feedforward_output)

        final_output = self.norm2(normalized_src + feedforward_output)

        return final_output
