from torch import Tensor, nn

from .feed_forward import FeedForward
from .layer_norm import LayerNorm
from .multi_head_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model=d_model, nhead=nhead, dropout=dropout
        )

        self.dropout1 = nn.Dropout(p=dropout)

        self.norm1 = LayerNorm(d_model=d_model, eps=layer_norm_eps)

        self.cross_attn = MultiHeadAttention(
            d_model=d_model, nhead=nhead, dropout=dropout
        )

        self.dropout2 = nn.Dropout(p=dropout)

        self.norm2 = LayerNorm(d_model=d_model, eps=layer_norm_eps)

        self.feed_forward = FeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation
        )

        self.dropout3 = nn.Dropout(p=dropout)

        self.norm3 = LayerNorm(d_model=d_model, eps=layer_norm_eps)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:

        self_attn_output = self.self_attn(
            tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )

        x = self.norm1(tgt + self.dropout1(self_attn_output))

        cross_attn_output = self.cross_attn(
            query=x,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )

        x = self.norm2(x + self.dropout2(cross_attn_output))

        ff_output = self.feed_forward(x)

        output = self.norm3(x + self.dropout3(ff_output))

        return output
