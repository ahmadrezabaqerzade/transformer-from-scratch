from torch import Tensor, nn

from .decoder import Decoder
from .decoder_layer import DecoderLayer
from .encoder import Encoder
from .encoder_layer import EncoderLayer


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.encoder = Encoder(
            EncoderLayer(
                d_model=d_model,
                nhead=nhead,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
            ),
            num_layers=num_encoder_layers,
        )

        self.decoder = Decoder(
            DecoderLayer(
                d_model=d_model,
                nhead=nhead,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
            ),
            num_layers=num_decoder_layers,
        )

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:

        memory = self.encoder(
            src=src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )

        return self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
