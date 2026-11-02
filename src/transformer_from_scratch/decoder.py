import copy

from torch import Tensor, nn

from .decoder_layer import DecoderLayer
from .positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(self, decoder_layer: DecoderLayer, num_layers: int):
        super().__init__()

        if isinstance(num_layers, bool) or not isinstance(num_layers, int):
            raise TypeError("num_layers must be an integer.")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.positional_encoding = PositionalEncoding(d_model=decoder_layer.d_model)

        self.decoder = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        tgt = self.positional_encoding(tgt)

        for layer in self.decoder:
            tgt = layer(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        return tgt
