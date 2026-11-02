import copy

from torch import nn, Tensor

from .encoder_layer import EncoderLayer
from .positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(
        self,
        encoder_layer: EncoderLayer,
        num_layers: int,
    ) -> None:
        super().__init__()

        if isinstance(num_layers, bool) or not isinstance(num_layers, int):
            raise TypeError("num_layers must be an integer.")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.positional_encoding = PositionalEncoding(d_model=encoder_layer.d_model)

        self.encoder = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward(self, src: Tensor, src_mask: Tensor | None = None,
                src_key_padding_mask: Tensor | None = None) -> Tensor:
        src = self.positional_encoding(src)

        for layer in self.encoder:
            src = layer(src=src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        return src
