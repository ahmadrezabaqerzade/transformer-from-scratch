import copy

import torch
from torch import nn

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.positional_encoding(x)

        for layer in self.encoder:
            x = layer(x)

        return x
