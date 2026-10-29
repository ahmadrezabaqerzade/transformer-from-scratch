import pytest
from transformer_from_scratch.encoder import Encoder
from transformer_from_scratch.encoder_layer import EncoderLayer
import torch

@pytest.mark.parametrize("batch_size, seq_len, d_model, nhead, d_ff, num_layers", 
                        [pytest.param(2, 32, 128, 4, 512, 1, id="1_layer_batch_2_seq32"),
                         pytest.param(4, 16, 64, 8, 256, 2, id="2_layer_batch_4_seq16"),
                         pytest.param(2, 32, 128, 4, 512, 3, id="3_layer_batch_2_seq32"), 
                         pytest.param(2, 16, 16, 2, 64, 4, id="4_layer_batch_2_seq16")
                         ])
def test_output_shape(batch_size: int, seq_len: int, d_model: int, nhead: int, d_ff: int, 
    num_layers: int) -> None:
    x = torch.rand(batch_size, seq_len, d_model)
    encoder_layer = EncoderLayer(d_model=d_model, nhead=nhead, d_ff=d_ff)
    encoder = Encoder(encoder_layer=encoder_layer, num_layers=num_layers)

    assert encoder(x).shape == (batch_size, seq_len, d_model)


@pytest.mark.parametrize(
    "invalid_num_layers",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(-10, id="negative-large"),
    ],
)
def test_invalid_num_layers_value(invalid_num_layers: int) -> None:
    encoder_layer = EncoderLayer(
        d_model=32,
        nhead=4,
        d_ff=128,
    )

    with pytest.raises(ValueError):
        Encoder(
            encoder_layer=encoder_layer,
            num_layers=invalid_num_layers,
        )
