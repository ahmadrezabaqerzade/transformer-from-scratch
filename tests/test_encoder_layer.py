import pytest
import torch

from transformer_from_scratch.encoder_layer import EncoderLayer


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, nhead, d_ff",
    [
        pytest.param(2, 32, 128, 4, 512, id="batch-2-seq-32"),
        pytest.param(1, 16, 64, 8, 256, id="batch-1-seq-16"),
        pytest.param(4, 8, 32, 1, 128, id="single-head"),
        pytest.param(2, 64, 64, 8, 256, id="batch-2-seq-64"),
    ],
)
def test_output_shape(
    batch_size: int, seq_len: int, d_model: int, nhead: int, d_ff: int
) -> None:
    x = torch.rand(batch_size, seq_len, d_model)
    encoder_layer = EncoderLayer(d_model=d_model, nhead=nhead, d_ff=d_ff)
    output = encoder_layer(x)

    assert output.shape == (batch_size, seq_len, d_model)


@pytest.mark.parametrize(
    "invalid_shape",
    [
        pytest.param((), id="zero-dimensional"),
        pytest.param((0,), id="empty-one-dimensional"),
        pytest.param((2,), id="one-dimensional"),
        pytest.param((2, 32), id="two-dimensional"),
        pytest.param((2, 32, 64, 32), id="four-dimensional"),
    ],
)
def test_invalid_shape(invalid_shape: tuple[int, ...]) -> None:
    x = torch.rand(invalid_shape)
    encoder_layer = EncoderLayer(d_model=32, nhead=2, d_ff=128)
    with pytest.raises(ValueError):
        encoder_layer(x)


def test_gradient_flow() -> None:
    x = torch.randn(2, 8, 16, requires_grad=True)
    encoder_layer = EncoderLayer(d_model=16, nhead=2, d_ff=64)
    output = encoder_layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for parameter in encoder_layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
