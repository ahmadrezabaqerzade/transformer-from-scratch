import pytest
import torch

from transformer_from_scratch.multi_head_attention import MultiHeadAttention


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, nhead",
    [
        pytest.param(2, 32, 128, 4, id="batch-2-seq-32"),
        pytest.param(1, 16, 64, 8, id="batch-1-seq-16"),
        pytest.param(4, 8, 32, 1, id="single-head"),
        pytest.param(2, 64, 64, 8, id="batch-2-seq-64"),
    ],
)
def test_output_shape(
    batch_size: int,
    seq_len: int,
    d_model: int,
    nhead: int,
) -> None:
    x = torch.rand(batch_size, seq_len, d_model)
    mha = MultiHeadAttention(d_model=d_model, nhead=nhead)

    output = mha(x)

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
    mha = MultiHeadAttention(d_model=32, nhead=1)

    with pytest.raises(ValueError):
        mha(x)


@pytest.mark.parametrize(
    "d_model",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
    ],
)
def test_invalid_d_model_value(d_model: int) -> None:
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=d_model, nhead=1)


@pytest.mark.parametrize(
    "d_model",
    [
        pytest.param(32.0, id="float"),
        pytest.param(True, id="boolean"),
        pytest.param("d_model", id="string"),
    ],
)
def test_invalid_d_model_type(d_model) -> None:
    with pytest.raises(TypeError):
        MultiHeadAttention(d_model=d_model, nhead=1)


@pytest.mark.parametrize(
    "nhead",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
    ],
)
def test_invalid_nhead_value(nhead: int) -> None:
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=32, nhead=nhead)


@pytest.mark.parametrize(
    "nhead",
    [
        pytest.param(4.0, id="float"),
        pytest.param(True, id="boolean"),
        pytest.param("nhead", id="string"),
    ],
)
def test_invalid_nhead_type(nhead) -> None:
    with pytest.raises(TypeError):
        MultiHeadAttention(d_model=32, nhead=nhead)


@pytest.mark.parametrize(
    "d_model, nhead",
    [
        pytest.param(4, 8, id="nhead-exceeds-d-model"),
        pytest.param(8, 3, id="not-divisible"),
    ],
)
def test_invalid_head_configuration(d_model: int, nhead: int) -> None:
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=d_model, nhead=nhead)



@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, nhead",
    [
        pytest.param(2, 32, 128, 4, id="batch-2-seq-32"),
        pytest.param(1, 16, 64, 8, id="batch-1-seq-16"),
        pytest.param(4, 8, 32, 1, id="single-head"),
        pytest.param(2, 64, 64, 8, id="batch-2-seq-64"),
    ],
)
def test_output_correctness(
    batch_size: int,
    seq_len: int,
    d_model: int,
    nhead: int,
) -> None:
    x = torch.rand(batch_size, seq_len, d_model)
    mha = MultiHeadAttention(d_model=d_model, nhead=nhead)

    output = mha(x)

    assert isinstance(output, torch.Tensor)
    assert torch.isfinite(output).all()


def test_gradient_flow() -> None:
    x = torch.randn(2, 8, 16, requires_grad=True)
    mha = MultiHeadAttention(d_model=16, nhead=4)

    output = mha(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for parameter in mha.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


