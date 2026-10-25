import pytest
import torch

from transformer_from_scratch.self_attention import SelfAttention


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((2, 32, 32), id="standard_shape"),
        pytest.param((1, 32, 32), id="batch_size_one"),
        pytest.param((4, 16, 32), id="shorter_sequence"),
        pytest.param((2, 64, 64), id="larger_feature_dimension"),
    ],
)
def test_output_shape(input_shape: tuple[int, ...]) -> None:
    x = torch.rand(input_shape)
    self_attention = SelfAttention(d_model=input_shape[-1])

    output = self_attention(x)

    assert output.shape == x.shape


@pytest.mark.parametrize(
    "invalid_shape",
    [
        pytest.param((), id="zero_dimensional"),
        pytest.param((0,), id="empty_one_dimensional"),
        pytest.param((2,), id="one_dimensional"),
        pytest.param((2, 32), id="two_dimensional"),
        pytest.param((2, 32, 64, 32), id="four_dimensional"),
    ],
)
def test_invalid_shape(invalid_shape: tuple[int, ...]) -> None:
    x = torch.rand(invalid_shape)
    self_attention = SelfAttention(d_model=32)

    with pytest.raises(ValueError):
        self_attention(x)


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((2, 32, 32), id="standard_shape"),
        pytest.param((1, 32, 32), id="batch_size_one"),
        pytest.param((4, 16, 32), id="shorter_sequence"),
        pytest.param((2, 64, 64), id="larger_feature_dimension"),
    ],
)
def test_output_correctness(input_shape: tuple[int, ...]) -> None:
    x = torch.rand(input_shape)
    self_attention = SelfAttention(d_model=input_shape[-1])

    output = self_attention(x)

    assert isinstance(output, torch.Tensor)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("input_shape", "d_model"),
    [
        pytest.param((2, 32, 32), 64, id="larger_d_model"),
        pytest.param((2, 32, 64), 32, id="smaller_d_model"),
    ],
)
def test_mismatch_input_with_dmodel(
    input_shape: tuple[int, ...],
    d_model: int,
) -> None:
    x = torch.rand(input_shape)
    self_attention = SelfAttention(d_model=d_model)

    with pytest.raises(ValueError):
        self_attention(x)


def test_gradient_flow():
    x = torch.randn(2, 8, 16, requires_grad=True)
    self_attention = SelfAttention(d_model=16)

    output = self_attention(x)
    loss = output.sum()

    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for parameter in self_attention.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
