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
def test_self_attention_output_shape(
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
    "batch_size, query_len, key_len, d_model, nhead",
    [
        pytest.param(2, 4, 8, 16, 4, id="query-shorter"),
        pytest.param(2, 8, 4, 16, 4, id="query-longer"),
        pytest.param(1, 16, 16, 64, 8, id="equal-length"),
    ],
)
def test_cross_attention_output_shape(
    batch_size: int,
    query_len: int,
    key_len: int,
    d_model: int,
    nhead: int,
) -> None:
    query = torch.randn(batch_size, query_len, d_model)
    key = torch.randn(batch_size, key_len, d_model)
    value = torch.randn(batch_size, key_len, d_model)

    mha = MultiHeadAttention(d_model=d_model, nhead=nhead)

    output = mha(query, key, value)

    assert output.shape == (batch_size, query_len, d_model)
    assert torch.isfinite(output).all()


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


def test_attention_mask():
    attention = MultiHeadAttention(d_model=8, nhead=2)

    x = torch.randn(2, 4, 8)

    mask = torch.tensor(
        [
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )

    output = attention(x, attn_mask=mask)

    assert output.shape == (2, 4, 8)
    assert torch.isfinite(output).all()


def test_attention_mask_invalid_dtype() -> None:
    attention = MultiHeadAttention(d_model=8, nhead=2)
    x = torch.randn(2, 4, 8)
    mask = torch.ones(4, 4)

    with pytest.raises(TypeError):
        attention(x, attn_mask=mask)


def test_key_padding_mask() -> None:
    attention = MultiHeadAttention(d_model=8, nhead=2)

    x = torch.randn(2, 4, 8)

    mask = torch.tensor(
        [
            [False, False, False, False],
            [False, False, True, True],
        ],
        dtype=torch.bool,
    )

    output = attention(x, key_padding_mask=mask)

    assert output.shape == (2, 4, 8)
    assert torch.isfinite(output).all()


def test_cross_attention_key_padding_mask() -> None:
    attention = MultiHeadAttention(d_model=8, nhead=2)

    query = torch.randn(2, 3, 8)
    key = torch.randn(2, 5, 8)
    value = torch.randn(2, 5, 8)

    mask = torch.tensor(
        [
            [False, False, False, True, True],
            [False, False, True, True, True],
        ],
        dtype=torch.bool,
    )

    output = attention(
        query,
        key,
        value,
        key_padding_mask=mask,
    )

    assert output.shape == (2, 3, 8)
    assert torch.isfinite(output).all()


def test_combined_attention_masks() -> None:
    attention = MultiHeadAttention(d_model=8, nhead=2)

    x = torch.randn(2, 4, 8)

    attn_mask = torch.tensor(
        [
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ],
        dtype=torch.bool,
    )

    key_padding_mask = torch.tensor(
        [
            [False, False, False, False],
            [False, False, True, True],
        ],
        dtype=torch.bool,
    )

    output = attention(
        x,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
    )

    assert output.shape == (2, 4, 8)
    assert torch.isfinite(output).all()
