import pytest


@pytest.fixture
def sample_shapes() -> list[tuple[int, ...]]:
    return [
        (2, 32, 32),
        (1, 32, 32),
        (4, 16, 32),
        (2, 64, 64),
    ]
