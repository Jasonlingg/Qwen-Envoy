import pytest

from src.product.arxiv_ids import validate_arxiv_id


@pytest.mark.parametrize("value", ["0704.0001", "2503.09516", "2503.09516v1", "2610.12345v12"])
def test_modern_ids(value):
    assert validate_arxiv_id(value) == value


@pytest.mark.parametrize("value", [
    "hep-th/9901001", "2500.09516", "2513.09516", "2503.09516v0", "2503.09516v01",
    "２５０３.０９５１６", "2503.09516/../x", "2503.09516?x=y", "2503.09516\n",
    "https://arxiv.org/abs/2503.09516", None, 123,
])
def test_unsupported_or_unsafe_ids(value):
    with pytest.raises(ValueError, match="modern arXiv ID"):
        validate_arxiv_id(value)
