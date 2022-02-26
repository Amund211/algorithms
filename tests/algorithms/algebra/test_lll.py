import numpy as np
import pytest

from algorithms.algebra.lll import Vector, gram_schmidt, lll

GRAM_SCHMIDT_CASES = (
    ((Vector([3, 1]), Vector([2, 2])), (Vector([3, 1]), Vector([-2 / 5, 6 / 5]))),
)


@pytest.mark.parametrize("basis, new_basis", GRAM_SCHMIDT_CASES)
def test_gram_schmidt(basis: tuple[Vector, ...], new_basis: tuple[Vector, ...]) -> None:
    for vector, new_vector in zip(gram_schmidt(basis)[0], new_basis, strict=True):
        assert np.allclose(vector, new_vector)


LLL_CASES = (
    (
        (Vector([1, 1, 1]), Vector([-1, 0, 2]), Vector([3, 5, 6])),
        3 / 4,
        (Vector([0, 1, 0]), Vector([1, 0, 1]), Vector([-1, 0, 2])),
    ),
)


@pytest.mark.parametrize("basis, delta, new_basis", LLL_CASES)
def test_lll(
    basis: tuple[Vector, ...], delta: float, new_basis: tuple[Vector, ...]
) -> None:
    assert lll(basis, delta=delta) == new_basis
