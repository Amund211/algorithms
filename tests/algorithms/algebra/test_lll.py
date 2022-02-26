import numpy as np
import pytest

from algorithms.algebra.lll import Vector, gram_schmidt

GRAM_SCHMIDT_CASES = (
    ((Vector([3, 1]), Vector([2, 2])), (Vector([3, 1]), Vector([-2 / 5, 6 / 5]))),
)


@pytest.mark.parametrize("basis, new_basis", GRAM_SCHMIDT_CASES)
def test_gram_schmidt(basis: tuple[Vector, ...], new_basis: tuple[Vector, ...]) -> None:
    for vector, new_vector in zip(gram_schmidt(basis)[0], new_basis, strict=True):
        assert np.allclose(vector, new_vector)
