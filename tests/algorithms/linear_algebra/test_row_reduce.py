import functools

import numpy as np
import numpy.typing as npt
import pytest

from algorithms.linear_algebra.row_reduce import row_reduce_mod_2

A = functools.partial(np.array, dtype=np.int32)

ROW_REDUCE_MOD_2_CASES = (
    # Square cases
    (A([[1, 1], [1, 1]]), A([[1, 1], [0, 0]]), (0,)),
    (A([[1, 1], [1, 0]]), A([[1, 0], [0, 1]]), (0, 1)),
    # Rectangular cases
    (A([[1], [1]]), A([[1], [0]]), (0,)),
    (A([[1, 1, 1], [1, 0, 1]]), A([[1, 0, 1], [0, 1, 0]]), (0, 1)),
)


@pytest.mark.parametrize("matrix, result, pivot_columns", ROW_REDUCE_MOD_2_CASES)
def test_row_reduce_mod_2(
    matrix: npt.NDArray[np.int32],
    result: npt.NDArray[np.int32],
    pivot_columns: tuple[int, ...],
) -> None:
    computed_matrix, computed_pivot_columns = row_reduce_mod_2(matrix.copy())
    assert (computed_matrix == result).all()
    assert computed_pivot_columns == pivot_columns
