import itertools
from typing import Iterable

import numpy as np
import numpy.typing as npt


def row_reduce_mod_2(
    matrix: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int32], tuple[int, ...]]:
    """Reduce the given matrix over Z_2 to echelon form out of place"""
    assert len(matrix.shape) == 2

    pivot_columns: list[int] = []

    # Don't mutate the input
    matrix = matrix.copy() % 2

    # Index where the next source row will be moved
    free_row = 0

    for column in range(matrix.shape[1]):
        try:
            # Find the first new row with non-zero entry in this column
            source_row = next(
                i for i in range(free_row, matrix.shape[0]) if matrix[i, column] == 1
            )
        except StopIteration:
            # No rows with non-zero entries in this column - continue
            continue

        # Swap the new source row to the top
        if source_row != free_row:
            matrix[[source_row, free_row], :] = matrix[[free_row, source_row], :]

        pivot_columns.append(column)

        # Eliminate this column from all other rows below
        for other_row in range(free_row + 1, matrix.shape[0]):
            if matrix[other_row, column] == 1:
                matrix[other_row, :] += matrix[free_row, :]
                matrix[other_row, :] %= 2

        free_row += 1

    # Use all the pivot columns to eliminate upwards
    for column in pivot_columns:
        # Find the last row with non-zero entry in this column
        source_row = next(
            i
            for i in reversed(range(min(column + 1, matrix.shape[0])))
            if matrix[i, column] == 1
        )

        # Eliminate this column from all other rows above
        for other_row in range(source_row):
            if matrix[other_row, column] == 1:
                matrix[other_row, :] += matrix[source_row, :]
                matrix[other_row, :] %= 2

    return matrix, tuple(pivot_columns)


def kernel_vectors_mod_2(
    matrix: npt.NDArray[np.int32],
) -> Iterable[tuple[int, ...]]:
    """Return the kernel vectors of the matrix mod 2"""
    assert len(matrix.shape) == 2

    # Don't mutate the input
    matrix = matrix.copy() % 2

    output_dim, input_dim = matrix.shape

    # Do gaussian elimination to get row-reduced echelon form
    reduced, pivot_columns = row_reduce_mod_2(matrix)

    # Components that we can choose freely
    non_pivot_columns = set(range(input_dim)) - set(pivot_columns)

    # Consider every vector in the null-space
    for non_pivot_values in itertools.product(
        *((0, 1) for _ in range(len(non_pivot_columns)))
    ):
        non_pivot_mapping = {
            index: value
            for index, value in zip(non_pivot_columns, non_pivot_values, strict=True)
        }

        # Sum of all the non-pivot columns in a row
        row_sum = {
            row: sum(
                non_pivot_mapping[column] * reduced[row, column]
                for column in non_pivot_columns
            )
            % 2
            for row in range(output_dim)
        }

        coefficients = tuple(
            # Set the free variables
            non_pivot_mapping[column] if column in non_pivot_columns
            # Set the pivot columns so that each row sums to 0
            else row_sum[pivot_columns.index(column)]
            for column in range(input_dim)
        )

        if input_dim >= output_dim:
            # We are guaranteed a solution Ax = 0 (mod 2)
            assert not any((matrix @ coefficients) % 2)
        else:
            # We are not guaranteed a solution Ax = 0 (mod 2)
            # Skip this combination if it is not in the null space mod 2
            if any((matrix @ coefficients) % 2):
                continue

        yield coefficients
