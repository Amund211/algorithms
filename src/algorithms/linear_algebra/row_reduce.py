import numpy as np
import numpy.typing as npt


def row_reduce_mod_2(
    matrix: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int32], tuple[int, ...]]:
    """Reduce the given matrix over Z_2 to echelon form in place"""
    assert len(matrix.shape) == 2
    pivot_columns: list[int] = []

    # Index where the next source row will be moved
    free_row = 0

    for column in range(matrix.shape[1]):
        try:
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

        for other_row in range(free_row + 1, matrix.shape[0]):
            # Eliminate this column from this row
            if matrix[other_row, column] == 1:
                matrix[other_row, :] += matrix[source_row, :]
                matrix[other_row, :] %= 2

        free_row += 1

    for column in filter(lambda col: col != 0, pivot_columns):
        source_row = next(i for i in range(column, -1, -1) if matrix[i, column] == 1)

        for other_row in range(source_row):
            # Eliminate this column from this row
            if matrix[other_row, column] == 1:
                matrix[other_row, :] += matrix[source_row, :]
                matrix[other_row, :] %= 2

    return matrix, tuple(pivot_columns)
