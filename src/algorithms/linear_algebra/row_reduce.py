import numpy as np
import numpy.typing as npt


def row_reduce_mod_2(
    matrix: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int32], tuple[int, ...]]:
    """Reduce the given matrix over Z_2 to echelon form out of place"""
    assert len(matrix.shape) == 2
    pivot_columns: list[int] = []

    # Don't mutate the input
    matrix = matrix.copy()

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
