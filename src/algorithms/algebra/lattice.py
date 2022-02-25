import numpy as np
import numpy.linalg
import numpy.typing as npt

Coordinate = npt.NDArray[np.int64]
Basis = list[Coordinate]

basis: Basis = [
    np.array([1, 10]),
    np.array([1, 1]),
]


def make_lattice_point(basis: Basis, coordinate: Coordinate) -> Coordinate:
    # Sum has a dumb type
    return sum(c * b for c, b in zip(coordinate, basis, strict=True))  # type: ignore


def reduce(basis: Basis, norm: int = 2) -> Basis:
    assert len(basis) == 2
    assert norm == 2
    basis = sorted(basis, key=lambda vec: np.linalg.norm(vec, ord=norm), reverse=True)

    while np.linalg.norm(basis[1], ord=norm) <= np.linalg.norm(basis[0], ord=norm):
        q = round(
            np.dot(basis[0], basis[1] / np.linalg.norm(basis[1], ord=norm))
        )  # 2-ip
        basis[0], basis[1] = basis[1], basis[0] - q * basis[1]

    return basis


print(np.dot(*basis), np.dot(*reduce(basis)))
print(reduce(basis))
