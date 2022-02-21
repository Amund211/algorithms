import numpy as np
import numpy.linalg

basis = [
    np.array([1, 10]),
    np.array([1, 1]),
]


def make_lattice_point(basis, coordinate):
    return sum(c * b for c, b in zip(coordinate, basis))


def reduce(basis, norm=2):
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
