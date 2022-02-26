from __future__ import annotations


class Vector(tuple[float, ...]):
    def __matmul__(self, other: tuple[float, ...]) -> float:
        """Euclidean inner product"""
        return sum(a * b for a, b in zip(self, other, strict=True))

    def __sub__(self, other: tuple[float, ...]) -> Vector:
        """Vector difference"""
        return Vector(a - b for a, b in zip(self, other, strict=True))

    def multiply(self, scale: float) -> Vector:
        """Multiply the vector by a scalar"""
        return Vector(scale * a for a in self)


def gram_schmidt(
    basis: tuple[Vector, ...]
) -> tuple[tuple[Vector, ...], list[list[float]]]:
    if not basis:
        return (), [[]]

    dimension = len(basis)
    assert all(len(vector) == dimension for vector in basis)

    new_basis: list[Vector] = [basis[0]]
    Q = [[0.0] * len(basis) for _ in range(len(basis))]
    for i in range(1, dimension):
        new_basis.append(basis[i])
        for j in range(0, i):
            Q[i][j] = (new_basis[j] @ new_basis[i]) / (new_basis[j] @ new_basis[j])
            new_basis[i] -= new_basis[j].multiply(Q[i][j])

    return tuple(new_basis), Q
