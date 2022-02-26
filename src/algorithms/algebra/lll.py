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
    """
    Compute the non-normalized Gram Schmidt basis

    https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Algorithm
    """
    if not basis:
        return (), [[]]

    dimension = len(basis)
    assert all(len(vector) == dimension for vector in basis)

    gram_basis: list[Vector] = [basis[0]]
    Q = [[0.0] * len(basis) for _ in range(len(basis))]
    for i in range(1, dimension):
        gram_basis.append(basis[i])
        for j in range(0, i):
            Q[i][j] = (basis[i] @ gram_basis[j]) / (gram_basis[j] @ gram_basis[j])
            gram_basis[i] -= gram_basis[j].multiply(Q[i][j])

    return tuple(gram_basis), Q


def lll(basis: tuple[Vector, ...], delta: float = 3 / 4) -> tuple[Vector, ...]:
    """
    Compute the LLL reduced basis

    https://en.wikipedia.org/wiki/Lenstra%E2%80%93Lenstra%E2%80%93Lov%C3%A1sz_lattice_basis_reduction_algorithm#LLL_algorithm_pseudocode  # noqa: E501
    """
    n = len(basis)
    k = 1

    reduced = list(basis)

    gram_basis, Q = gram_schmidt(tuple(reduced))

    while k < n:
        for j in range(k - 1, -1, -1):
            if abs(Q[k][j]) > 1 / 2:
                reduced[k] -= reduced[j].multiply(round(Q[k][j]))

                # TODO: recompute only as necessary
                gram_basis, Q = gram_schmidt(tuple(reduced))

        if gram_basis[k] @ gram_basis[k] >= (delta - Q[k][k - 1] ** 2) * (
            gram_basis[k - 1] @ gram_basis[k - 1]
        ):
            k += 1
        else:
            # Swap
            reduced[k], reduced[k - 1] = reduced[k - 1], reduced[k]

            # TODO: recompute only as necessary
            gram_basis, Q = gram_schmidt(tuple(reduced))

            k = max(k - 1, 1)

    return tuple(reduced)
