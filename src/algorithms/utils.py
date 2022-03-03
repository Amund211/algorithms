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
