import math
from typing import Iterable

# print(_number_theoretic_transform([1, 2, 3, 4, 5, 0, 0, 0], 3*2**8 + 1))

"""
https://conferences.matheo.si/event/16/contribution/23/material/paper/0.pdf
https://math.stackexchange.com/questions/1926714/ntt-for-fast-multiplication-of-polynomials-over-a-finite-field-and-the-connecti
https://eprint.iacr.org/2021/563.pdf
https://math.stackexchange.com/questions/1437624/number-theoretic-transform-ntt-example-not-working-out/1439150
"""

def is_prime(n: int) -> bool:
    return all(n % i != 0 for i in range(2, math.ceil(math.sqrt(n)) + 1))


def is_coprime(a: int, b: int) -> bool:
    return math.gcd(a, b) == 1


def primitive_roots(n: int) -> Iterable[int]:
    """Dumb n^3 brute force for the primitive roots modulo n"""
    assert n > 0

    if n < 2:
        yield 1
        return

    for root in range(1, n):
        for target in range(1, n):
            if not is_coprime(target, n):
                continue
            if not any(target == pow(root, i, n) for i in range(1, n)):
                break
        else:
            yield root


def primitive_root(n: int) -> int | None:
    """Return the smallest primitive root modulo n or None"""
    try:
        return next(primitive_roots(n))
    except StopIteration:
        return None


def primitive_k_th_roots_of_unity(k: int, n: int) -> Iterable[int]:
    """Return a primitive k-th root of unity modulo n"""
    if not 1 < k < n:
        raise ValueError("k must be in [2, n-1]")

    if (n - 1) % k != 0:
        raise ValueError("k must divide n - 1")

    if not is_prime(n):
        # Only consider prime n, so that lambda(n) is n - 1
        raise ValueError("n must be prime")

    return (pow(root, (n - 1) // k, n) for root in primitive_roots(n))


def primitive_k_th_root_of_unity(k: int, n: int) -> int:
    """Return a primitive k-th root of unity modulo n"""
    try:
        return next(primitive_k_th_roots_of_unity(k, n))
    except StopIteration:
        raise ValueError(f"Found no primitive root modulo {n}")
