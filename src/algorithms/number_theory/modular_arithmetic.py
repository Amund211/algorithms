import functools
import itertools
from typing import Iterable, Iterator

from algorithms.number_theory.primes import is_coprime, is_prime


def primitive_roots(n: int) -> Iterator[int]:
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


def primitive_k_th_roots_of_unity(k: int, n: int) -> Iterator[int]:
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


def product_mod_n(iterable: Iterable[int], n: int, *, initial: int = 1) -> int:
    """Compute the product of the elements of the iterable mod n"""
    return functools.reduce(lambda a, b: (a * b) % n, iterable, initial % n)


def tonelli_shanks(n: int, p: int) -> int:
    """
    Find a square root of n mod p

    https://en.wikipedia.org/wiki/Tonelli%E2%80%93Shanks_algorithm#The_algorithm
    """

    assert n > 0
    assert p != 2

    Q = p - 1
    for S in itertools.count():
        if Q % 2 != 0:
            break
        Q //= 2

    for z in range(p):
        if pow(z, (p - 1) // 2, p) == p - 1:
            break
    else:
        raise ValueError(f"Found no non-quadratic residues in Z_{p}")

    M = S
    c = pow(z, Q, p)
    t = pow(n, Q, p)
    R = pow(n, (Q + 1) // 2, p)

    while True:
        if t == 0:
            return 0
        elif t == 1:
            return R

        t_squared = t
        for i in itertools.count():
            if t_squared == 1:
                break
            t_squared = pow(t_squared, 2, p)

        b = pow(c, 2 ** (M - i - 1), p)

        M = i
        c = pow(b, 2, p)
        t = (t * c) % p
        R = (R * b) % p
