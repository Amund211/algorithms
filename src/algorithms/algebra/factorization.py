import functools
import math
import random
from typing import Iterable

import numpy as np
import numpy.linalg
import numpy.typing as npt

from algorithms.linear_algebra.row_reduce import row_reduce_mod_2


class FactorizationError(ValueError):
    pass


def product_mod_n(iterable: Iterable[int], n: int, *, initial: int = 1) -> int:
    """Compute the product of the elements of the iterable mod n"""
    return functools.reduce(lambda a, b: (a * b) % n, iterable, initial % n)


def factor(r: int, primes: tuple[int, ...]) -> tuple[int, ...]:
    """
    Return the powers of `primes` in r

    Raise FactorizationError if r can't be factorized into `primes`
    """

    assert r != 0, "Cannot factor 0"

    factor = r
    powers = [0] * len(primes)

    for i, p in enumerate(primes):
        if factor == 1:
            break

        while factor % p == 0:
            powers[i] += 1
            factor //= p

    if factor != 1:
        raise FactorizationError(
            "{r} has a factor {factor} not accounted for in primes"
        )

    return tuple(powers)


def index_calculus(n: int, primes: tuple[int, ...]) -> tuple[int, int]:
    """Compute the factorization of n=pq using index calculus"""

    L = len(primes)

    # Sanity checks
    assert L > 0
    assert n > L
    assert n > max(primes) ** 2

    # List of all the rs we have found that factors into `primes`
    r_list: list[int] = [-1] * (L + 1)
    # The power of `primes` in each r. One column per r.
    powers_matrix: npt.NDArray[np.int32] = np.empty((L, L + 1), dtype=np.int32)

    r_index = 0
    seen_rs: set[int] = set()

    # Look for rs until we have L+1
    while r_index < L + 1:
        r = random.randrange(1, n)

        # Only consider unique rs
        if r in seen_rs:
            if len(seen_rs) > n / 2:
                raise RuntimeError(f"Exhausted half of all possible numbers mod {n=}")
            continue

        seen_rs.add(r)

        try:
            powers = factor(pow(r, 2, n), primes)
        except FactorizationError:
            # r^2 doesn't factor into `primes` - try again
            continue

        r_list[r_index] = r
        powers_matrix[:, r_index] = powers

        r_index += 1

    # Find a linear combination of the prime powers from our rs
    # with all even powers (square)

    # Do gaussian elimination to get row-reduced echelon form
    reduced, pivot_columns = row_reduce_mod_2(powers_matrix % 2)

    # Components that we can choose freely
    non_pivot_columns = set(range(L + 1)) - set(pivot_columns)

    # Choose an arbitrary vector from the kernel
    # Set the free variables to 1, so we're sure we don't get the 0-vector
    coefficients = tuple(
        1 if column in non_pivot_columns  # Set the free variables to 1
        # Set the pivot columns so that each row sums to 0
        else (sum(reduced[pivot_columns.index(column), :]) - 1) % 2
        for column in range(L + 1)
    )

    # The product of r_j^a_j
    x = product_mod_n(
        (
            pow(r, int(coefficient), n)
            for r, coefficient in zip(r_list, coefficients, strict=True)
        ),
        n,
    )

    # The product of l_i^t_i
    y = product_mod_n(
        (
            pow(p, int(sum(powers_matrix[prime_index, :] * coefficients)) // 2, n)
            for prime_index, p in enumerate(primes)
        ),
        n,
    )

    assert pow(x, 2, n) == pow(y, 2, n)

    if (x + y) % n == 0 or (x - y) % n == 0:
        # We hit a bad pair of roots - try again
        return index_calculus(n, primes)

    p1 = math.gcd(x - y, n)
    p2 = n // p1

    assert n % p1 == 0

    if p1 > p2:
        return (p2, p1)

    return (p1, p2)
