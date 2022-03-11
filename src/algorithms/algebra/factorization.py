import functools
import math
import random
from typing import Iterable, Iterator, Optional, Protocol

import numpy as np
import numpy.linalg
import numpy.typing as npt

from algorithms.linear_algebra.row_reduce import row_reduce_mod_2
from algorithms.number_theory.primes import FactorizationError, factor_into


class SuggestR(Iterable[int], Protocol):
    def restart(self) -> None:
        ...

    def increase_search_space(self) -> None:
        ...


def product_mod_n(iterable: Iterable[int], n: int, *, initial: int = 1) -> int:
    """Compute the product of the elements of the iterable mod n"""
    return functools.reduce(lambda a, b: (a * b) % n, iterable, initial % n)


def index_calculus(n: int, primes: tuple[int, ...]) -> tuple[int, int]:
    """Compute the factorization of n=pq using index calculus"""

    class RandomSample:
        def __init__(self, n: int) -> None:
            self.n = n

        def restart(self) -> None:
            pass

        def increase_search_space(self) -> None:
            raise ValueError("Can't increase seach space of random sample")

        def __iter__(self) -> Iterator[int]:
            return self

        def __next__(self) -> int:
            return random.randrange(1, self.n)

    return smooth_factor(n, RandomSample(n), primes)


def quadratic_sieve(n: int, primes: tuple[int, ...]) -> tuple[int, int]:
    """Compute the factorization of n=pq using a quadratic sieve"""

    class QuadraticSieveSample:
        def __init__(self, n: int, width: int, primes: tuple[int, ...]) -> None:
            self.n = n
            self.primes = primes
            self.width = width
            self.found_rs: Optional[tuple[int]] = None
            self.shuffled_rs: Optional[Iterator[int]] = None

        def restart(self) -> None:
            center = math.ceil(math.sqrt(n))
            left = max(center - self.width, 1)  # inclusive
            right = max(center + self.width, n)  # non-inclusive

            assert right > left

            # Two versions:
            # 1:
            #   Compute x^2 - n (mod p) for each x in [left, right]
            #   Solve x^2 = n (mod p) for each p
            #   Divide through by p for each (a + kp) and (b + kp)
            #   Return those with value 1 remaining
            # 2:
            #   Solve x^2 = n (mod p) for each p
            #   Add log p to each (a + kp) and (b + kp)
            #   Return those with high sum of log - threshold?

            # 1
            remainder: npt.NDArray[np.int32] = np.array(right - left, dtype=np.int32)
            # 2
            indicies: npt.NDArray[np.float32] = np.array(right - left, dtype=np.float32)

        def increase_search_space(self) -> None:
            if self.width > math.sqrt(n):
                raise ValueError("Search space is already at max")

            self.width *= 2

        def __iter__(self) -> Iterator[int]:
            if self.found_rs is None:
                self.restart()
                assert self.found_rs is not None

            shuffled = list(self.found_rs)
            random.shuffle(shuffled)
            self.shuffled_rs = iter(shuffled)

            return self

        def __next__(self) -> int:
            if self.shuffled_rs is None:
                raise RuntimeError("Must call iter before next")

            return next(self.shuffled_rs)

    # Include only primes where n is a quadratic residue, so that we can solve
    # x^2 = n (mod p)
    primes = tuple(filter(lambda p: p == 2 or pow(n, (p - 1) // 2, p) == 1, primes))

    return smooth_factor(
        n, QuadraticSieveSample(n, width=len(primes), primes=primes), primes
    )


def smooth_factor(
    n: int, r_suggestions: Iterable[int], primes: tuple[int, ...]
) -> tuple[int, int]:
    """Compute the factorization of n=pq using smooth numbers"""

    L = len(primes)

    # Sanity checks
    assert L > 0
    assert n > L
    assert n > max(primes) ** 2

    while True:
        # Restart the iterable to get new rs
        r_iterator = iter(r_suggestions)

        # List of all the rs we have found that factors into `primes`
        r_list: list[int] = [-1] * (L + 1)
        # The power of `primes` in each r. One column per r.
        powers_matrix: npt.NDArray[np.int32] = np.empty((L, L + 1), dtype=np.int32)

        r_index = 0
        seen_rs: set[int] = set()

        # Set to True if we ran out of suggestions
        out_of_suggestions = False

        # Look for rs until we have L+1
        while r_index < L + 1:
            try:
                r = next(r_iterator)
            except StopIteration:
                out_of_suggestions = True
                break

            # Only consider unique rs
            if r in seen_rs:
                if len(seen_rs) > n / 2:
                    raise RuntimeError(
                        f"Exhausted half of all possible numbers mod {n=}"
                    )
                continue

            seen_rs.add(r)

            try:
                powers = factor_into(pow(r, 2, n), primes)
            except FactorizationError:
                # r^2 doesn't factor into `primes` - try again
                continue

            r_list[r_index] = r
            powers_matrix[:, r_index] = powers

            r_index += 1

        # Restart the iterable to get more rs
        if out_of_suggestions:
            continue

        # Find a linear combination of the prime powers from our rs
        # with all even powers (square)

        # Do gaussian elimination to get row-reduced echelon form
        reduced, pivot_columns = row_reduce_mod_2(powers_matrix % 2)

        # Components that we can choose freely
        non_pivot_columns = set(range(L + 1)) - set(pivot_columns)

        # TODO: evaluate every vector in the null-space by looping over
        # itertools.product("[0, 1] * len(non_pivot)") to find the coefficients
        # Remember a check that coeff != 0_vec

        # TODO: Put stuff into functions so it is bearable to work with

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
            continue

        p1 = math.gcd(x - y, n)
        p2 = n // p1

        assert n % p1 == 0

        if p1 > p2:
            return (p2, p1)

        return (p1, p2)
