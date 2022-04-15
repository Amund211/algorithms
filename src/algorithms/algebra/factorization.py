import math
import random
from typing import Iterable, Iterator, Optional, Protocol

import numpy as np
import numpy.linalg
import numpy.typing as npt

from algorithms.linear_algebra.row_reduce import kernel_vectors_mod_2
from algorithms.number_theory.modular_arithmetic import product_mod_n, tonelli_shanks
from algorithms.number_theory.primes import FactorizationError, factor_into


class BadRootsError(ValueError):
    ...


class SuggestR(Iterable[tuple[int, int]], Protocol):
    def increase_search_space(self) -> None:
        ...


def index_calculus(n: int, primes: tuple[int, ...]) -> tuple[int, int]:
    """Compute the factorization of n=pq using index calculus"""

    class RandomSample:
        def __init__(self, n: int) -> None:
            self.n = n

        def increase_search_space(self) -> None:
            raise ValueError("Can't increase seach space of random sample")

        def __iter__(self) -> Iterator[tuple[int, int]]:
            return self

        def __next__(self) -> tuple[int, int]:
            r = random.randrange(2, self.n)
            return r, pow(r, 2, self.n)

    return dixon_factorization(n, RandomSample(n), primes)


def quadratic_sieve(n: int, primes: tuple[int, ...]) -> tuple[int, int]:
    """Compute the factorization of n=pq using a quadratic sieve"""

    # TODO: compute smoothness bound and sieve primes

    class QuadraticSieveSample:
        def __init__(self, n: int, width: int, primes: tuple[int, ...]) -> None:
            self.n = n
            assert n < 2**63
            self.primes = primes
            self.width = width
            self.found_rs: Optional[tuple[int, ...]] = None
            self.shuffled_rs: Optional[Iterator[int]] = None

        def restart(self) -> None:
            left = math.ceil(math.sqrt(n))
            right = min(left + self.width, n)  # non-inclusive

            assert right > left

            # The plan
            # Compute x^2 - n (mod p) for each x in [left, right]
            # Solve x^2 = n (mod p) for each p
            # Divide through by p for each (a + kp) and (b + kp)
            # The indicies with 1 remaining are square roots of a smooth number

            remainder: npt.NDArray[np.int64] = (
                np.arange(left, right, dtype=np.int64) ** 2 - n
            )
            for p in self.primes:
                if p == 2:
                    base_root = 1
                else:
                    base_root = tonelli_shanks(n, p)

                for root in (base_root, -base_root) if p != 2 else (base_root,):
                    remainder[root - left - p * ((root - left) // p) :: p] //= p

            self.found_rs = tuple(map(int, np.where(remainder == 1)[0] + left))
            print(f"Found {len(self.found_rs)} rs")

        def increase_search_space(self) -> None:
            if (
                self.width > self.n
                or (math.ceil(math.sqrt(self.n)) + self.width) ** 2 - n > 2**63
            ):
                raise ValueError("Search space is already at max")

            self.width *= 2
            print(f"Increasing search space to {self.width}")
            self.found_rs = None
            self.shuffled_rs = None

        def __iter__(self) -> Iterator[tuple[int, int]]:
            if self.found_rs is None:
                self.restart()
                assert self.found_rs is not None

            shuffled = list(self.found_rs)
            random.shuffle(shuffled)
            self.shuffled_rs = iter(shuffled)

            return self

        def __next__(self) -> tuple[int, int]:
            if self.shuffled_rs is None:
                raise RuntimeError("Must call iter before next")

            r = next(self.shuffled_rs)
            return r, r**2 - self.n

    # Include only primes where n is a quadratic residue, so that we can solve
    # x^2 = n (mod p)
    primes = tuple(filter(lambda p: p == 2 or pow(n, (p - 1) // 2, p) == 1, primes))
    print(f"Using {len(primes)} primes")

    return dixon_factorization(
        n, QuadraticSieveSample(n, width=len(primes), primes=primes), primes
    )


def dixon_factorization(
    n: int, r_suggestions: SuggestR, primes: tuple[int, ...]
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
                r, r_squared = next(r_iterator)
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
                powers = factor_into(r_squared, primes)
            except FactorizationError:
                # r^2 doesn't factor into `primes` - try again
                continue

            r_list[r_index] = r
            powers_matrix[:, r_index] = powers

            r_index += 1

        if out_of_suggestions:
            # Restart the iterable to get more rs
            r_suggestions.increase_search_space()

            # If we don't have more than 1 r, there is no chance of factoring
            if r_index <= 1:
                continue

            # If we have at least 2 rs, even though we are not guaranteed linear
            # dependence, we may get lucky
            # We send just the rs we have found
            r_list = r_list[:r_index]
            powers_matrix = powers_matrix[:, :r_index]

        try:
            p, q = _compute_dixon_factorization(n, tuple(r_list), powers_matrix, primes)
        except BadRootsError:
            # Try again
            continue
        else:
            return p, q


def _compute_dixon_factorization(
    n: int,
    r_list: tuple[int, ...],
    powers_matrix: npt.NDArray[np.int32],
    primes: tuple[int, ...],
) -> tuple[int, int]:
    """
    Find products of r^2 that are squares of small primes, and use them to factor n

    Raise BadRootsError if none of the products give useful roots
    """
    assert powers_matrix.shape[0] == len(primes)  # L
    assert powers_matrix.shape[1] > 1  # number of relations

    # Consider every vector in the null-space
    for coefficients in kernel_vectors_mod_2(powers_matrix):
        if not any(powers_matrix @ coefficients):
            # Check that the resulting number is not 1
            continue

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

    raise BadRootsError("Bad pair of roots")


if __name__ == "__main__":
    from algorithms.number_theory.primes import sieve_of_eratosthenes

    # Close to the 64-bit limit on n
    p, q = 2028156391, 3458934959
    found_p, found_q = quadratic_sieve(p * q, sieve_of_eratosthenes(2000))

    assert (p, q) == (found_p, found_q)
    print(f"{p*q} = {found_p} * {found_q}")
