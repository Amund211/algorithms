import pytest
import sympy  # type: ignore

from algorithms.number_theory.primes import (
    FactorizationError,
    factor_into,
    sieve_of_eratosthenes,
)
from tests.algorithms.constants import SMALL_PRIMES

PRIME_SIEVE_CASES = tuple(range(1, 100))


@pytest.mark.parametrize("n", PRIME_SIEVE_CASES)
def test_sieve_of_eratosthenes(n: int) -> None:
    assert sieve_of_eratosthenes(n) == tuple(sympy.sieve.primerange(n + 1))


def test_sieve_of_eratosthenes_inclusive() -> None:
    assert 13 in sieve_of_eratosthenes(13)


FACTOR_INTO_CASES = (
    (1, (), ()),
    (1, SMALL_PRIMES[:3], (0, 0, 0)),
    (2, SMALL_PRIMES[:3], (1, 0, 0)),
    (5, SMALL_PRIMES[:3], (0, 0, 1)),
    (10, SMALL_PRIMES[:3], (1, 0, 1)),
    (50, SMALL_PRIMES[:3], (1, 0, 2)),
    (150, SMALL_PRIMES[:3], (1, 1, 2)),
    (6, SMALL_PRIMES[:4], (1, 1, 0, 0)),
    # Non-primes
    (12, (3, 4), (1, 1)),
    # Missing primes
    (22, (2, 11), (1, 1)),
)


@pytest.mark.parametrize("r, primes, result", FACTOR_INTO_CASES)
def test_factor_into(r: int, primes: tuple[int, ...], result: tuple[int, ...]) -> None:
    assert factor_into(r, primes) == result


FACTOR_INTO_RAISE_CASES = (
    (2, ()),
    (7, (2, 3, 5)),
    (14, (2, 3, 5)),
    (62, (2, 3, 5, 62)),
    # Missing 3
    (15, (2, 5)),
)


@pytest.mark.parametrize("r, primes", FACTOR_INTO_RAISE_CASES)
def test_factor_into_raises(r: int, primes: tuple[int, ...]) -> None:
    with pytest.raises(FactorizationError):
        factor_into(r, primes)
