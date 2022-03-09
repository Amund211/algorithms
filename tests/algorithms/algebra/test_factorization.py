from typing import Iterable

import pytest

from algorithms.algebra.factorization import (
    FactorizationError,
    factor,
    index_calculus,
    product_mod_n,
)

PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)

PRODUCT_CASES = (
    ((), 10, 1),
    (range(5), 10000, 0),
    (range(1, 2), 10, 1),
    (range(1, 3), 10, 2),
    (range(1, 4), 10, 6),
    (range(1, 5), 10, 4),
    (range(1, 6), 10, 0),
    (range(1, 11), 10, 0),
    (range(1, 2), 7, 1),
    (range(1, 3), 7, 2),
    (range(1, 4), 7, 6),
    (range(1, 5), 7, 3),
    (range(1, 6), 7, 1),
    (range(1, 7), 7, 6),
    (range(1, 8), 7, 0),
)


@pytest.mark.parametrize("iterable, n, result", PRODUCT_CASES)
def test_product_mod_n(iterable: Iterable[int], n: int, result: int) -> None:
    assert product_mod_n(iterable, n) == result


FACTOR_CASES = (
    (1, (), ()),
    (1, PRIMES[:3], (0, 0, 0)),
    (2, PRIMES[:3], (1, 0, 0)),
    (5, PRIMES[:3], (0, 0, 1)),
    (10, PRIMES[:3], (1, 0, 1)),
    (50, PRIMES[:3], (1, 0, 2)),
    (150, PRIMES[:3], (1, 1, 2)),
    (6, PRIMES[:4], (1, 1, 0, 0)),
    # Non-primes
    (12, (3, 4), (1, 1)),
    # Missing primes
    (22, (2, 11), (1, 1)),
)


@pytest.mark.parametrize("r, primes, result", FACTOR_CASES)
def test_factor(r: int, primes: tuple[int, ...], result: tuple[int, ...]) -> None:
    assert factor(r, primes) == result


FACTOR_RAISE_CASES = (
    (2, ()),
    (7, (2, 3, 5)),
    (14, (2, 3, 5)),
    (62, (2, 3, 5, 62)),
    # Missing 3
    (15, (2, 5)),
)


@pytest.mark.parametrize("r, primes", FACTOR_RAISE_CASES)
def test_factor_raises(r: int, primes: tuple[int, ...]) -> None:
    with pytest.raises(FactorizationError):
        factor(r, primes)


FACTORIZATION_CASES = (
    # All cases on the form factor(p_i * p_i+1, [p_1, ..., p_i-1])
    *((PRIMES[i], PRIMES[i + 1], PRIMES[:i]) for i in range(1, len(PRIMES) - 1)),
)


@pytest.mark.parametrize("p, q, primes", FACTORIZATION_CASES)
def test_index_calculus(p: int, q: int, primes: tuple[int, ...]) -> None:
    assert index_calculus(p * q, primes) == (p, q)
