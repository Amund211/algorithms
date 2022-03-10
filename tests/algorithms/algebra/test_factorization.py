from typing import Iterable

import pytest

from algorithms.algebra.factorization import index_calculus, product_mod_n
from tests.algorithms.constants import SMALL_PRIMES

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


FACTORIZATION_CASES = (
    # All cases on the form factor(p_i * p_i+1, [p_1, ..., p_i-1])
    *(
        (SMALL_PRIMES[i], SMALL_PRIMES[i + 1], SMALL_PRIMES[:i])
        for i in range(1, len(SMALL_PRIMES) - 1)
    ),
)


@pytest.mark.parametrize("p, q, primes", FACTORIZATION_CASES)
def test_index_calculus(p: int, q: int, primes: tuple[int, ...]) -> None:
    assert index_calculus(p * q, primes) == (p, q)
