from typing import Iterable

import pytest

from algorithms.number_theory.modular_arithmetic import product_mod_n, tonelli_shanks
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


@pytest.mark.parametrize("n", range(1, 50))
def test_tonelli_shanks(n: int) -> None:
    primes = tuple(
        filter(lambda p: p != 2 and pow(n, (p - 1) // 2, p) == 1, SMALL_PRIMES)
    )

    for p in primes:
        root = tonelli_shanks(n, p)
        assert pow(root, 2, p) == pow(-root, 2, p) == n % p
