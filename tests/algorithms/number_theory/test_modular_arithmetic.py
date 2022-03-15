import pytest

from algorithms.number_theory.modular_arithmetic import tonelli_shanks
from tests.algorithms.constants import SMALL_PRIMES


@pytest.mark.parametrize("n", range(1, 50))
def test_tonelli_shanks(n: int) -> None:
    primes = tuple(
        filter(lambda p: p != 2 and pow(n, (p - 1) // 2, p) == 1, SMALL_PRIMES)
    )

    for p in primes:
        root = tonelli_shanks(n, p)
        assert pow(root, 2, p) == pow(-root, 2, p) == n % p
