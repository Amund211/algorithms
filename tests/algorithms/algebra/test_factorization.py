import pytest

from algorithms.algebra.factorization import index_calculus, quadratic_sieve
from algorithms.number_theory.primes import sieve_of_eratosthenes
from tests.algorithms.constants import SMALL_PRIMES

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


def test_quadratic_sieve() -> None:
    # Only 3 numbers smooth wrt factor base (2, 17, 23, 29) (primes where n is a^2)
    p, q = 103, 149
    assert quadratic_sieve(p * q, sieve_of_eratosthenes(30)) == (p, q)

    p, q = 7907, 7919
    assert quadratic_sieve(p * q, sieve_of_eratosthenes(1000)) == (p, q)

    p, q = 319993, 331999
    assert quadratic_sieve(p * q, sieve_of_eratosthenes(1000)) == (p, q)

    p, q = 39916801, 479001599
    assert quadratic_sieve(p * q, sieve_of_eratosthenes(2000)) == (p, q)

    # Close to the 64-bit limit on n
    p, q = 2028156391, 3458934959
    assert quadratic_sieve(p * q, sieve_of_eratosthenes(2000)) == (p, q)
