import math

import numpy as np
import numpy.typing as npt


def is_prime(n: int) -> bool:
    return all(n % i != 0 for i in range(2, math.ceil(math.sqrt(n)) + 1))


def is_coprime(a: int, b: int) -> bool:
    return math.gcd(a, b) == 1


def is_power_of_2(n: int) -> bool:
    """Return True if n is on the form 2^i, i>=0"""
    return n & (n - 1) == 0


def sieve_of_eratosthenes(n: int) -> tuple[int, ...]:
    """Return the primes in [1, n]"""
    assert n >= 1

    remaining_primes: npt.NDArray[np.int32] = np.ones(n, dtype=np.int32)
    remaining_primes[0] = 0  # 1 is not prime

    index = 0
    while index < n:
        if remaining_primes[index] == 0:
            index += 1
            continue

        # The number at this index is prime - eliminate all multiples of it
        p = index + 1
        remaining_primes[p**2 - 1 :: p] = 0
        index += 1

    return tuple(int(index) + 1 for index in np.where(remaining_primes)[0])


class FactorizationError(ValueError):
    pass


def factor_into(r: int, primes: tuple[int, ...]) -> tuple[int, ...]:
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
