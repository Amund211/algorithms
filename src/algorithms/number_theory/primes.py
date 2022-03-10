import numpy as np
import numpy.typing as npt


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
