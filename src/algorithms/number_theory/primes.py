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
