import pytest

from algorithms.algebra.number_field_sieve import nfs

FACTORIZATION_CASES: tuple[tuple[dict[str, int], int, int], ...] = (
    ## Returns early bc of phi(I)
    ({"r": 2, "e": 23, "s": 1, "prime_bound": 47, "sieve_bound": 2}, 47, 178481),
    ({"r": 2, "e": 15, "s": 1, "prime_bound": 7, "sieve_bound": 2}, 7, 31 * 151),
    ({"r": 2, "e": 11, "s": 1, "prime_bound": 23, "sieve_bound": 2}, 23, 89),
    ({"r": 2, "e": 9, "s": 1, "prime_bound": 7, "sieve_bound": 2}, 7, 73),
    ## Actual factorizations
    # ~1 secs
    (
        {"r": 2, "e": 29, "s": 1, "prime_bound": 100, "sieve_bound": 100},
        2089,
        233 * 1103,
    ),
    (
        {"r": 2, "e": 41, "s": 1, "prime_bound": 150, "sieve_bound": 500},
        13367,
        164511353,
    ),
    (
        {"r": 2, "e": 47, "s": 1, "prime_bound": 200, "sieve_bound": 500},
        2351 * 4513,
        13264529,
    ),
    # ~3 secs
    (
        {"r": 2, "e": 59, "s": 1, "prime_bound": 400, "sieve_bound": 600},
        179951,
        3203431780337,
    ),
)


@pytest.mark.parametrize("kwargs, p, q", FACTORIZATION_CASES)
def test_number_field_sieve(kwargs: dict[str, int], p: int, q: int) -> None:
    assert nfs(**kwargs) == (p, q)
