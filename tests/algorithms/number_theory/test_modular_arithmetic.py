import itertools
from typing import Iterable

import pytest
from sympy.ntheory.residue_ntheory import is_primitive_root  # type: ignore
from sympy.ntheory.residue_ntheory import primitive_root as sympy_primitive_root

from algorithms.number_theory.modular_arithmetic import (
    primitive_k_th_roots_of_unity,
    primitive_root,
    primitive_roots,
    product_mod_n,
    tonelli_shanks,
)
from algorithms.number_theory.primes import is_coprime, is_prime
from tests.algorithms.constants import SMALL_PRIMES

MAX_K = MAX_N = 11


def sympy_primitive_roots(n: int) -> Iterable[int]:
    """Generate all primitive roots mod n using sympy"""
    if n == 1:
        return [1]

    return (
        root
        for root in range(1, n)
        if is_coprime(root, n) and is_primitive_root(root, n)
    )


@pytest.mark.parametrize("n", range(1, MAX_N))
def test_primitive_roots(n: int) -> None:
    assert list(primitive_roots(n)) == list(sympy_primitive_roots(n))


@pytest.mark.parametrize("n", range(1, MAX_N))
def test_primitive_root(n: int) -> None:
    assert primitive_root(n) == sympy_primitive_root(n)


def filter_k_th_root(k: int, n: int) -> bool:
    """Filter valid inputs for k-th root of Z_n"""
    return k <= n - 1 and (n - 1) % k == 0


k_th_root_cases = filter(
    lambda t: filter_k_th_root(t[0], t[1]),
    itertools.product(range(2, MAX_K), filter(is_prime, range(1, MAX_N))),
)


# Only consider prime n, so that lambda(n) is n - 1
@pytest.mark.parametrize("k, n", k_th_root_cases)
def test_primitive_k_th_roots_of_unity(k: int, n: int) -> None:
    roots = set(primitive_k_th_roots_of_unity(k, n))
    if not roots:
        pytest.skip("Found no roots")

    for root in roots:
        assert pow(root, k, n) == 1, f"{pow(root, k, n)=} {root=}"


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
