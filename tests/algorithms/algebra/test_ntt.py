import itertools
from typing import Iterable

import pytest
from sympy.ntheory.residue_ntheory import is_primitive_root  # type: ignore
from sympy.ntheory.residue_ntheory import primitive_root as sympy_primitive_root

from algorithms.algebra.ntt import (
    is_coprime,
    is_prime,
    ntt,
    ntt_multiply_polynomials,
    ntt_r,
    primitive_k_th_roots_of_unity,
    primitive_root,
    primitive_roots,
)

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


NTT_CASES = (
    ((1, 2), 257, 2**4, (33, -31 % 257)),
    ((3, 4), 257, 2**4, (67, -61 % 257)),
    ((-5 % 257, 10), 257, 2**4, (155, -165 % 257)),
    ((1, 2, 3, 4), 257, 2**2, (56, 42, 97, 66)),
    ((5, 6, 7, 8), 257, 2**2, (139, 95, 52, 248)),
)


@pytest.mark.parametrize("seq, p, omega, result", NTT_CASES)
def test_ntt(seq: tuple[int], p: int, omega: int, result: tuple[int]) -> None:
    assert ntt(seq=seq, p=p, omega=omega) == result


@pytest.mark.parametrize("seq, p, omega, result", NTT_CASES)
def test_ntt_r(seq: tuple[int], p: int, omega: int, result: tuple[int]) -> None:
    assert ntt_r(seq=seq, p=p, omega=omega) == result


def test_ntt_multiply_polynomials() -> None:
    assert ntt_multiply_polynomials((1, 2), (3, 4), 257, 2**4) == (155, -165 % 257)
