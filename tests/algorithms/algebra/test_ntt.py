import pytest

from algorithms.algebra.ntt import (
    intt_r,
    ntt,
    ntt_multiply_polynomials,
    ntt_r,
    schoolbook_multiply_polynomials,
)

NTT_CASES = (
    ((1, 2), 257, 2**4, (33, -31 % 257)),
    ((3, 4), 257, 2**4, (67, -61 % 257)),
    ((-5 % 257, 10), 257, 2**4, (155, -165 % 257)),
    ((1, 2, 3, 4), 257, 2**2, (56, 42, 97, 66)),
    ((5, 6, 7, 8), 257, 2**2, (139, 95, 52, 248)),
)


@pytest.mark.parametrize("coefficients, p, omega, transformed", NTT_CASES)
def test_ntt(
    coefficients: tuple[int, ...], p: int, omega: int, transformed: tuple[int, ...]
) -> None:
    assert ntt(seq=coefficients, p=p, omega=omega) == transformed


@pytest.mark.parametrize("coefficients, p, omega, transformed", NTT_CASES)
def test_ntt_r(
    coefficients: tuple[int, ...], p: int, omega: int, transformed: tuple[int, ...]
) -> None:
    assert ntt_r(seq=coefficients, p=p, omega=omega) == transformed


@pytest.mark.parametrize("coefficients, p, omega, transformed", NTT_CASES)
def test_intt_r(
    coefficients: tuple[int, ...], p: int, omega: int, transformed: tuple[int, ...]
) -> None:
    assert intt_r(seq=transformed, p=p, omega=omega) == coefficients


POLY_MULT_CASES = (
    ((1, 2), (3, 4), 257, 2**4, (-5 % 257, 10)),
    ((1, 2, 3, 4), (5, 6, 7, 8), 257, 2**2, (201, 221, 2, 60)),
)


@pytest.mark.parametrize("poly1, poly2, p, omega, result", POLY_MULT_CASES)
def test_ntt_multiply_polynomials(
    poly1: tuple[int, ...],
    poly2: tuple[int, ...],
    p: int,
    omega: int,
    result: tuple[int, ...],
) -> None:
    assert ntt_multiply_polynomials(poly1, poly2, p, omega) == result


@pytest.mark.parametrize("poly1, poly2, p, omega, result", POLY_MULT_CASES)
def test_schoolbook_multiply_polynomials(
    poly1: tuple[int, ...],
    poly2: tuple[int, ...],
    p: int,
    omega: int,
    result: tuple[int, ...],
) -> None:
    assert schoolbook_multiply_polynomials(poly1, poly2, p) == result
