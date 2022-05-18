"""
https://link.springer.com/book/10.1007/BFb0091534

https://link.springer.com/content/pdf/10.1007%2FBFb0091534.pdf
"""

import cmath
import contextlib
import datetime
import functools
import itertools
import logging
import math
import operator
import time
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Iterator, Literal, cast

import numpy as np
import numpy.typing as npt
import sympy  # type: ignore

from algorithms.linear_algebra.row_reduce import kernel_vectors_mod_2
from algorithms.number_theory.primes import (
    FactorizationError,
    factor_into,
    sieve_of_eratosthenes,
)

"""
nfs: factor n
    1: relations
        K = Q[alpha]/Q
        f(x) = x^d - t
        alpha = nroot(t)
        [a, b, c, ...] - d lange
        a + b alpha + c alpha^2

        phi: Z[alpha] -> Z_n
        phi(alpha) = m

        Z[alpha] - [a, b, 0, ...]

        phi(gamma) = phi(g)^p * phi(u)^i

        (a + b alpha) = Π(curlyp^e(curlyp))

        curlyp = (p, c mod p), f(c) = 0 mod p

        (g) = curlyp

        norm(gamma) ?= p
        norm(gamma) ?= +-1

        u1 * u2 = u3
        a + b alpha = Π(g^e(g)) * Π(u^e(u))
        [u1, u1, u1, u1, u1, u1, u1, u1, u1]
        [2, 3, 4, 5, 6]
        [2, 3, 5]

        [u1, u2, u3, u4]
        [u1, u2]

        [u1, u1, u1]




        phi(g)
        phi(u)

    2: linalg
        phi(gamma)^2 = phi(g)^2p * phi(u)^2i

        e(phi(gamma)) -e(g) -e(u)

        x = phi(gamma) / phi(g)^p / phi(u)^i = 1 (mod n)

    3: profit
        x^2 = 1 mod n
        gcd(n, x-1)
"""


class LogLevel(int, Enum):
    """Shim for logging loglevels for typing"""

    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    TRACE = 5
    NOTSET = logging.NOTSET


logging.addLevelName(LogLevel.TRACE, "TRACE")

main_logger = logging.getLogger()


class Logger(Enum):
    MISC = "\x1b[1;33m"
    TIMER = "\x1b[1;34m"
    SIEVE = "\x1b[1;35m"
    FACTORBASE = "\x1b[1;36m"
    SEARCH = "\x1b[1;37m"

    @classmethod
    @functools.cache
    def max_length(cls) -> int:
        """Length of longest logger name"""
        return max(map(lambda l: len(l.name), cls))  # type: ignore

    @property
    def logger(self) -> logging.Logger:
        """
        Return the logging.Logger instance for this Logger

        Uses an ugly hack to color the logger name
        """
        return main_logger.getChild(
            f"{self.value}{self.name.ljust(self.max_length())}\x1b[0m"
        )


def log(msg: str, logger: Logger, level: int = LogLevel.DEBUG) -> None:
    """Log the message to the given Logger at the given level"""
    logger.logger.log(msg=msg, level=level)


@contextlib.contextmanager
def time_section(task: str, level: int = LogLevel.DEBUG) -> Iterator[None]:
    """Time a section of code and log the timings"""
    start_time = time.monotonic()
    log(
        f"Started {task} at {datetime.datetime.now().isoformat(timespec='seconds')}",
        level=level,
        logger=Logger.TIMER,
    )
    yield
    log(
        f"Completed {task} in {(time.monotonic() - start_time) / 60:.2f} minutes",
        level=level,
        logger=Logger.TIMER,
    )


@dataclass(frozen=True)
class Params:
    """Class storing parameters used in the snfs implementation"""

    n: int  # number to factor
    multiple: int  # N = n * multiple
    # N = r^e - s
    N: int
    r: int
    e: int
    s: int

    # f(x) = x^d - t
    d: int
    t: int
    abs_alpha: float  # |t|^(1/d)
    m: int  # A root of f over Z/Zn
    r1: int  # Number of real roots of f
    r2: int  # Number of complex roots of f up to conjugation
    l: int  # Number of units - 1. l = r1 + r2 - 1
    alpha_i: tuple[float | complex]  # Roots of f over C. len(alpha_i) == l

    # Sieving bounds
    a_max: int
    a_min: int
    b_max: int

    B: int  # Rational and algebraic smoothness bound
    extra_relations: int  # Number of relations + #I to gather
    M: int  # Multiplier bound for generators
    C: float  # Norm search bound for generators


def L(n: float, alpha: float, c: float, fudge: float = 0.1) -> float:
    """Compute L_n[alpha, c]"""
    return math.exp(
        (c + fudge) * math.log(n) ** alpha * math.log(math.log(n)) ** (1 - alpha)
    )


def sgn(x: float) -> Literal[-1, 1]:
    """Return the sign of x, 1 if x=0"""
    return 1 if x >= 0 else -1


def map_to_integer(gamma: tuple[float, ...]) -> tuple[int, ...]:
    """Map the elements to ints"""
    assert all(map(lambda s: s == int(s), gamma)), f"{gamma}"
    return tuple(map(int, gamma))


def negate(elements: tuple[int, ...]) -> tuple[int, ...]:
    """Map (a, b, c, ...) -> (-a, -b, -c, ...)"""
    return tuple(map(lambda x: -x, elements))


def f(x: int, p: int, params: Params) -> int:
    """Evaluate f(x) (mod p)"""
    return (pow(x, params.d, p) - params.t) % p


def small_norm(a: int, b: int, params: Params) -> int:
    """Compute the norm of a + b alpha in Z[alpha]"""
    assert b > 0
    result = pow(a, params.d) - params.t * pow(-b, params.d)
    assert isinstance(result, int), type(result)
    return result


def is_nth_power(x: int, n: int) -> bool:
    assert n > 0

    if x < 0 and n % 2 == 0:
        return False

    if x < 0:
        x *= -1
    nth_root = int(pow(x, 1 / n))
    return int(pow(nth_root, n)) == x


def phi(gamma: tuple[int, ...], params: Params) -> int:
    """Ring homomorphism phi: Z[alpha] -> Z_n, phi(alpha) = (m mod n)"""
    # a + bm and a + balpha have the same image under phi
    return sum(s * pow(params.m, i, params.n) for i, s in enumerate(gamma)) % params.n


def to_full_vector(a: int, b: int, params: Params) -> tuple[int, ...]:
    """Return the representation in Z[alpha] of a + b alpha"""
    return (a, b, *itertools.repeat(0, params.d - 2))


def search_value(gamma: tuple[int, ...], params: Params) -> float:
    assert len(gamma) == params.d
    return sum(s_i**2 * params.abs_alpha ** (2 * i) for i, s_i in enumerate(gamma))


def mult_matrix(gamma: tuple[float, ...], params: Params) -> sympy.Matrix:
    return sympy.Matrix(
        [
            [
                gamma[offset - i] * (params.t if i > offset else 1)
                for i in range(params.d)
            ]
            for offset in range(params.d)
        ]
    )


def float_multiply(
    gamma1: tuple[float, ...], gamma2: tuple[float, ...], params: Params
) -> sympy.Matrix:
    return mult_matrix(gamma1, params) * sympy.Matrix(params.d, 1, gamma2)


def multiply(
    gamma1: tuple[float, ...], gamma2: tuple[float, ...], params: Params
) -> tuple[int, ...]:
    return map_to_integer(float_multiply(gamma1, gamma2, params))


def is_inverse(
    gamma1: tuple[float, ...], gamma2: tuple[float, ...], params: Params
) -> bool:
    """Return True if gamma1 = gamma2^-1 (gamma1 * gamma2 = 1)"""
    return multiply(gamma1, gamma2, params) == to_full_vector(1, 0, params)


def norm(gamma: tuple[int, ...], params: Params) -> int:
    """Return the norm of gamma = sum(s_i * alpha^i)"""
    # Uses the definition that the norm is the determinant of the matrix for the
    # multiplication with gamma
    # Uses sympy matrix to get the precise integer answer
    # TODO: Use sympy to compute a polynomial for this
    return int(mult_matrix(gamma, params).det())


def invert(gamma: tuple[float, ...], params: Params) -> sympy.Matrix:
    return mult_matrix(gamma, params).solve(
        sympy.Matrix(params.d, 1, [1 if i == 0 else 0 for i in range(params.d)])
    )


def compute_power(
    gamma: tuple[float, ...], exponent: int, params: Params
) -> tuple[float, ...]:
    """Compute gamma^exponent in the number field"""
    if exponent == 0:
        return to_full_vector(1, 0, params)  # 1

    if exponent < 0:
        exponent *= -1
        gamma = invert(gamma, params)

    # Dumb linear time power increase
    result = gamma
    for _ in range(exponent - 1):
        result = float_multiply(result, gamma, params)

    return result


@functools.cache
def invert_generator(g: tuple[int, ...], e_g: int, params: Params) -> tuple[float, ...]:
    """Compute g^(-e_g)"""
    assert e_g > 0
    result = compute_power(g, -e_g, params)

    assert is_inverse(result, compute_power(g, e_g, params), params)
    return result


def phi_i(gamma: tuple[float, ...], i: int, params: Params) -> complex:
    """
    phi_i: K -> C, phi_i maps sum(s_k alpha^k) to sum(s_k alpha_i^k)

    NOTE: 0-indexed (i in [0, ..., l-1]
    """
    return sum(s * params.alpha_i[i] ** k for k, s in enumerate(gamma))


def nu(gamma: tuple[int, ...], params: Params) -> tuple[float, ...]:
    assert any(gamma)
    return tuple(
        math.log(abs(phi_i(gamma, i=i, params=params)))
        - math.log(abs(norm(gamma, params))) / params.d
        for i in range(params.l)
    )


def is_negative(gamma: tuple[int, ...], params: Params) -> bool:
    """Return True if gamma is negative under the chosen embedding into R"""
    assert params.r1 > 0, "Not considering embeddings into C yet"
    # u0 = -1
    # Use phi_0(gamma) as the embedding into R
    value = phi_i(gamma, i=0, params=params)
    assert isinstance(value, float)
    return value < 0


def fix_sign(gamma: tuple[int, ...], params: Params) -> tuple[int, ...]:
    """Multiply by u0 so the element behaves properly under the chosen embedding"""
    assert any(gamma)
    if params.r1 > 0:
        return negate(gamma) if is_negative(gamma, params) else gamma
    else:
        assert False


def compute_v(
    gamma: tuple[int, ...],
    G: tuple[tuple[int, ...], ...],
    G_contribution: tuple[int, ...],
    params: Params,
) -> tuple[int, ...]:
    """Compute (a + b alpha) * Πg^(-e(g))"""
    for g, e_g in zip(G, G_contribution):
        if e_g == 0:
            continue
        gamma = multiply(gamma, invert_generator(g, e_g, params), params)
    return gamma


def _setup_nfs(
    r: int,
    e: int,
    s: int,
    multiple: int = 1,
    extra_relations: int = 5,
    prime_bound: int | None = None,
    sieve_bound: int | None = None,
) -> Params:
    """Compute and validate parameters for nfs"""
    assert sieve_bound is None or sieve_bound > 1
    assert prime_bound is None or prime_bound > 1

    N = r**e - s
    n = N // multiple

    log(f"{r=} {e=} {s=} {multiple=}", logger=Logger.MISC)
    log(f"{n=}", logger=Logger.MISC)
    log(f"{N=}", logger=Logger.MISC)

    assert math.gcd(N, multiple) == multiple

    # TODO: Deal with small multiples of r**e-s

    # Select factor base
    # factor_base = sieve_of_eratosthenes(1000)

    # Collect relations (factorizations) until it slightly exceeds the #factor_base

    # Linalg => x^2 = 1 (mod n)

    # Construction of the number field [2.5]

    # The extension degree [6.3]
    # The degree of f, where our number field K = Q(alpha), f(alpha) = 0
    d_fudge = 0.1  # o(1)
    d = math.ceil(
        pow(
            ((3 + d_fudge) * math.log(N)) / (2 * math.log(math.log(N))),
            1 / 3,
        )
    )

    assert d >= 2

    # The polynomial f(X) = X^d - t (assumed irreducible)
    # k * d >= e
    k = math.ceil(e / d)
    assert k * d >= e

    t = s * pow(r, k * d - e)

    m = pow(r, k)  # A root of f (mod n)

    log(f"{d=} {k=} {t=} m={r}^{k}", logger=Logger.MISC)

    # Assert irreducibility
    # f is reducible if and only if either there is
    # a prime number p dividing d such that t is a pth power,
    assert not any(
        is_nth_power(t, n=p) for p in sieve_of_eratosthenes(d) if math.gcd(p, d) == d
    )
    # or 4 divides d and 4t is a fourth power
    assert not (math.gcd(4, d) == 4 and is_nth_power(4 * t, n=4))

    # Implied by Z[alpha] being the ring of integers in K
    assert e % d == 0 or e % d == (-1) % d, f"{e=} {d=}"

    # Assume Z[alpha] UFD

    # Step 1 of the sieve [2.7]
    # Select smoothness bound B (=B1=B2)
    # TODO: This value is way larger then the one they present in [8] (pg. 37)
    B = prime_bound or math.ceil(L(N, 1 / 3, (2 / 3) ** (2 / 3)))  # [6.3]
    a_max = b_max = sieve_bound or math.ceil(L(N, 1 / 3, (2 / 3) ** (2 / 3)))  # [6.3]
    a_min = -a_max

    log(f"{B=}", logger=Logger.FACTORBASE)

    abs_alpha = abs(t) ** (1 / d)

    # Ordered list of roots used in search
    # NOTE: 0-indexed (i in [0, ..., l-1]
    alpha_i: list[float | complex] = []

    # Let f have r1 real roots and 2r2 complex roots => r2 = (d-r1)/2
    if d % 2 == 1:
        r1 = 1
        alpha_i.append(abs_alpha * sgn(t))
    elif t < 0:
        r1 = 0
    elif t > 0:
        r1 = 2
        alpha_i.append(abs_alpha)
        alpha_i.append(-abs_alpha)

    r2 = (d - r1) // 2

    l = r1 + r2 - 1  # noqa: E741

    log(f"Roots of the polynomial: {r1=} {2*r2=} {l=}", logger=Logger.SEARCH)

    assert r1 > 0, "Have not implemented r1 == 0 yet"

    for i in range(1, r2):
        assert (2 * i + (1 - sgn(t)) // 2) % d != 0, "Don't repeat a real root"
        alpha_i.append(
            cmath.rect(abs_alpha, (2 * cmath.pi / d) * (i + (1 - sgn(t)) / 4))
        )

    log(f"{alpha_i=}", logger=Logger.SEARCH)

    # TODO: Fix these values [3.6]
    M = 10  # Multiplier bound
    C = d * abs(t) ** ((d - 1) / d) * B ** (2 / d)  # Search bound

    return Params(
        n=n,
        multiple=multiple,
        N=N,
        r=r,
        e=e,
        s=s,
        d=d,
        t=t,
        abs_alpha=abs_alpha,
        m=m,
        r1=r1,
        r2=r2,
        l=l,
        alpha_i=cast(tuple[float | complex], tuple(alpha_i)),
        a_max=a_max,
        a_min=a_min,
        b_max=b_max,
        B=B,
        extra_relations=extra_relations,
        M=M,
        C=C,
    )


def _find_ideals(
    P: tuple[int, ...], params: Params
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, ...], ...]]:
    """
    Compute all the first degree prime ideals of Z[alpha] as pairs (p, c mod p)

    Returns a tuple with two elements:
        ((p1, c11), (p1, c12), (p2, c21), (p4, c41), ...)
        ((c11, c12), (c21,), (), (c41, ...), ...)
    """
    # U and G - see [3]
    # Ideals
    # https://en.wikipedia.org/wiki/Berlekamp%E2%80%93Rabin_algorithm#Berlekamp's_method
    ideals_per_prime = tuple(
        tuple(c for c in range(p) if f(c, p, params) == 0) for p in P
    )

    ideals = sum(
        (tuple((p, c) for c in c_list) for p, c_list in zip(P, ideals_per_prime)),
        start=(),
    )

    log(
        "Ideals per prime: (p, (c1, c2, ...))",
        logger=Logger.FACTORBASE,
        level=LogLevel.TRACE,
    )
    for p, c_list in zip(P, ideals_per_prime):
        log(
            f"\t{(p, c_list)}",
            logger=Logger.FACTORBASE,
            level=LogLevel.TRACE,
        )

    log(f"#G={len(ideals)}", logger=Logger.FACTORBASE)

    return ideals, ideals_per_prime


def _search_for_elements(
    P: tuple[int, ...], ideals_per_prime: tuple[tuple[int, ...], ...], params: Params
) -> tuple[
    tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], npt.NDArray[np.float64]
]:
    # The units of Z[alpha] are generated by a suitable root of unity u0, and l
    # multiplicatively independent units u1, ..., ul
    # If r1 > 0 we can let u0 = -1
    # Otherwise we need some complex embedding
    if params.r1 > 0:
        u0 = to_full_vector(-1, 0, params)
    else:
        assert False

    # An element x = sum(s_i alpha^i) belongs to the ideal "(p, c)"
    # if sum(s_i c^i) = 0 (mod p)
    # An element x generates "(p, c)" if it belongs to it, and has norm N(x) = +- p

    # Search for elements of U and G

    # For all gamma in Z[alpha] with sum(s_i^2 |alpha|^2i) <= C
    # TODO: use search for short vectors in lattice
    s_bounds = tuple(
        math.floor(math.sqrt(params.C / params.abs_alpha ** (2 * i)))
        for i in range(params.d)
    )

    log(
        f"Searching {functools.reduce(operator.mul, s_bounds) * 2**params.d} "
        f"members of Z[alpha]. Bounds: {s_bounds} ({params.C=:.2f})",
        logger=Logger.SEARCH,
    )

    multipliers = [[params.M + 1] * len(c_list) for c_list in ideals_per_prime]
    generators: list[list[tuple[int, ...] | None]] = [
        [None] * len(c_list) for c_list in ideals_per_prime
    ]
    found_units: set[tuple[int, ...]] = set()
    small_ideals: list[tuple[int, ...] | None] = [None] * (params.M - 1)

    for gamma in itertools.product(*(range(-bound, bound + 1) for bound in s_bounds)):
        if not any(gamma):
            # Skip 0
            continue

        if search_value(gamma, params) > params.C:
            continue

        norm_g = norm(gamma, params)

        if abs(norm_g) == 1:
            # Store found units
            found_units.add(gamma)
        elif 2 <= abs(norm_g) <= params.M:
            # Store found ideals with norm <= M
            # Used to divide out the multiplier from the generators after
            small_ideals[abs(norm_g) - 2] = gamma

        try:
            norm_factors = factor_into(abs(norm_g), P)
        except FactorizationError:
            continue

        # This alg looks at all ideals, and assumes only one valid per p
        # The paper says: identify the first ideal where this is true
        for prime_index, (p, c_list, power) in enumerate(
            zip(P, ideals_per_prime, norm_factors)
        ):
            if not c_list:
                continue

            if norm_factors[prime_index] == 0:
                continue

            k = norm_g // p
            if abs(k) > params.M:
                continue

            for ideal_index, c in enumerate(c_list):
                if sum(s_i * c**i for i, s_i in enumerate(gamma)) % p == 0:
                    break
            else:
                # No appropriate ideal found of this norm
                continue

            if abs(multipliers[prime_index][ideal_index]) > abs(k):
                multipliers[prime_index][ideal_index] = k
                generators[prime_index][ideal_index] = gamma

    inverse_small_ideals: list[tuple[int, ...] | None] = [
        invert(g, params) if g is not None else None for g in small_ideals
    ]

    log(
        f"{multipliers=}",
        logger=Logger.SEARCH,
        level=LogLevel.TRACE,
    )

    log(
        f"{small_ideals=}",
        logger=Logger.SEARCH,
        level=LogLevel.TRACE,
    )

    for g, g_inv in zip(small_ideals, inverse_small_ideals):
        if g is None:
            continue
        assert g_inv is not None

        assert is_inverse(g, g_inv, params)

    for prime_index, (current_multipliers, current_generators) in enumerate(
        zip(multipliers, generators)
    ):
        assert all(
            abs(m) <= params.M for m in current_multipliers
        ), "Too large multiplier at {prime_index=}"
        assert not any(
            g is None for g in current_generators
        ), "Missing generator at {prime_index=}"

        for i, (mult, gen) in enumerate(zip(current_multipliers, current_generators)):
            assert gen is not None
            if abs(mult) == 1:
                continue
            else:
                # The inverse of an ideal with norm `mult`
                multiplier_inverse = inverse_small_ideals[abs(mult) - 2]
                assert multiplier_inverse is not None, f"{prime_index=} {mult=}"
                result = current_generators[i] = multiply(
                    gen, multiplier_inverse, params
                )
                current_multipliers[i] = norm(result, params)

                assert abs(current_multipliers[i]) == 1

        if params.r1 > 0:
            # u0 = -1
            # Use phi_0(gamma) as the embedding into R
            for i, (mult, gen) in enumerate(
                zip(current_multipliers, current_generators)
            ):
                assert gen is not None
                assert abs(mult) == 1
                current_generators[i] = fix_sign(gen, params)
        else:
            # Complex embedding
            assert False

    G: tuple[tuple[int, ...], ...] = sum(
        map(tuple, generators), start=()  # type: ignore
    )

    # TODO: find more units by dividing generators of the same abs norm

    assert params.r1 > 0
    # u0 = -1

    # Remove units generated by another unit
    for u in found_units.copy():
        assert abs(norm(u, params)) == 1
        if u not in found_units:
            # Already been removed
            continue

        span = tuple(
            map_to_integer(compute_power(u, exponent, params))
            for exponent in range(-10, 10)
        )
        for new_unit in span:
            shifted = multiply(new_unit, u0, params)

            if new_unit != u and new_unit in found_units:
                found_units.remove(new_unit)

            if shifted in found_units:
                found_units.remove(shifted)

    # Remove units generated by a pair of other units
    for u1 in found_units.copy():
        for u2 in found_units - {u1}:
            if u1 not in found_units or u2 not in found_units:
                # Already been removed
                continue

            span1 = tuple(
                map_to_integer(compute_power(u1, exponent, params))
                for exponent in range(-10, 10)
            )
            span2 = tuple(
                map_to_integer(compute_power(u2, exponent, params))
                for exponent in range(-10, 10)
            )

            for u1_power, u2_power in itertools.product(span1, span2):
                new_unit = multiply(u1_power, u2_power, params)
                shifted = multiply(new_unit, u0, params)

                if new_unit != u1 and new_unit != u2 and new_unit in found_units:
                    found_units.remove(new_unit)

                if shifted in found_units:
                    found_units.remove(shifted)

    # TODO: Implement for arbitrary l
    # TODO: Use nu(u) to determine dependencies

    assert (
        len(found_units) == params.l
    ), "Did not find a sufficient set of units during search"
    assert (
        params.l <= 2
    ), "Linear dependencies are currently only removed for l<=2"  # noqa: E741

    # If we first assert r1 > 0 and then make the unit positive under embedding
    # We can mby skip out on u0
    # Either way we can mby look at every selection of l units from the found units and
    # see what they generate

    U: tuple[tuple[int, ...], ...] = (
        u0,
        *map(functools.partial(fix_sign, params=params), found_units),
    )
    nu_u = tuple(map(functools.partial(nu, params=params), U[1:]))

    log(f"#U={len(U)}", logger=Logger.FACTORBASE)

    W_inv = np.linalg.inv(np.array(nu_u).T)

    """
    TODO:
    It also helps to select (or to change) the elements 9 E G
    such that the coordinates of W- 1 . v(g) lie between and one can achieve
    this by multiplying 9 by an appropriate product of units (to be determined with
    the help of v).
    """

    log(
        f"{generators=}",
        logger=Logger.SEARCH,
        level=LogLevel.TRACE,
    )
    log(
        f"{U=}",
        logger=Logger.SEARCH,
        level=LogLevel.TRACE,
    )

    return U, G, W_inv


def _create_factorbase(
    P: tuple[int, ...],
    U: tuple[tuple[int, ...], ...],
    G: tuple[tuple[int, ...], ...],
    params: Params,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], tuple[int, int] | None]:
    """Create the factorbase I and its image under phi"""
    # Define I = P and U and G
    I = tuple(  # noqa: E741
        itertools.chain(map(lambda p: to_full_vector(p, 0, params), P), U, G)
    )
    phi_I = tuple(map(functools.partial(phi, params=params), I))

    log(
        f"#I={len(I)}",
        logger=Logger.FACTORBASE,
    )

    log(
        f"{I=}",
        logger=Logger.FACTORBASE,
        level=LogLevel.TRACE,
    )
    log(
        f"{phi_I=}",
        logger=Logger.FACTORBASE,
        level=LogLevel.TRACE,
    )

    trivial_factorization = None

    # Assume gcd(a_i, n) == 1, otherwise we have a trivial factorization
    for a in phi_I:
        if math.gcd(a, params.n) != 1:
            log(
                f"Member of phi(I) {a} had {math.gcd(a, params.n)=} != 1",
                logger=Logger.MISC,
            )

            p1 = math.gcd(a, params.n)
            p2 = params.n // p1

            assert params.n % p1 == 0

            if p1 > p2:
                trivial_factorization = (p2, p1)
            trivial_factorization = (p1, p2)

            break

    return I, phi_I, trivial_factorization


def _sieve_for_relations(
    ideals: tuple[tuple[int, int], ...],
    ideals_per_prime: tuple[tuple[int, ...], ...],
    P: tuple[int, ...],
    U: tuple[tuple[int, ...], ...],
    G: tuple[tuple[int, ...], ...],
    W_inv: npt.NDArray[np.float64],
    params: Params,
) -> npt.NDArray[np.int32]:
    # Sieve like described in [4]
    assert abs(params.a_min) < 2**63 and abs(params.a_max) < 2**63
    a_range: npt.NDArray[np.int64] = np.arange(
        params.a_min, params.a_max + 1, dtype=np.int64
    )

    total_sieved = 0
    total_valid = 0

    factorbase_size = len(P) + len(U) + len(G)

    log_P = np.log(P)

    powers_matrix: npt.NDArray[np.int32] = np.empty(
        (factorbase_size, factorbase_size + params.extra_relations), dtype=np.int32
    )

    # TODO: Free relations [2.13]

    # TODO: multiprocess bs or chunks of bs
    for b in itertools.count(1):
        log(
            f"Found {total_valid}/{factorbase_size + params.extra_relations} "
            f"relations before {b=}",
            logger=Logger.SIEVE,
        )

        if total_valid == factorbase_size + params.extra_relations:
            break

        log(f"{b=} {'-' * 40}", logger=Logger.SIEVE, level=LogLevel.TRACE)

        if b > params.b_max:
            log(
                f"Sieve exceeded value of b_max {b=}>{params.b_max}",
                logger=Logger.SIEVE,
                level=LogLevel.WARNING,
            )
            break

        # Assume a + bm > 0, else replace with (-a, -b)
        assert ((a_range + b * params.m) > 0).all()

        # First sieve
        # Use log of last prime as fudge factor
        threshold = np.log((a_range + b * params.m).astype(np.float32)) - log_P[-1]

        logsum = np.zeros_like(threshold)

        for p, log_p in zip(P, log_P, strict=True):
            offset = (-b * params.m) % p
            logsum[(offset - params.a_min) % p :: p] += log_p

        rational_sieve_valid = logsum >= threshold
        log(
            f"{np.sum(rational_sieve_valid)} valid from rational sieve",
            logger=Logger.SIEVE,
            level=LogLevel.TRACE,
        )

        # Second sieve
        logsum = np.zeros_like(threshold)
        threshold = (
            np.log(
                np.abs(
                    np.power(a_range, params.d, dtype=np.float32)
                    - params.t * (-b) ** params.d
                )
            )
            - log_P[-1]
        )

        for c_list, p, log_p in zip(ideals_per_prime, P, log_P, strict=True):
            if b % p == 0:
                # No integers a s.t. N(a + b alpha) = 0 (mod p)
                # TODO: The paper says this, but I'm not sure I believe it
                # Then for any fixed integer b with 0 < b :::; u and b =1= 0 mod p,
                # the integers a with N(a + bo:) == 0 modp are those with a == -br modp
                # for some r E R(p).  Note that if b == 0 mod p, then there are no
                # integers a with (a, b) E U and N(a + bo:) 0 mod p.
                continue

            for c in c_list:
                offset = (-b * c) % p
                logsum[(offset - params.a_min) % p :: p] += log_p

        algebraic_sieve_valid = logsum >= threshold

        log(
            f"{np.sum(algebraic_sieve_valid)} valid from algebraic sieve",
            logger=Logger.SIEVE,
            level=LogLevel.TRACE,
        )

        coprime_valid = np.gcd(a_range, b) == 1

        log(
            f"{np.sum(coprime_valid)} valid from gcd",
            logger=Logger.SIEVE,
            level=LogLevel.TRACE,
        )

        a_candidates = (
            np.where(rational_sieve_valid & algebraic_sieve_valid & coprime_valid)[0]
            + params.a_min
        )
        total_sieved += len(a_candidates)
        log(
            f"{len(a_candidates)} candidates for a with: {a_candidates}",
            logger=Logger.SIEVE,
            level=LogLevel.TRACE,
        )

        for a in a_candidates:
            try:
                P_contribution = factor_into(a + b * params.m, P)
            except FactorizationError:
                # a + bm not B1-smooth
                log(
                    f"{a} + {b} * {params.m} does not factor into P",
                    logger=Logger.SIEVE,
                    level=LogLevel.TRACE,
                )
                continue

            current_norm = small_norm(int(a), int(b), params)

            assert current_norm != 0, f"Cannot factor 0 = small_norm({a=}, {b=})"

            G_contribution_list = [0] * len(ideals)

            for i, (p, c) in enumerate(ideals):
                if (a + b * c) % p != 0:
                    continue

                while current_norm % p == 0:
                    G_contribution_list[i] += 1
                    current_norm //= p

                if abs(current_norm) == 1:
                    break
            else:
                # a + b alpha not B2-smooth
                if abs(current_norm) != 1:
                    log(
                        f"({a} + {b} * alpha) does not factor into the prime ideals",
                        logger=Logger.SIEVE,
                        level=LogLevel.TRACE,
                    )
                    for p in P:
                        assert (
                            current_norm % p != 0
                        ), f"Missed a prime ideal of norm {p}"

                    continue

            G_contribution = tuple(G_contribution_list)

            gamma = to_full_vector(a, b, params)

            if params.r1 > 0:
                e_u0 = 1 if is_negative(gamma, params) else 0
            else:
                # TODO: fix
                assert False, "Compute e(u_0)"

            v_u = compute_v(gamma, G, G_contribution, params)
            unitless = multiply(gamma, invert(v_u, params), params)

            U_plus_contribution = W_inv @ nu(v_u, params)
            # YOLO rounding:
            U_contribution = (e_u0, *map(round, U_plus_contribution))

            # Compute (a + b alpha) from the factorization to verify
            prime_result = functools.reduce(
                functools.partial(multiply, params=params),
                itertools.starmap(
                    lambda base, exponent: compute_power(base, exponent, params),
                    zip(G, G_contribution),
                ),
                to_full_vector(1, 0, params),
            )
            unit_result = functools.reduce(
                functools.partial(multiply, params=params),
                itertools.starmap(
                    lambda base, exponent: compute_power(base, exponent, params),
                    zip(U, U_contribution),
                ),
                to_full_vector(1, 0, params),
            )

            result = multiply(prime_result, unit_result, params)

            assert prime_result == unitless, f"{prime_result=} {unitless=}"

            try:
                assert unit_result == v_u, f"{unit_result=} {v_u=}"
                assert result == gamma, f"{result=} {gamma}"
            except AssertionError:
                log(
                    f"Factorization of {a} + {b}*alpha failed due to rounding error. "
                    f"{v_u=} {nu(v_u, params)=} estimated powers={U_plus_contribution}",
                    logger=Logger.SIEVE,
                    level=LogLevel.ERROR,
                )
                continue

            new_relation = (
                *P_contribution,
                *negate(U_contribution),
                *negate(G_contribution),
            )

            powers_matrix[:, total_valid] = new_relation
            total_valid += 1
            if total_valid == factorbase_size + params.extra_relations:
                break

    log(f"Found {total_sieved} pairs from sieveing", logger=Logger.SIEVE)
    log(f"Found {total_valid} valid pairs from sieveing", logger=Logger.SIEVE)

    # assert total_valid >= factorbase_size + extra_relations, total_valid
    if total_valid < factorbase_size + params.extra_relations:
        log(
            f"Proceeding with only {total_sieved} relations with #I={factorbase_size}",
            logger=Logger.SIEVE,
        )
        powers_matrix = powers_matrix[:total_valid, :]

    return powers_matrix


def _evaluate_kernel_vectors(
    powers_matrix: npt.NDArray[np.int32],
    kernel_vectors: Iterable[tuple[int, ...]],
    phi_I: tuple[int, ...],
    params: Params,
) -> tuple[int, int]:
    """Factor n by using the kernel vectors mod 2 of the relation matrix"""
    i = 0  # In case there are no kernel vectors

    for i, coefficients in enumerate(kernel_vectors):
        if not any(powers_matrix @ coefficients):
            # Check that the resulting number is not 1
            continue

        assert all(
            map(lambda exponent: exponent % 2 == 0, powers_matrix @ coefficients)
        )

        x = functools.reduce(
            lambda x, y: (x * y) % params.n,
            itertools.starmap(
                lambda a, exponent: pow(a, int(exponent), params.n),
                zip(phi_I, (powers_matrix @ coefficients) // 2),
            ),
        )

        if (x + 1) % params.n == 0 or (x - 1) % params.n == 0:
            # We hit a bad pair of roots - try again
            continue

        p1 = math.gcd(x - 1, params.n)
        p2 = params.n // p1

        if p1 == 1 or p2 == 1:
            continue

        assert params.n % p1 == 0

        log(f"Obtained factorization with kernel vector nr. {i+1}", logger=Logger.MISC)

        if p1 > p2:
            return (p2, p1)

        return (p1, p2)

    assert False, f"Failed in {i+1} vectors"


def nfs(
    r: int,
    e: int,
    s: int,
    multiple: int = 1,
    extra_relations: int = 5,
    prime_bound: int | None = None,
    sieve_bound: int | None = None,
) -> tuple[int, int]:
    """
    Return the factorization of r^e - s using the special number field sieve

    multiple: n * multiple should be r**e - s
    """
    params = _setup_nfs(
        r=r,
        e=e,
        s=s,
        multiple=multiple,
        extra_relations=extra_relations,
        prime_bound=prime_bound,
        sieve_bound=sieve_bound,
    )

    with time_section(f"prime sieveing up to {params.B}"):
        P = sieve_of_eratosthenes(params.B)
        log(f"#P={len(P)}", logger=Logger.FACTORBASE)

    with time_section("ideal finding"):
        ideals, ideals_per_prime = _find_ideals(P, params)

    with time_section("search for elements"):
        U, G, W_inv = _search_for_elements(P, ideals_per_prime, params)

    I, phi_I, trivial_factorization = _create_factorbase(P, U, G, params)

    if trivial_factorization is not None:
        return trivial_factorization

    # Step 2 of the sieve [2.8]
    # Search for relations gcd(a, b) == 1, |a + bm| B1 smooth, a + balpha B2 smooth

    with time_section("sieveing"):
        powers_matrix = _sieve_for_relations(
            ideals, ideals_per_prime, P, U, G, W_inv, params
        )

    with time_section("computing kernel"):
        kernel_vectors = kernel_vectors_mod_2(powers_matrix)

    with time_section("evaluating kernel vectors"):
        factorization = _evaluate_kernel_vectors(
            powers_matrix, kernel_vectors, phi_I, params
        )

    return factorization


if __name__ == "__main__":
    examples = (
        {
            "r": 2,
            "e": 41,
            "s": 1,
            "prime_bound": 150,
            "sieve_bound": 500,
            "extra_relations": 1,
        },
        {"r": 2, "e": 101, "s": 1},  # ~6 mins
        {"r": 3, "e": 239, "s": 1, "prime_bound": 479910, "sieve_bound": 5 * 10**6},
        {"r": 2, "e": 373, "s": -1, "prime_bound": 287120, "sieve_bound": 5 * 10**6},
        {
            "r": 2,
            "e": 23,
            "s": 1,
            "prime_bound": 46,
            "sieve_bound": 250,
        },  # Fails w like 8
        # {"r":2, "e":512, "s":-1},  # alpha^2 / 2
    )
    logging.basicConfig(
        level=LogLevel.DEBUG,
        format="%(name)s;\x1b[0;36m%(levelname)-7s;\x1b[0m%(message)s",
    )

    loglevels: dict[Logger, int] = {
        Logger.MISC: LogLevel.DEBUG,
        Logger.TIMER: LogLevel.DEBUG,
        Logger.SIEVE: LogLevel.DEBUG,
        Logger.FACTORBASE: LogLevel.DEBUG,
        Logger.SEARCH: LogLevel.DEBUG,
    }

    quiet = False

    for logger, level in loglevels.items():
        logger.logger.setLevel(LogLevel.WARNING if quiet else level)

    kwargs = examples[1]

    with time_section(f"nfs(**{kwargs})"):
        n = kwargs["r"] ** kwargs["e"] - kwargs["s"]
        p, q = nfs(**kwargs)
        print(f"{n=}, {p=} {q=}")
        assert p * q == n
