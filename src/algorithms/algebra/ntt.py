import math
from typing import Iterator, Sequence

# print(_number_theoretic_transform([1, 2, 3, 4, 5, 0, 0, 0], 3*2**8 + 1))

"""
https://conferences.matheo.si/event/16/contribution/23/material/paper/0.pdf
https://math.stackexchange.com/questions/1926714/ntt-for-fast-multiplication-of-polynomials-over-a-finite-field-and-the-connecti
https://eprint.iacr.org/2021/563.pdf
https://math.stackexchange.com/questions/1437624/number-theoretic-transform-ntt-example-not-working-out/1439150
"""


def is_prime(n: int) -> bool:
    return all(n % i != 0 for i in range(2, math.ceil(math.sqrt(n)) + 1))


def is_coprime(a: int, b: int) -> bool:
    return math.gcd(a, b) == 1


def primitive_roots(n: int) -> Iterator[int]:
    """Dumb n^3 brute force for the primitive roots modulo n"""
    assert n > 0

    if n < 2:
        yield 1
        return

    for root in range(1, n):
        for target in range(1, n):
            if not is_coprime(target, n):
                continue
            if not any(target == pow(root, i, n) for i in range(1, n)):
                break
        else:
            yield root


def primitive_root(n: int) -> int | None:
    """Return the smallest primitive root modulo n or None"""
    try:
        return next(primitive_roots(n))
    except StopIteration:
        return None


def primitive_k_th_roots_of_unity(k: int, n: int) -> Iterator[int]:
    """Return a primitive k-th root of unity modulo n"""
    if not 1 < k < n:
        raise ValueError("k must be in [2, n-1]")

    if (n - 1) % k != 0:
        raise ValueError("k must divide n - 1")

    if not is_prime(n):
        # Only consider prime n, so that lambda(n) is n - 1
        raise ValueError("n must be prime")

    return (pow(root, (n - 1) // k, n) for root in primitive_roots(n))


def primitive_k_th_root_of_unity(k: int, n: int) -> int:
    """Return a primitive k-th root of unity modulo n"""
    try:
        return next(primitive_k_th_roots_of_unity(k, n))
    except StopIteration:
        raise ValueError(f"Found no primitive root modulo {n}")


def is_power_of_2(n: int) -> bool:
    """Return True if n is on the form 2^i, i>=0"""
    return n & (n - 1) == 0


def ntt(seq: Sequence[int], p: int, omega: int) -> Sequence[int]:
    """Perform the ntt in the ring X^(2^m) + 1"""
    length = len(seq)
    assert is_power_of_2(length), "Sequence length must be a power of 2"
    m = int(math.log(length, 2))
    assert 2**m == length

    # Fold count is number of folds performed so far
    for fold_count in range(m):
        # Length of the sequence we will fold this iteration
        fold_length = 2 ** (m - fold_count)

        # Empty sequence where we write the new folds
        new_seq = [-1] * length

        # The length of the new folds we will create
        new_fold_length = fold_length // 2

        # Index into the current folds
        for fold_index in range(2**fold_count):
            # Depending on the parity of the fold we want a factor sqrt(-1) in v
            # This is because of the sign difference in the two rings
            # X^(2^(i+1)) v^2 ~ (X^(2^i) - v) x (X^(2^i) + v)
            offset = 2 ** (m - 1) if fold_index % 2 else 0

            m_v = pow(omega, 2 ** (m - fold_count - 1) + 2**m + offset, p)
            p_v = pow(omega, 2 ** (m - fold_count - 1) + offset, p)

            m_offset = fold_index * fold_length
            p_offset = m_offset + new_fold_length

            new_seq[m_offset : m_offset + new_fold_length] = (
                (seq[m_offset + i] + p_v * seq[m_offset + i + new_fold_length]) % p
                for i in range(new_fold_length)
            )
            new_seq[p_offset : p_offset + new_fold_length] = (
                (seq[m_offset + i] + m_v * seq[m_offset + i + new_fold_length]) % p
                for i in range(new_fold_length)
            )

        seq = tuple(new_seq)

    return seq


def ntt_multiply_polynomials(
    poly1: tuple[int, ...], poly2: tuple[int, ...], p: int, omega: int
) -> tuple[int, ...]:
    """Multiply two polynomials from X^(2^m) + 1 by using ntt"""
    return tuple(
        a * b % p
        for a, b in zip(ntt(poly1, p, omega), ntt(poly2, p, omega), strict=True)
    )


# Make recursive version
def ntt_r(seq: tuple[int, ...], p: int, omega: int) -> tuple[int, ...]:
    """Perform the ntt in the ring X^(2^m) + 1"""
    length = len(seq)
    assert is_power_of_2(length), "Sequence length must be a power of 2"
    m = int(math.log(length, 2))
    assert 2**m == length

    return _ntt_r(seq, p, omega, m=m, depth=0, index=0)


def _ntt_r(
    seq: tuple[int, ...], p: int, omega: int, m: int, depth: int, index: int
) -> tuple[int, ...]:
    """Actually perform the ntt"""
    new_fold_length = len(seq) // 2

    if new_fold_length == 0:
        # Base case
        return seq

    # Depending on the parity of the fold we want a factor sqrt(-1) in v
    # This is because of the sign difference in the two rings
    # X^(2^(i+1)) v^2 ~ (X^(2^i) - v) x (X^(2^i) + v)
    offset = 2 ** (m - 1) if index % 2 else 0

    # Plus and minus v in our ring X^(2^i) - v^2
    m_v = pow(omega, 2 ** (m - depth - 1) + 2**m + offset, p)
    p_v = pow(omega, 2 ** (m - depth - 1) + offset, p)

    left_fold = tuple(
        (seq[i] + p_v * seq[i + new_fold_length]) % p for i in range(new_fold_length)
    )

    right_fold = tuple(
        (seq[i] + m_v * seq[i + new_fold_length]) % p for i in range(new_fold_length)
    )

    return _ntt_r(left_fold, p, omega, m=m, depth=depth + 1, index=index) + _ntt_r(
        right_fold, p, omega, m=m, depth=depth + 1, index=index + 1
    )
