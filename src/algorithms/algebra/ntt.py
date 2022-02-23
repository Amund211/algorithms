import math
from typing import Iterable, Sequence

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


def primitive_roots(n: int) -> Iterable[int]:
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


def primitive_k_th_roots_of_unity(k: int, n: int) -> Iterable[int]:
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


def is_power_of_2(n: int):
    """Return True if n is on the form 2^i, i>=0"""
    return n & (n - 1) == 0


# Make recursive version


def ntt(seq: Sequence[int], p: int, omega: int) -> Sequence[int]:
    """Perform the ntt in the ring X^(2^m) + 1"""
    length = len(seq)
    assert is_power_of_2(length), "Sequence length must be a power of 2"
    m = int(math.log(length, 2))
    assert 2**m == length

    # Fold count is number of folds performed so far
    for fold_count in range(m):
        print(f"-----------{fold_count=}--------------")

        new_seq = [None] * length
        print(f"{seq=} {new_seq=}")
        # Length of the sequence we will fold this iteration
        fold_length = 2 ** (m - fold_count)

        # The length of the new folds we will create
        new_fold_length = fold_length // 2

        print(f"{fold_length=} {new_fold_length=}")

        # Index into the current folds
        for fold_index in range(2**fold_count):
            print(f"---Iteration {fold_index=} ----------")
            # Depending on the parity of the fold we want a factor sqrt(-1) in v
            offset = 2 ** (m - 1) if fold_index % 2 else 0

            m_v = pow(omega, 2 ** (m - fold_count - 1) + 2**m + offset, p)
            p_v = pow(omega, 2 ** (m - fold_count - 1) + offset, p)

            m_offset = fold_index * fold_length
            p_offset = m_offset + new_fold_length

            print(f"{m_v=} {p_v=}")
            print(f"{m_offset=} {p_offset=}")
            print(f"{m_offset=} : {m_offset + new_fold_length=}")
            print(f"{p_offset=} : {p_offset + new_fold_length=}")

            for i in range(new_fold_length):
                print(
                    f"\t{i=} {seq[m_offset + i]=} {seq[m_offset + i + new_fold_length]=}"
                )
                print(
                    "\t",
                    (seq[m_offset + i] + p_v * seq[m_offset + i + new_fold_length]) % p,
                )
                print(
                    "\t",
                    (seq[m_offset + i] + m_v * seq[m_offset + i + new_fold_length]) % p,
                )

            new_seq[m_offset : m_offset + new_fold_length] = (
                (seq[m_offset + i] + p_v * seq[m_offset + i + new_fold_length]) % p
                for i in range(new_fold_length)
            )
            new_seq[p_offset : p_offset + new_fold_length] = (
                (seq[m_offset + i] + m_v * seq[m_offset + i + new_fold_length]) % p
                for i in range(new_fold_length)
            )

        seq = new_seq

    return seq


def ntt_r(seq: Sequence[int], p: int, omega: int) -> Sequence[int]:
    """Perform the ntt in the ring X^(2^m) + 1"""
    length = len(seq)
    assert is_power_of_2(length), "Sequence length must be a power of 2"
    m = int(math.log(length, 2))
    assert 2**m == length

    return fold_left(seq, p, omega) + fold_right(seq, p, omega)


def fold_left(seq: Sequence[int], p: int, omega: int) -> Sequence[int]:
    new_seq[m_offset : m_offset + new_fold_length] = (
        (seq[m_offset + i] + p_v * seq[m_offset + i + new_fold_length]) % p
        for i in range(new_fold_length)
    )


def fold_right(seq: Sequence[int], p: int, omega: int) -> Sequence[int]:
    new_seq[p_offset : p_offset + new_fold_length] = (
        (seq[m_offset + i] + m_v * seq[m_offset + i + new_fold_length]) % p
        for i in range(new_fold_length)
    )


"""
    # Fold count is number of folds performed so far
    for fold_count in range(m):
        print(f"-----------{fold_count=}--------------")

        new_seq = [None] * length
        print(f"{seq=} {new_seq=}")
        # Length of the sequence we will fold this iteration
        fold_length = 2 ** (m - fold_count)
        m_v = pow(omega, 2 ** (m - fold_count - 1) + 2**m, p)
        p_v = pow(omega, 2 ** (m - fold_count - 1), p)

        # The length of the new folds we will create
        new_fold_length = fold_length // 2

        print(f"{fold_length=} {new_fold_length=} {m_v=} {p_v=}")

        # Index into the current folds
        for fold_index in range(2**fold_count):
            print(f"---Iteration {fold_index=} ----------")

            m_offset = fold_index * fold_length
            p_offset = m_offset + new_fold_length

            print(f"{m_offset=} {p_offset=}")
            print(f"{m_offset=} : {m_offset + new_fold_length=}")
            print(f"{p_offset=} : {p_offset + new_fold_length=}")

            for i in range(new_fold_length):
                print(
                    f"\t{i=} {seq[m_offset + i]=} {seq[m_offset + i + new_fold_length]=}"
                )
                print(
                    "\t",
                    (seq[m_offset + i] + p_v * seq[m_offset + i + new_fold_length]) % p,
                )
                print(
                    "\t",
                    (seq[m_offset + i] + m_v * seq[m_offset + i + new_fold_length]) % p,
                )

            new_seq[m_offset : m_offset + new_fold_length] = (
                (seq[m_offset + i] + p_v * seq[m_offset + i + new_fold_length]) % p
                for i in range(new_fold_length)
            )
            new_seq[p_offset : p_offset + new_fold_length] = (
                (seq[m_offset + i] + m_v * seq[m_offset + i + new_fold_length]) % p
                for i in range(new_fold_length)
            )

        seq = new_seq

    return seq
"""
