import itertools


def tonelli_shanks(n: int, p: int) -> int:
    """
    Find a square root of n mod p

    https://en.wikipedia.org/wiki/Tonelli%E2%80%93Shanks_algorithm#The_algorithm
    """

    assert n > 0
    assert p != 2

    Q = p - 1
    for S in itertools.count():
        if Q % 2 != 0:
            break
        Q //= 2

    for z in range(p):
        if pow(z, (p - 1) // 2, p) == p - 1:
            break
    else:
        raise ValueError(f"Found no non-quadratic residues in Z_{p}")

    M = S
    c = pow(z, Q, p)
    t = pow(n, Q, p)
    R = pow(n, (Q + 1) // 2, p)

    while True:
        if t == 0:
            return 0
        elif t == 1:
            return R

        t_squared = t
        for i in itertools.count():
            if t_squared == 1:
                break
            t_squared = pow(t_squared, 2, p)

        b = pow(c, 2 ** (M - i - 1), p)

        M = i
        c = pow(b, 2, p)
        t = (t * c) % p
        R = (R * b) % p
