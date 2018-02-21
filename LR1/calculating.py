import math


def sum(p0, p1):
    p3 = [p0[0] + p1[0], p0[1] + p1[1]]
    return p3


def diff(p0, p1):
    p3 = [p0[0] - p1[0], p0[1] - p1[1]]
    return p3


def multipl(a, p0):
    p1 = [a * p0[0], a * p0[1]]
    return p1


def scalar_multipl(p0, p1):
    return p0[0] * p1[0] + p0[1] * p1[1]


def norm(e):
    norma = math.sqrt(e[0] * e[0] + e[1] * e[1])
    e_norm = [e[0] / norma, e[1] / norma]
    return e_norm


def sign(a):
    if a < 0:
        return -1
    elif a == 0:
        return 0
    else:
        return 1
