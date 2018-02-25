from LR1.calculating import *
from math import *


def intersection_with_ellipsoid(r0, e, p0, a_el, b_el):
    temp1 = [b_el * e[0], a_el * e[1]]
    temp2 = [b_el * (r0[0] - p0[0]), a_el * (r0[1] - p0[1])]
    a = scalar_multipl(temp1, temp1)
    b = 2 * scalar_multipl(temp1, temp2)
    c = scalar_multipl(temp2, temp2) - pow(a_el*b_el, 2)
    D = b * b - 4 * a * c
    t1 = (- b + sqrt(D)) / 2 * a
    t2 = (- b - sqrt(D)) / 2 * a
    # TODO check from intersection
    if t1 < 0:
        return t2
    elif t2 < 0:
        return t1
    else:
        return min(t1, t2)


def normal(r0, e, p0, t):
    temp = diff(sum(r0, multipl(t, e)), p0)
    n = multipl(1 / math.sqrt(scalar_multipl(temp, temp)), temp)
    n = norm(n)
    return n


def refraction_after_ellipsoid(n_refr, e, n):
    e_refr = diff(multipl(n_refr[0], e), multipl(n_refr[0] * abs(scalar_multipl(e, n)) - n_refr[1] * math.sqrt(
        1 - math.pow(n_refr[0] / n_refr[1], 2) * (1 - math.pow(scalar_multipl(e, n), 2))),
                                                 multipl(sign(scalar_multipl(e, n)), n)))
    e_refr = multipl(1 / n_refr[1], e_refr)
    e_refr = norm(e_refr)
    return e_refr


def reflection_from_ellipsoid(e, n):
    e_refl = diff(e, multipl(2 * scalar_multipl(e, n), n))
    e_refl = norm(e_refl)
    return e_refl