from LR1.calculating import *
import math


def intersection_with_sphere(r0, R, p0, e):
    temp = math.pow(scalar_multipl(diff(r0, p0), e), 2) - (scalar_multipl(diff(r0, p0), diff(r0, p0)) - R * R)
    if temp < 0:
        print('Луч не пересекает сферу')
    else:
        t1 = (scalar_multipl(diff(r0, p0), e)) - math.sqrt(temp)
        t2 = (scalar_multipl(diff(r0, p0), e)) + math.sqrt(temp)
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


def refraction_after_sphere(n_refr, e, n):
    e_refr = diff(multipl(n_refr[0], e), multipl(n_refr[0] * abs(scalar_multipl(e, n)) - n_refr[1] * math.sqrt(
        1 - math.pow(n_refr[0] / n_refr[1], 2) * (1 - math.pow(scalar_multipl(e, n), 2))),
                                                 multipl(sign(scalar_multipl(e, n)), n)))
    e_refr = multipl(1 / n_refr[1], e_refr)
    return e_refr


def reflection_from_sphere(e, n):
    e_refl = diff(e, multipl(2 * scalar_multipl(e, n), n))
    return e_refl
