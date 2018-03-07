from LR1.calculating import *
import math


def intersection_with_plane(n, r0, p0, e):
    t = scalar_multipl(n, e)
    if t == 0:
        if scalar_multipl(n, diff(r0, p0)) == 0:
            print("Луч принадлежит плоскости")
        else:
            print("Луч параллелен плоскости")
    else:
        t = scalar_multipl(n, diff(r0, p0)) / t
        if t < 0:
            print("Луч не пересекает плоскость")
        else:
            return t


def refraction_after_plane(n_refr, e, n):
    e_refr = diff(multipl(n_refr[0], e), multipl(n_refr[0] * abs(scalar_multipl(e, n)) - n_refr[1] * math.sqrt(
        1 - math.pow(n_refr[0] / n_refr[1], 2) * (1 - math.pow(scalar_multipl(e, n), 2))),
                                                 multipl(sign(scalar_multipl(e, n)), n)))
    e_refr = multipl(1 / n_refr[1], e_refr)
    e_refr = norm(e_refr)
    return e_refr


def reflection_from_plane(e, n):
    e_refl = diff(e, multipl(2 * scalar_multipl(e, n), n))
    e_refl = norm(e_refl)
    return e_refl
