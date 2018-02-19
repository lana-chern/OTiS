import numpy as np
import matplotlib.pyplot as pl
import math


def sum(p0, p1):
    p3 = [p0[0] + p1[0], p0[1] + p1[1]]
    return p3


def diff(p0, p1):
    p3 = [p0[0] - p1[0], p0[1] - p1[1]]
    return p3


def product(a, p0):
    p1 = [a * p0[0], a * p0[1]]
    return p1


def scalar_product(p0, p1):
    return p0[0] * p1[0] + p0[1] * p1[1]


def norm(e):
    norma = math.sqrt(e[0] * e[0] + e[1] * e[1])
    e[0] /= norma
    e[1] /= norma
    return e


def intersection_with_plane(n, r0, p0, e):
    temp = scalar_product(n, e)
    if temp == 0:
        if scalar_product(n, diff(r0, p0)) == 0:
            print("Луч принадлежит плоскости")
        else:
            print("Луч параллелен плоскости")
    # elif temp < 0:
    #     print("Луч не пересекает плоскость")
    else:
        return scalar_product(n, diff(r0, p0)) / scalar_product(n, e)


def intersection_with_sphere(n, r0, R, p0, t):
    return 0


def intersection_with_ellipsoid(n, r0, R, p0, t):
    return 0


def refraction_after_plane(n_refr, e, n):
    e_refr = diff(product(n_refr[0], e), product(n_refr[0] * abs(scalar_product(e, n)) - n_refr[1] * math.sqrt(
        1 - math.pow(n_refr[0] / n_refr[1], 2) * (1 - math.pow(scalar_product(e, n), 2))), n))
    e_refr = product(1 / n_refr[1], e_refr)
    return e_refr


def reflection_from_plane(e, n):
    e_refl = diff(e, product(2 * scalar_product(e, n), n))
    return e_refl


def values_of_plane(i, n, r0):
    # if n[1] == 0:
    #     return r0[0]
    # else:
    y = -n[0] * (i - r0[0]) / n[1] + r0[1]
    return y


def values_of_ray(i, p0, e):
    return p0[1] + (i - p0[0]) * e[1] / e[0]


if __name__ == '__main__':
    # print('Введите параметры луча: координаты исходной точки и вектор-направление:')
    # p0 = [float(input('x: ')), float(input('y: ')), float(input('z: '))]
    # e = [float(input('xe: ')), float(input('ye: ')), float(input('ze: '))]
    # print('Введите параметры плоскости: радус вектор и вектор нормали')
    # r0 = [float(input('x: ')), float(input('y: ')), float(input('z: '))]
    # n = [float(input('xn: ')), float(input('yn: ')), float(input('zn: '))]
    p0 = [1, 3]
    e = [1, 0]
    r0 = [4, 2]
    n = [-math.sqrt(2) / 2, math.sqrt(2) / 2]
    n_r = [1, 2]

    t = intersection_with_plane(n, r0, p0, e)
    print(t)
    p0_refr = sum(p0, product(t, e))
    p0_refl = p0_refr
    print(p0_refr)
    e_refr = refraction_after_plane(n_r, e, n)
    print('e_refr', e_refr)
    e_refr = norm(e_refr)
    print('e_refr', e_refr)
    e_refl = reflection_from_plane(e, n)
    print('e_refl', e_refl)
    e_refl = norm(e_refr)
    print('e_refl', e_refl)

    i = np.linspace(-5, 10, 100)
    j = np.linspace(p0[0], p0_refl[0], 100)
    k = np.linspace(p0_refr[0], p0_refr[0] + 1, 100)
    l = np.linspace(p0_refl[0], p0_refl[0] + 1, 100)

    pl.plot(i, values_of_plane(i, n, r0), 'g')
    pl.plot(j, values_of_ray(j, p0, e), 'b')
    pl.plot(k, values_of_ray(k, p0_refr, e_refr), 'c')
    pl.plot(l, values_of_ray(l, p0_refl, e_refl), 'r')
    pl.show()
