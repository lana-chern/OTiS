import numpy as np
import matplotlib.pyplot as pl
import math
from LR1.calculating import *
from LR1.plane import *


def values_of_plane(i, n, r0):
    if n[0] == 0:
        return r0[1]
    else:
        return -n[0] * (i - r0[0]) / n[1] + r0[1]


def plot_ray(p0, e, t):
    j = np.linspace(0, t, 100)
    x = p0[0] + e[0] * j
    y = p0[1] + e[1] * j
    pl.plot(x, y)
    return p0[1] + (i - p0[0]) * e[1] / e[0]


if __name__ == '__main__':
    # print('Введите параметры луча: координаты исходной точки и вектор-направление:')
    # p0 = [float(input('x: ')), float(input('y: ')), float(input('z: '))]
    # e = [float(input('xe: ')), float(input('ye: ')), float(input('ze: '))]
    # print('Введите параметры плоскости: радус вектор и вектор нормали')
    # r0 = [float(input('x: ')), float(input('y: ')), float(input('z: '))]
    # n = [float(input('xn: ')), float(input('yn: ')), float(input('zn: '))]
    p0 = [2, 5]
    e = [1, 0]
    r0 = [4, 2]
    n = [-math.sqrt(2) / 2, math.sqrt(2) / 2]
    n_r = [1, 0.8]

    t = intersection_with_plane(n, r0, p0, e)
    print(t)
    p0_refr = sum(p0, multipl(t, e))
    p0_refl = p0_refr
    print(p0_refr)
    e_refr = refraction_after_plane(n_r, e, n)
    print('e_refr', e_refr)
    e_refr = norm(e_refr)
    print('e_refr norm', e_refr)
    e_refl = reflection_from_plane(e, n)
    print('e_refl', e_refl)
    e_refl = norm(e_refl)
    print('e_refl norm', e_refl)

    i = np.linspace(0, 10, 100)
    pl.plot(i, values_of_plane(i, n, r0), 'g')

    plot_ray(p0, e, 5)
    plot_ray(p0_refl, e_refl, t)
    plot_ray(p0_refr, e_refr, t)

    pl.show()
