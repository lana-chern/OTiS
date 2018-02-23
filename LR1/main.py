import numpy as np
import matplotlib.pyplot as pl
from math import *
from LR1.calculating import *
from LR1.plane import *
from LR1.sphere import *
from LR1.ellipsoid import *


def plot_sphere(p0, R):
    i = np.linspace(0, 2 * pi, 100)
    pl.plot(R * np.cos(i) + p0[0], R * np.sin(i) + p0[1])


def plot_ellipsoid(p0, a, b):
    i = np.linspace(0, 2 * pi, 100)
    pl.plot(a * np.cos(i) + p0[0], b * np.sin(i) + p0[1])


def values_of_plane(i, n, r0):  # TODO correct from n[1]==0
    if n[0] == 0:
        return r0[1]
    else:
        return -n[0] * (i - r0[0]) / n[1] + r0[1]


def plot_ray(p0, e, t):
    j = np.linspace(0, t, 100)
    x = p0[0] + e[0] * j
    y = p0[1] + e[1] * j
    pl.plot(x, y)


if __name__ == '__main__':
    # print('Введите параметры луча: координаты исходной точки и вектор-направление:')
    # p0 = [float(input('x: ')), float(input('y: ')), float(input('z: '))]
    # e = [float(input('xe: ')), float(input('ye: ')), float(input('ze: '))]
    # print('Введите параметры плоскости: радус вектор и вектор нормали')
    # r0 = [float(input('x: ')), float(input('y: ')), float(input('z: '))]
    # n = [float(input('xn: ')), float(input('yn: ')), float(input('zn: '))]

    # Параметры вектора:
    p0 = [7, 4]
    e = [1, 0]

    # Параметры плоскости и сред
    r0 = [4, 2]
    n = [-sqrt(2) / 2, sqrt(2) / 2]
    n_r = [1, 1]

    # t = intersection_with_plane(n, r0, p0, e)
    # print(t)
    # p0_refr = sum(p0, multipl(t, e))
    # p0_refl = p0_refr
    # print(p0_refr)
    # e_refr = refraction_after_plane(n_r, e, n)
    # print('e_refr', e_refr)
    # e_refr = norm(e_refr)
    # print('e_refr norm', e_refr)
    # e_refl = reflection_from_plane(e, n)
    # print('e_refl', e_refl)
    # e_refl = norm(e_refl)
    # print('e_refl norm', e_refl)

    # Параметры сферы
    sph = [7, 5]
    R = 2

    # i = np.linspace(0, 10, 100)
    # pl.plot(i, values_of_plane(i, n, r0), 'g')
    #
    # plot_ray(p0, e, t)
    # plot_ray(p0_refl, e_refl, t)
    # plot_ray(p0_refr, e_refr, t)

    # t_s = intersection_with_sphere(p0, R, sph, e)
    # print(t_s)
    # p0_refl_s = sum(p0, multipl(t_s[1], e))
    # n_s = normal(p0, e, r0, t_s)
    # print(n_s)
    # e_refl_s = reflection_from_sphere(e, n_s[1])
    #
    # p0_refr_s = p0_refl_s
    # e_refr_s = refraction_after_sphere(n_r, e, n_s[1])
    #
    # plot_sphere(sph, R)
    # plot_ray(p0, e, t_s[1])
    # plot_ray(p0_refl_s, e_refl_s, t_s[1])
    # plot_ray(p0_refr_s, e_refr_s, R)

    a = 1
    b = 2
    el = [2, 2]
    plot_ellipsoid(el, a, b)

    pl.show()
