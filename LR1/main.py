import numpy as np
import matplotlib.pyplot as pl
from LR1.plane import *
from LR1.sphere import *
from LR1.ellipsoid import *


def plot_sphere(p0, R):
    i = np.linspace(0, 2 * pi, 100)
    pl.plot(R * np.cos(i) + p0[0], R * np.sin(i) + p0[1])


def plot_ellipsoid(p0, a, b):
    i = np.linspace(0, 2 * pi, 100)
    pl.plot(a * np.cos(i) + p0[0], b * np.sin(i) + p0[1])


def plot_plane(i, n, r0):
    x = r0[0] + n[1] * i
    y = r0[1] - n[0] * i
    pl.plot(x, y, label='Плоскость')


def plot_ray(p0, e, t, name):
    j = np.linspace(0, t, 100)
    x = p0[0] + e[0] * j
    y = p0[1] + e[1] * j
    pl.plot(x, y, label=name)


def check_plane():
    t = intersection_with_plane(n, r0, p0, e)
    if t is None:
        print('Конец работы функции')
    else:
        print(t)

        p0_refl = sum(p0, multipl(t, e))
        e_refl = reflection_from_plane(e, n)
        print('e_refl', e_refl)

        p0_refr = p0_refl
        e_refr = refraction_after_plane(n_r, e, n)
        print('e_refr', e_refr)

        i = np.linspace(0, 10, 100)

        plot_plane(i, n, r0)
        plot_ray(p0, e, t, 'Исходный луч')
        plot_ray(p0_refl, e_refl, t, 'Отражённый луч')
        plot_ray(p0_refr, e_refr, t, 'Преломлённый луч')
        pl.legend(loc=1)
        pl.grid()
        pl.show()


def check_sphere():
    t_s = intersection_with_sphere(p0, R, sph, e)
    if t_s is None:
        print('Конец работы функции')
    else:
        print(t_s)
        n_s = normal(p0, e, sph, t_s)
        print(n_s)

        p0_refl_s = sum(p0, multipl(t_s, e))
        e_refl_s = reflection_from_sphere(e, n_s)
        print('e_refl_s', e_refl_s)

        p0_refr_s = p0_refl_s
        e_refr_s = refraction_after_sphere(n_r, e, n_s)
        print('e_refr_s', e_refr_s)

        plot_sphere(sph, R)
        plot_ray(p0, e, t_s, 'Исходный луч')
        plot_ray(p0_refl_s, e_refl_s, t_s, 'Отражённый луч')
        plot_ray(p0_refr_s, e_refr_s, R, 'Преломлённый луч')
        pl.legend(loc=1)
        pl.grid()
        pl.show()


def check_ellipsoid():
    t_e = intersection_with_ellipsoid(p0, e, el, a, b)
    print(t_e)
    if t_e is None:
        print('Конец работы функции')
    else:

        p0_refl_e = sum(p0, multipl(t_e, e))
        n_e = normal(p0, e, el, t_e)
        print('normal ', n_e)
        e_refl_e = reflection_from_ellipsoid(e, n_e)
        print('e_refl_e', e_refl_e)
        p0_refr_e = p0_refl_e
        e_refr_e = refraction_after_ellipsoid(n_r, e, n_e)
        print('e_refr_e', e_refr_e)

        plot_ellipsoid(el, a, b)
        plot_ray(p0, e, t_e, 'Исходный луч')
        plot_ray(p0_refl_e, e_refl_e, t_e, 'Отражённый луч')
        plot_ray(p0_refr_e, e_refr_e, t_e, 'Преломлённый луч')
        pl.legend(loc=1)
        pl.grid()
        pl.show()


if __name__ == '__main__':
    # Параметры вектора:
    p0 = [0, 2]
    e = [1, -1]
    e = norm(e)

    # Параметры плоскости и сред
    r0 = [0, 0]
    n = [0, 1]
    n = norm(n)
    print(e, n)
    n_r = [1, 1.2]

    check_plane()

    # Параметры сферы
    sph = [0, 0]
    R = 4

    # check_sphere()

    # Параметры эллипса
    a = 2
    b = 3
    el = [2, 2]

    # check_ellipsoid()
