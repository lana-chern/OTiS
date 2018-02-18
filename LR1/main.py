import numpy as np
import matplotlib.pyplot as pl
import math


def sum(p0, p1):
    p3 = [p0[0] + p1[0], p0[1] + p1[1], p0[2] + p1[2]]
    return p3


def diff(p0, p1):
    p3 = [p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2]]
    return p3


def product(a, p0):
    p1 = [a * p0[0], a * p0[1], a * p0[2]]
    return p1


def scalar_product(p0, p1):
    return p0[0] * p1[0] + p0[1] * p1[1] + p0[2] * p1[2]


def intersection_with_plane(n, r0, p0, e):
    temp = scalar_product(n, e)
    if temp == 0:
        if scalar_product(n, diff(r0, p0)) == 0:
            print("Луч принадлежит плоскости")
        else:
            print("Луч не пересекает плоскость")
    elif temp > 0:
        print("Луч не пересекает плоскость")
    else:
        return scalar_product(n, diff(r0, p0)) / scalar_product(n, e)


def intersection_with_sphere(n, r0, R, p0, t):
    return 0


def intersection_with_ellipsoid(n, r0, R, p0, t):
    return 0


if __name__ == '__main__':
    # print('Введите параметры луча: координаты исходной точки и вектор-направление:')
    # p0 = [float(input('x: ')), float(input('y: ')), float(input('z: '))]
    # e = [float(input('xe: ')), float(input('ye: ')), float(input('ze: '))]
    # print('Введите параметры плоскости: радус вектор и вектор нормали')
    # r0 = [float(input('x: ')), float(input('y: ')), float(input('z: '))]
    # n = [float(input('xn: ')), float(input('yn: ')), float(input('zn: '))]
    p0 = [1, 1, 0]
    e = [-1, -1, 0]
    r0 = [0, 1, 0]
    n = [1, 0, 0]

    t = intersection_with_plane(n, r0, p0, e)
    print(t)
    print(sum(p0, product(t, e)))
