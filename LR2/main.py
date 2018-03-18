import math
import numpy
import matplotlib.pyplot as pl


def f(x):
    return numpy.power(x, 3) * numpy.exp(-x * x)


def gaussian_beam(x):
    return numpy.exp(-(x * x))


def amplitude(f):
    a = []
    for i in f:
        a.append(abs(i))
    return a


def phase(f):
    a = []
    for i in f:
        a.append(math.atan2(i.imag, i.real))
    return a


def swap_half(a):
    temp = 0
    n = len(a)
    for i in range(int(n / 2)):
        a[i], a[int(n / 2) + i] = a[int(n / 2) + i], a[i]
        # temp = a[i]
        # a[i] = a[n - 1 - i]
        # a[n - 1 - i] = temp


def append_zeros(f, number_of_zeros):
    f_with_zeros = []
    for i in range(number_of_zeros):
        f_with_zeros.append(0)
    for i in range(len(f)):
        f_with_zeros.append(f[i])
    for i in range(number_of_zeros):
        f_with_zeros.append(0)
    return f_with_zeros


def extract_function(f, N, number_of_zeros):
    f_without_zeros = []
    for i in range(number_of_zeros, number_of_zeros + N):
        f_without_zeros.append(f[i])
    return f_without_zeros


def finite_fourier_transform(f, N, hx, number_of_zeros):
    swap_half(f)
    f1 = append_zeros(f, number_of_zeros)
    F = hx * numpy.fft.fft(f1)
    swap_half(F)
    F = extract_function(F, N, number_of_zeros)
    swap_half(f)
    return F


# def integral(f, a, b):


if __name__ == '__main__':
    a = 5
    N = 1000
    number_of_zeros = 10000
    hx = 2 * a / N
    M = N + 2 * number_of_zeros

    x = numpy.linspace(-a, a, N)
    gaus_beam = gaussian_beam(x)

    gaus_beam_f = finite_fourier_transform(gaus_beam, N, hx, number_of_zeros)

    x1 = numpy.linspace(-(N * N) / (4 * a * M), N * N / (4 * a * M), N)

    pl.subplot(121)
    pl.plot(x, amplitude(gaus_beam))
    pl.plot(x1, amplitude(gaus_beam_f))

    pl.subplot(122)
    pl.plot(x, phase(gaus_beam))
    pl.plot(x1, phase(gaus_beam_f))
    pl.grid()
    pl.show()
