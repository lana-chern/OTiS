import math
import cmath
import numpy
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D


def rect(x):
    if (x >= -0.5) & (x <= 0.5):
        return 1
    else:
        return 0


def f(x, y):
    return rect(x / 4) * numpy.cos(4 * math.pi * y)


def gaussian_beam(x):
    return numpy.exp(-(x * x))


def gaussian_beam2(x, y):
    return numpy.exp(-x * x) * numpy.exp(-y * y)


def amplitude(func):
    a = []
    for i in func:
        a.append(abs(i))
    return a


def amplitude2(func):
    func_abs = []
    for i in range(len(func)):
        func_abs.append(amplitude(z[i]))
    func_abs = numpy.array(func_abs)
    return func_abs


def phase(func):
    a = []
    for i in func:
        a.append(cmath.phase(i))
    return a


def phase2(func):
    func_phase = []
    for i in range(len(func)):
        func_phase.append(phase(z[i]))
    func_phase = numpy.array(func_phase)
    return func_phase


def append_zeros(func, number_of_zeros):
    f_with_zeros = [0j] * (2 * number_of_zeros + len(func))
    for i in range(len(func)):
        f_with_zeros[number_of_zeros + i] = func[i]
    return f_with_zeros


def extract_function(func, N, number_of_zeros):
    f_without_zeros = []
    for i in range(number_of_zeros, number_of_zeros + N):
        f_without_zeros.append(func[i])
    return f_without_zeros


def finite_fourier_transform(func, N, M, hx, number_of_zeros):
    f1 = append_zeros(func, int(number_of_zeros))
    f1 = numpy.fft.fftshift(f1)
    F = hx * numpy.fft.fft(f1, M)
    F = numpy.fft.ifftshift(F)
    F = extract_function(F, N, int(number_of_zeros))
    return F


def my_fft(func, N, M, hx, number_of_zeros):
    beam = [0j] * N
    for i in range(N):
        beam[i] = [0j] * N

    for i in range(N):
        for j in range(N):
            beam[i][j] = func(x[i], y[j]) + 0j

    beam_f = []
    for i in range(N):
        beam_f.append(finite_fourier_transform(beam[i], N, M, hx, number_of_zeros))

    temp = [1j] * N
    for i in range(N):
        for j in range(N):
            temp[j] = beam_f[j][i]
        temp = finite_fourier_transform(temp, N, M, hx, number_of_zeros)
        for j in range(N):
            beam_f[j][i] = temp[j]

    return [beam, beam_f]


def slice_polar(x, y, f, a):
    for i in range(len(x)):
        for j in range(len(y)):
            if numpy.power(x[i], 2) + numpy.power(y[i], 2) > numpy.power(a, 2):
                f[i][j] = 0


if __name__ == '__main__':
    a = 5
    N = 256
    M = 1024
    b = N * N / (4 * a * M)
    number_of_zeros = (M - N) / 2
    hx = 2 * a / N
    hu = 2 * b / M

    x = numpy.linspace(-a, a, N)
    y = numpy.linspace(-a, a, N)
    x1 = numpy.linspace(-b, b, N)
    y1 = numpy.linspace(-b, b, N)

    result = my_fft(f, N, M, hx, number_of_zeros)
    my_beam = result[0]
    my_beam_f = result[1]

    fig = pl.figure()
    ax = fig.gca(projection='3d')

    x, y = numpy.meshgrid(x, y)
    x1, y1 = numpy.meshgrid(x1, y1)
    # phi = numpy.arctan2(y, x)

    z = my_beam
    z_abs = amplitude2(z)
    z_phase = phase2(z)

    pl.subplot(221)
    z_abs_max = z_abs.max()
    z_min, z_max = -z_abs_max, z_abs_max
    pl.pcolor(x, y, z_abs, cmap='Spectral', vmin=z_min, vmax=z_max)
    pl.axis([x.min(), x.max(), y.min(), y.max()])

    pl.subplot(222)
    z_phase_max = z_phase.max()
    z_min, z_max = -z_phase_max, z_phase_max
    pl.pcolor(x, y, z_phase, cmap='Spectral', vmin=z_min, vmax=z_max)
    pl.axis([x.min(), x.max(), y.min(), y.max()])

    z = my_beam_f
    z_abs = amplitude2(z)
    z_phase = phase2(z)

    pl.subplot(223)
    z_abs_max = z_abs.max()
    z_min, z_max = -z_abs_max, z_abs_max
    pl.pcolor(x1, y1, z_abs, cmap='Spectral', vmin=z_min, vmax=z_max)
    pl.axis([x1.min(), x1.max(), y1.min(), y1.max()])

    pl.subplot(224)
    z_phase_max = z_phase.max()
    z_min, z_max = -z_phase_max, z_phase_max
    pl.pcolor(x1, y1, z_phase, cmap='Spectral', vmin=z_min, vmax=z_max)
    pl.axis([x1.min(), x1.max(), y1.min(), y1.max()])

    # pl.colorbar()
    pl.show()
