import math
import cmath
import numpy
import matplotlib.pyplot as pl


def f(x):
    return numpy.power(x, 3) * numpy.exp(-x * x)


def gaussian_beam(x):
    return numpy.exp(-(x * x))


def amplitude(func):
    a = []
    for i in func:
        a.append(abs(i))
    return a


def phase(func):
    a = []
    for i in func:
        #a.append(math.atan2(i.imag, i.real))
        a.append(cmath.phase(i))
    return a


def swap_half(a):
    n = len(a)
    for i in range(int(n / 2)):
        a[i], a[int(n / 2) + i] = a[int(n / 2) + i], a[i]


def append_zeros(func, number_of_zeros):
    f_with_zeros = numpy.zeros(2*number_of_zeros+len(func))
    for i in range(len(func)):
        f_with_zeros[number_of_zeros + i] = func[i]
    return f_with_zeros


def extract_function(func, N, number_of_zeros):
    f_without_zeros = []
    for i in range(number_of_zeros, number_of_zeros + N):
        f_without_zeros.append(func[i])
    return f_without_zeros


def finite_fourier_transform(func, N, hx, number_of_zeros):
    f1 = append_zeros(func, int(number_of_zeros))
    swap_half(f1)
    F = hx * numpy.fft.fft(f1)
    swap_half(F)
    F = extract_function(F, N, int(number_of_zeros))
    return F


def method_rect(function, a, b, n):
    h = (b - a) / float(n)
    v = []
    for k in range(n):
        v = [function(a + (k * h))]
    total = sum(v)
    result = h * total
    return result


def integral(function, a, b):
    n = 2
    a1 = method_rect(function, 1, 10, n)
    n *= 2
    a2 = method_rect(function, 1, 10, n)

    while abs(a1 - a2) > 0.001:
        n *= 2
        a1 = method_rect(function, 1, 10, n)
        n *= 2
        a2 = method_rect(function, 1, 10, n)
    return a2


def my_ftt(func, a, b, N, hx, hu):
    sumf = [0j]*N
    u = numpy.zeros(N)
    for i in range(N):
        u[i] = -b + i * hu
        for k in range(N):
            r = cmath.exp(-2 * math.pi * u[i] * (-a + k * hx)*1j)
            f_mas = func(-a + k * hx)
            sumf[i] += f_mas * r
        sumf[i] *= hx
    return sumf


if __name__ == '__main__':
    a = 5
    N = 256
    M = 2048
    b = N * N / (4 * a * M)
    number_of_zeros = (M - N) / 2
    hx = 2 * a / N
    hu = 2 * b / M

    x = numpy.linspace(-a, a, N)
    x1 = numpy.linspace(-b, b, N)

    gaus_beam = gaussian_beam(x)

    gaus_beam_f = finite_fourier_transform(gaus_beam, N, hx, number_of_zeros)
    gaus_beam_f2 = my_ftt(gaussian_beam, a, b, N, hx, hu)

    pl.subplot(121)
    pl.title('Ампитуда')
    pl.plot(x, amplitude(gaus_beam), label='Исходный')
    pl.plot(x1, amplitude(gaus_beam_f), label='Фурье встр.')
    #pl.plot(x1, amplitude(gaus_beam_f2), label='Фурье моё')
    pl.grid()
    pl.legend()

    pl.subplot(122)
    pl.title('Фаза')
    pl.plot(x, phase(gaus_beam), label='Исходный')
    pl.plot(x1, phase(gaus_beam_f), label='Фурье встр.')
    #pl.plot(x1, phase(gaus_beam_f2), label='Фурье моё')
    pl.grid()
    pl.legend()

    pl.show()
