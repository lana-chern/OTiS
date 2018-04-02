import math
import cmath
import numpy
import matplotlib.pyplot as pl


def f(x):
    return numpy.power(x, 3) * numpy.exp(-x * x)


def gaussian_beam(x):
    return numpy.exp(-(x * x))


def f_fourier(x):
    return -math.pi * math.sqrt(math.pi) * gaussian_beam(math.pi * x) * x * (2 * math.pi * math.pi * x * x - 3) / (2 * 1j)


def amplitude(func):
    a = []
    for i in func:
        a.append(abs(i))
    return a


def phase(func):
    a = []
    for i in func:
        a.append(cmath.phase(i))
    return a


def swap_half(a):
    n = len(a)
    for i in range(int(n / 2)):
        a[i], a[int(n / 2) + i] = a[int(n / 2) + i], a[i]


def append_zeros(func, number_of_zeros):
    f_with_zeros = numpy.zeros(2 * number_of_zeros + len(func))
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


def my_ftt(func, a, b, N, hx, hu):
    sumf = [0j] * M
    for i in range(M):
        u = -b + i * hu
        for k in range(N):
            r = cmath.exp(-2 * math.pi * u * (-a + k * hx) * 1j)
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
    x2 = numpy.linspace(-b, b, M)
    x3 = numpy.linspace(-a, a, M)

    gaus_beam = gaussian_beam(x)
    gaus_beam_f = finite_fourier_transform(gaus_beam, N, M, hx, number_of_zeros)
    gaus_beam_f2 = my_ftt(gaussian_beam, a, b, N, hx, hu)

    #my_beam = f(x)
    #my_beam_f = finite_fourier_transform(my_beam, N, M, hx, number_of_zeros)
    #my_beam_f2 = my_ftt(f, a, b, N, hx, hu)


    pl.subplot(131)
    pl.plot(x, gaussian_beam(x))
    pl.title("Пучок Гаусса")
    pl.grid()

    pl.subplot(132)
    pl.title('Амплитуда')
    pl.plot(x, amplitude(gaus_beam), label='Исходный')
    pl.plot(x1, amplitude(gaus_beam_f), label='Фурье встр.')
    pl.plot(x2, amplitude(gaus_beam_f2), label='Фурье числ.')
    pl.grid()
    pl.legend()

    pl.subplot(133)
    pl.title('Фаза')
    pl.plot(x, phase(gaus_beam), label='Исходный')
    pl.plot(x1, phase(gaus_beam_f), label='Фурье встр.')
    pl.plot(x2, phase(gaus_beam_f2), label='Фурье числ.')
    pl.gca().set_ylim(-math.pi, math.pi)
    pl.grid()
    pl.legend()


    """
    pl.subplot(131)
    pl.plot(x, f(x))
    pl.title("Исходный пучок")
    pl.grid()

    pl.subplot(132)
    pl.title('Амплитуда')
    pl.plot(x, amplitude(my_beam), label='Исходный')
    pl.plot(x1, amplitude(my_beam_f), label='Фурье встр.')
    # pl.plot(x2, amplitude(my_beam_f2), label='Моё')
    pl.plot(x, amplitude(f_fourier(x)), label='Аналитика', color='c')
    pl.grid()
    pl.legend()

    pl.subplot(133)
    pl.title('Фаза')
    pl.plot(x, phase(my_beam), label='Исходный')
    pl.plot(x1, phase(my_beam_f), label='Фурье встр.')
    # pl.plot(x2, phase(my_beam_f2), label='Моё')
    pl.plot(x, phase(f_fourier(x)), label='Аналитика', color='c')
    # pl.gca().set_ylim(-math.pi, math.pi)
    pl.grid()
    pl.legend()
    """

    pl.show()
