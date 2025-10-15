import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(f'点{x}处的导数值为: {d}')
    b = f(x) - d * x
    return lambda t: d * t + b

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
f = tangent_line(function_1, 10)
y2 = f(x)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
plt.plot(x, y2)
plt.show()