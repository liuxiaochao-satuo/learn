import numpy as np
import matplotlib.pylab as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y, label="ReLU")
plt.ylim(-1, 5)
plt.title("ReLU function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()