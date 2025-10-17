import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
import numpy as np
from common_me.gradient_me import numerical_gradient
from common_me.functions_me import softmax, CEE

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = CEE(y, t)

        return loss
    
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = SimpleNet()
print("Initial weights:\n", net.W)

f = lambda w: net.loss(x, t)
dw = numerical_gradient(f, net.W)
print("Gradient of weights:\n", dw)