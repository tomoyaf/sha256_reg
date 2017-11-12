import numpy as np
from numpy.random import *
import hashlib
from bitstring import BitArray
import matplotlib.pyplot as plt

key = ["w1", "b1", "w2", "b2", "w3", "b3"]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def swish(x):
    return x * sigmoid(x)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    return exp_a / np.sum(exp_a)

def differential(a):
    return [j - i for i, j in zip(a[:-1], a[1:])]

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    print("ng:", len(x))
    for i in range(len(x)):
        tmp_val = x[i]

        x[i] = tmp_val + h
        fxh1 = f(x)

        x[i] = tmp_val - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2.0 * h)
        x[i] = tmp_val
    
    return grad

class KernelNet:
    def __init__(self, layer_sizes, weight_init_std = 0.01):
        self._params = {}
        self._params["w1"] = weight_init_std * randn(layer_sizes[0], layer_sizes[1])
        self._params["b1"] = np.zeros(layer_sizes[1])

        self._params["w2"] = weight_init_std * randn(layer_sizes[1], layer_sizes[2])
        self._params["b2"] = np.zeros(layer_sizes[2])

        self._params["w3"] = weight_init_std * randn(layer_sizes[2], layer_sizes[3])
        self._params["b3"] = np.zeros(layer_sizes[3])

    def output(self, x):
        z1 = swish(np.dot(x, self._params["w1"]) + self._params["b1"])
        z2 = swish(np.dot(z1, self._params["w2"]) + self._params["b2"])
        z3 = swish(np.dot(z2, self._params["w3"]) + self._params["b3"])

        th = 0.0
        a = np.zeros_like(z3, dtype=int)
        a[z3 < th] = 0
        a[z3 >= th] = 1
        
        #out = BitArray("".join(map(str, z3)))
        #out = [int("".join(map(str, a[i:i+8])), 2) for i in range(0, len(a), 8)]
        #s = "".join(map(str, a))
        #print(s)
        out = BitArray(a)

        return int.from_bytes(hashlib.sha256(out.bytes).digest()[0:17], "big")

    def loss(self, xs):
        return sum([i ** 2 for i in differential(differential([self.output(x) for x in xs]))])
    
    def numerical_gradient(self, xs):
        loss_w = lambda w: self.loss(xs)

        grads = {}
        for s in key:
            print("numerical_gradient : ", s)
            grads[s] = numerical_gradient(loss_w, self._params[s])
        
        return grads

train_size = 10
start_start = 0
start_stop = 100
length_start = 20
length_stop = 100
step_start = 1
step_stop = 5

x_train_p = []
for _ in range(train_size):
    start = randint(start_start, start_stop)
    stop = start + randint(length_start, length_stop)
    step = randint(step_start, step_stop)
    x_train_p.append([[int(j) for j in format(i, "0640b")] for i in range(start, stop, step)])

x_train = np.array(x_train_p)

network = KernelNet(layer_sizes = [640, 640, 640, 640])

iters_num = 10
#batch_size = 1
learning_rate = 0.1

h = {}
eta = {}

for s in key:
    h[s] = 1e-8
    eta[s] = 0.001

for itr in range(iters_num):
    #batch_mask = choice(train_size, batch_size)
    #x_batch = x_train[batch_mask]
    print("itr : ", itr)
    grad = network.numerical_gradient(x_train[itr])

    for s in key:
        h[s] += grad[s] * grad[s]
        eta[s] /= h[s] ** 0.5
        network._params[s] -= eta[s] * grad[s]

x = [i for i in range(10)]
y = [network.output(i) for i in x]

plt.plot(x, y)
