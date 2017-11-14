import numpy as np
from numpy.random import *
import hashlib
from bitstring import BitArray
import matplotlib.pyplot as plt
import os
import re
from file_utility.file_utility import *
try:
    import cPickle as pickle
except ImportError:
    import pickle

key = ["w1", "b1", "w2", "b2", "w3", "b3"]

def save_params(path, d):
    with open(path, 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

def load_params(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def input_with_confirm(check_func = lambda x: True, before_msg = "", success_msg = "", failed_msg = ""):
    i = input(before_msg)
    if check_func(i):
        print(success_msg)
        return i
    print(failed_msg)
    return input_with_confirm(check_func, before_msg, success_msg, failed_msg)

def load_params_with_confirm(params):
    path = "./params"
    if not os.path.exists(path):
        os.makedirs(path)
    params_files_name = [f for f in read_child_files_name(path) if is_file_extension(f, ".pkl")]

    if params_files_name:
        params_files = [[int(re.findall("\d+", i)[0]), i] for i in params_files_name]
        params_files.sort()

        epochs = [i[0] for i in params_files]

        print("You can use the learned parameter file.")
        s = input("Do you use it? (y/n) : ")
        if s == "y" or s == "Y":
            print("What epochs number do you want to start with?")
            for i in params_files:
                print("epoch ", i[0])

            before_msg = "Please enter the number : "
            success_msg = "The parameter file was loaded!"
            failed_msg = "The number you entered is invalid.\nPlease enter the valid number."
            epoch_input = int(input_with_confirm(lambda x: int(x) in epochs, before_msg, success_msg, failed_msg))

            params_file_name = params_files[epochs.index(epoch_input)][1]
            params = load_params(path + "/" + params_file_name)
            return epoch_input + 1

    return 0

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
        #print(a)
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

start_epoch = load_params_with_confirm(network._params)

print(network._params)

num_epochs = 10
batch_size = 1
learning_rate = 0.1

h = {}
eta = {}

for s in key:
    h[s] = 1e-8
    eta[s] = 0.001

for itr in range(num_epochs):
    #batch_mask = choice(train_size, batch_size)
    #x_batch = x_train[batch_mask]
    print("itr : ", itr)
    grad = network.numerical_gradient(x_train[itr])
    print(grad)
    for s in key:
        h[s] += grad[s] * grad[s]
        eta[s] /= h[s] ** 0.5
        network._params[s] -= eta[s] * grad[s]
    
    save_params("./params/" + str(start_epoch + itr) + ".pkl", network._params)


x = [i for i in range(10)]
y = [network.output(i) for i in x]

plt.plot(x, y)
