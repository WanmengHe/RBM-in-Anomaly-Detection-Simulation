from random import uniform, gauss, choice, randint
from math import pow, sin
import numpy as np


class DataGenerator():
    func = ("uniform", "linear", "power", "sin")

    def __init__(self, len, count):
        self.len = len
        self.count = count

    def noise(self):
        noise_avg = 0
        noise_std = 1
        return gauss(noise_avg, noise_std)


    def uniform_function(self, x, noise_factor, K):
        y = K * (1 + self.noise() * noise_factor)
        if y < 0:
            y = 0
        return y

    def linear_function(self, x, noise_factor, K, bias, phase):
        y = K * (1 + self.noise() * noise_factor) * (x - phase) + bias
        if y < 0:
            y = 0
        return y

    def power_function(self, x, noise_factor, K, bias, phase, exp):
        y = K * (1 + self.noise() * noise_factor) * pow(x - phase, exp) + bias
        if y < 0:
            y = 0
        return y

    def sin_function(self, x, noise_factor, K, bias, phase):
        y = K * (sin(x - phase) + self.noise() * noise_factor) + bias
        if y < 0:
            y = 0
        return y

    def signal_iterator(self, func_id):
        signal = []
        # The proportion of noise could not over 30% of signal.
        noise_factor = uniform(0, 0.3)
        # Factor of any kind of function, used to amplify or minify the signal.
        K = uniform(-10, 10)
        # Bias of any kind of function, used to shift the signal to up(increase) or down(decrease).
        # The proportion of absolute value of bias could not over 100% of signal.
        bias = randint(0, 5)
        # Phase of any kind of function(Except uniform function), used to shift the signal to left(backward) or right(forward).
        phase = uniform(-10, 10)
        # Exponent of power funtion
        exp = randint(-2, 2)
        for x in range(0, self.len):
            if func_id == 0:
                y = self.uniform_function(x, noise_factor, abs(K))
            elif func_id == 1:
                y = self.linear_function(x, noise_factor, K, bias, phase)
            elif func_id == 2:
                y = self.power_function(x, noise_factor, K, bias, phase, exp)
            elif func_id == 3:
                y = self.sin_function(x, noise_factor, K, bias, phase)
            else:
                y = -1
            signal.append(int(y))
        return signal

    def generate(self):
        dataset = []
        func_info = []
        func_ids = [0, 3, 3]
        for i in range(0, self.count):
            # func_id = choice([0, 3])
            func_id = func_ids.pop()
            func_info.append(self.func[func_id])
            dataset.append(self.signal_iterator(func_id))
        return dataset, func_info

def generate_one_node_dataset(T, c_num):
    dg = DataGenerator(T, c_num)
    dataset, func = dg.generate()
    return np.array(dataset).T

def generate_multi_nodes_dataset(n_num, T, c_num):
    dataset = []
    for i in range(n_num):
        dataset.append(generate_one_node_dataset(T, c_num))
    return np.array(dataset)


def binomial_function(mean):
    return np.random.binomial(1, mean)

def generate_multi_nodes_binomial(n_num, T, c_num, mean):
    dataset = []
    for i in range(n_num):
        t_seq = []
        for j in range(T):
            c_set = []
            for k in range(c_num):
                c_set.append(binomial_function(mean))
            t_seq.append(c_set)
        dataset.append(t_seq)
    return np.array(dataset)

if __name__ == "__main__":
    # dg = DataGenerator(10, 10)
    # ds1, funcs1 = dg.generate()
    # print funcs1
    # for d in ds1:
    #     print d
    print generate_multi_nodes_binomial(2, 4, 88, 0.5)

