import numpy as np
import matplotlib.pyplot as plt
from random import uniform, gauss, choice, randint

# Global variables
noise_sigma = 100
sin_lam_max = 0.1
sin_bias_max = 1000

# Kernel functions
def poisson_seq(n=10, lam=100, noise_k=1):
    seq = np.random.poisson(lam, n)
    return np.array(map(lambda x: abs(int(x + noise_k * (np.random.normal(lam, noise_sigma) - lam))), seq))

def sin_seq(n=10, lam=2, phase=5, bias=0, K=100, noise_k=1):
    _x = np.linspace((0 + phase) * lam, (n + phase) * lam, n)
    seq = np.sin(_x) + 1
    return np.array(map(lambda x: abs(int(K * x + bias + noise_k * (np.random.normal(K, noise_sigma) - K))), seq))

def linear_seq():
    pass

# Every content sequence consist of one poisson component and several sin components
def content_seq(n=10, lam=100, noise_k=1):
    print "Content seq:"
    print "----------------------------------------------------"
    k_list = list(decomposition(lam))
    print "The proportion of each component includes: ", k_list
    pop_index = randint(0, len(k_list) - 1)
    print "The proportion of poisson is: ", k_list[pop_index]
    seq = poisson_seq(n=n, lam=k_list.pop(pop_index), noise_k=noise_k)
    print "Poisson component is:\n ", seq

    for k in k_list:
        phase = uniform(-np.pi, np.pi)
        sin_lam = uniform(0, sin_lam_max)
        sin_bias = uniform(0, sin_bias_max)
        new_seq = sin_seq(n=n, lam=sin_lam, phase=phase, bias=sin_bias, K=k, noise_k=noise_k)
        seq += new_seq
        print "Sin component is (proportion:[%d]):\n " % k, new_seq

    return seq

# Utils
def decomposition(i):
    while i > 0:
        n = randint(1, i)
        yield n
        i -= n

def plot_graph(seqs, n=10):
    x = np.linspace(0, n*10, n)

    # with plt.style.context('fivethirtyeight'):
    for seq in seqs:
        plt.plot(x, seq)

    plt.show()

if __name__ == "__main__":
    seqs = []
    n = 100
    seqs.append(content_seq(n=n, lam=8000, noise_k=1))
    # seqs.append(content_seq(n=n, lam=7000, noise_k=1))
    # seqs.append(content_seq(n=n, lam=6000, noise_k=1))
    # seqs.append(content_seq(n=n, lam=1000, noise_k=1))
    # seqs.append(content_seq(n=n, lam=3000, noise_k=1))
    # for i in range(5):
    #     seqs.append(content_seq(n=n, lam=2000, noise_k=1))
    plot_graph(seqs, n=n)
