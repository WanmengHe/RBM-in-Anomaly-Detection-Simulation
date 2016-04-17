import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from arrow import utcnow
from sklearn.manifold import TSNE


def plot_trendency(seqs):
    n = seqs.shape[1]
    x = np.linspace(0, n * 10, n)
    # with plt.style.context('fivethirtyeight'):
    for seq in seqs:
        plt.plot(x, seq)
    plt.show()


def scatter_3D_space(points, exp_name, file_name):
    if points.shape[1] != (3 + 1):
        print "Error! The dimension of data is not 3."
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for poi in points:
        if poi[0] == 0:
            c = "r"
            m = "o"
        else:
            c = "b"
            m = "^"
        ax.scatter(poi[1], poi[2], poi[3], c=c, marker=m)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    if not os.path.exists("data/" + exp_name):
        os.makedirs("data/" + exp_name)
    fig.savefig("data/" + exp_name + "/" + file_name + "_3D_scatter.png", dpi=fig.dpi)
    plt.show()


def scatter_2D_space(points, exp_name, file_name):
    fig = plt.figure()
    if points.shape[1] != (2 + 1):
        print "Error! The dimension of data is not 2."
        return
    for poi in points:
        if poi[0] == 0:
            c = "r"
            m = "o"
        else:
            c = "b"
            m = "^"
        plt.scatter(poi[1], poi[2], c=c, marker=m)
    if not os.path.exists("data/" + exp_name):
        os.makedirs("data/" + exp_name)
    fig.savefig("data/" + exp_name + "/" + file_name + "_2D_scatter.png", dpi=fig.dpi)
    plt.show()


def tsne(seqs, exp_name, file_name, T=[], embedded_dim=2):
    model = TSNE(n_components=embedded_dim, random_state=0)
    np.set_printoptions(suppress=True)
    new_dim_points = model.fit_transform(seqs)
    # Insert 0 at head of each line.
    new_dim_points = np.insert(new_dim_points,
                               0,  # position in which inserts.
                               0,  # value which is inserted.
                               axis=1)
    # Revamp the value of the specific line which was specified by array T to -1.
    for t in T:
        new_dim_points[t][0] = -1
    if embedded_dim == 3:
        scatter_3D_space(new_dim_points, exp_name, file_name)
    elif embedded_dim == 2:
        scatter_2D_space(new_dim_points, exp_name, file_name)
    else:
        print "Error! It cannot be visualized! the dimension of data is out of 3."
    return new_dim_points


def write_data(seqs, exp_name, file_name=utcnow().format('YYYY-MM-DD HH:mm:ss ZZ')):
    if not os.path.exists("data/" + exp_name):
        os.makedirs("data/" + exp_name)
    f = open("data/" + exp_name + "/" + file_name + ".txt", "w")
    for line in seqs:
        for ele in line:
            f.write(str(ele) + "\t")
        f.write("\n")
    f.close()


def read_data(exp_name, file_name):
    seqs = []
    if not os.path.exists("data/" + exp_name):
        os.makedirs("data/" + exp_name)
    f = open("data/" + exp_name + "/" + file_name + ".txt")
    line = f.readline()
    while "" != line:
        seq_str = line.strip().split("\t")
        seq_int = map(lambda x: int(float(x)), seq_str)
        seqs.append(seq_int)
        line = f.readline()
    f.close()
    return np.array(seqs)


if __name__ == "__main__":
    # data = [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    # T = [0, 1, 2]
    # dataset = read_data("exp1/one_exception_N6_n1000_t1")
    # new_data = tsne(dataset, T, 3)
    # print new_data
    pass
