import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from arrow import utcnow
from sklearn.manifold import TSNE

def plot_trendency(seqs):
    n = seqs.shape[1]
    x = np.linspace(0, n*10, n)

    # with plt.style.context('fivethirtyeight'):
    for seq in seqs:
        plt.plot(x, seq)

    plt.show()

def scatter_3D_space(points):
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

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def tsne(seqs, T=[], embedded_dim=2):
    model = TSNE(n_components=embedded_dim, random_state=0)
    np.set_printoptions(suppress=True)
    new_dim_points = model.fit_transform(seqs)
    # Insert 0 at head of each line.
    new_dim_points = np.insert(new_dim_points,
        0, # position in which inserts.
        0, # value which is inserted.
        axis=1)
    # Revamp the value of the specific line which was specified by array T to -1.
    for t in T:
        new_dim_points[t][0] = -1
    scatter_3D_space(new_dim_points)
    return new_dim_points

def write_data(seqs, file_name=utcnow().format('YYYY-MM-DD HH:mm:ss ZZ')):
    f = open("data/" + file_name + ".txt", "w")
    for line in seqs:
        for ele in line:
            f.write(str(ele) + "\t")
        f.write("\n")
    f.close()

def read_data(file_name):
    seqs = []
    f = open("data/" + file_name + ".txt")
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
    T = [0, 1, 2]
    dataset = read_data("decode_one_exception_N6_n1000_t1")
    new_data = tsne(dataset, T, 2)