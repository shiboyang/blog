import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets._samples_generator import make_blobs


def make_dataset():
    return make_blobs(n_samples=100,
                      n_features=2,
                      centers=4,
                      cluster_std=0.6,
                      random_state=0
                      )


def main():
    # create dataset
    dataset = make_dataset()
    points, _ = dataset

    # initialize centroids
    K = 4
    centers = points[:K]
    N = points.shape[0]
    truth_table = np.zeros([N, K])

    fig, ax = plt.subplots()
    
    while True:
        ax.clear()
        # calculate the E2 distance between the center and points
        for i in range(K):
            truth_table[:, i] = np.linalg.norm(points - centers[i], axis=1)

        min_dis_index = np.argmin(truth_table, axis=1)
        old_centers = centers.copy()
        for i in range(K):
            mask = min_dis_index == i
            centers[i] = np.average(points[mask], axis=0)
            x, y = points[:, 0], points[:, 1]
            plt.scatter(x, y)

        if np.array_equal(old_centers, centers):
            print("Finsh compute")
            break

        plt.show()


if __name__ == '__main__':
    main()
