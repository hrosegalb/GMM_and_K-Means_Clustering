import numpy as np
import matplotlib.pyplot as plt

import random
import copy

def calculate_distances(data, centroids, k):
    distances = np.zeros((data.shape[0], k))
    
    for i in range(centroids.shape[0]):
        c = centroids[i]
        dist = (data - c)**2
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)
        distances[:, i] = dist.T
    
    return distances


def group_data(data, centroids, k):
    distances = calculate_distances(data, centroids, k)
    
    clusters = np.zeros((distances.shape[0], 1))
    for i in range(distances.shape[0]):
        row = distances[i]
        label = np.argmin(row)
        clusters[i] = label
    
    return clusters


def main():
    data = [[0, 1],
            [4, 5],
            [12,9],
            [4, 3],
            [8, 9],
            [3, 1],
            [5, 6],
            [7, 2],
            [1, 3]
    ]
    data = np.array(data)
    print(data)
    print("\n")

    k = 3
    centroids = np.array([[random.randint(0, 15), random.randint(0, 15)]for i in range(3)])
    print(centroids)

    x_vals = np.array(data[:, 0])
    y_vals = np.array(data[:, 1])
    plt.scatter(x_vals, y_vals, s=7)

    cent_x = np.array(centroids[:, 0])
    cent_y = np.array(centroids[:, 1])
    plt.scatter(cent_x, cent_y, marker='*', s=200, c='g')
    #plt.show()

    prev_centroids = np.zeros(centroids.shape)
    
    clusters = group_data(data, centroids, k)
    prev_centroids = copy.deepcopy(centroids)

    for i in range(k):
        data_pts = [data[j] for j in range(data.shape[0]) if clusters[j] == i]
        centroids[i] = np.mean(data_pts, axis=0)
    
    print(prev_centroids)
    print(centroids)
    
    error = np.linalg.norm(centroids - prev_centroids)
    print("\n")
    print(error)

if __name__ == '__main__':
    main()
