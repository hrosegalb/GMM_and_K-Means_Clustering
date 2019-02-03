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


def group_data(distances):
    clusters = np.zeros((distances.shape[0], 1))
    for i in range(distances.shape[0]):
        row = distances[i]
        label = np.argmin(row)
        clusters[i] = label
    
    return clusters


def sum_of_squares_error(prev_centroids, centroids):
    return np.sum((centroids - prev_centroids)**2)


def plot_data(x_vals, y_vals, cent_x, cent_y):
    plt.scatter(x_vals, y_vals, s=7)
    plt.scatter(cent_x, cent_y, marker='*', s=200, c='g')
    plt.show()


def main():
    # Toy data
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

    # k sets the number of groups for the data
    k = 3
    r = 10

    # Randomly initialize the centroids
    centroid_list = []
    centroids = np.array([[random.randint(0, 15), random.randint(0, 15)]for i in range(3)])
    print(centroids)

    error_list = []
    for _ in range(r):
        prev_centroids = np.zeros(centroids.shape)
    
        distances = calculate_distances(data, centroids, k)
        clusters = group_data(distances)
        prev_centroids = copy.deepcopy(centroids)

        for i in range(k):
            data_pts = [data[j] for j in range(data.shape[0]) if clusters[j] == i]
            if data_pts != []:
                centroids[i] = np.mean(data_pts, axis=0)
    
        print(prev_centroids)
        print(centroids)
    
        error = sum_of_squares_error(prev_centroids, centroids)
        centroid_list.append(prev_centroids)
        error_list.append(error)
    
    iterations = list(zip(error_list, centroid_list))
    print(iterations)
    min_idx = error_list.index(min(error_list))
    print("Min index: {0}".format(min_idx))

    # Split data up in order to plot it
    x_vals = np.array(data[:, 0])
    y_vals = np.array(data[:, 1])
    cent_x = np.array(centroids[:, 0])
    cent_y = np.array(centroids[:, 1])
    #plot_data(x_vals, y_vals, cent_x, cent_y)

if __name__ == '__main__':
    main()
