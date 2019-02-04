from read_in_csv import read_csv

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


def plot_data(x_vals, y_vals, cent_x, cent_y, targets, k):
    for i in range(targets.shape[0]):
        if targets[i] == 0:
            plt.scatter(x_vals[i], y_vals[i], c='c', s=7)
        elif targets[i] == 1:
            plt.scatter(x_vals[i], y_vals[i], c='m', s=7)
        elif targets[i] == 2:
            plt.scatter(x_vals[i], y_vals[i], c='y', s=7)
        elif targets[i] == 3:
            plt.scatter(x_vals[i], y_vals[i], c='g', s=7)
    
    plt.scatter(cent_x, cent_y, marker='*', s=200, c='b')
    plt.show()


def main():
    # Toy data
    #data = [[0, 1],
    #        [4, 5],
    #        [12,9],
    #        [4, 3],
    #        [8, 9],
    #        [3, 1],
    #        [5, 6],
    #        [7, 2],
    #        [1, 3]
    #]
    #data = np.array(data)
    #print(data)
    #print("\n")

    # k sets the number of groups for the data
    # r sets the number of trials
    k = 4
    r = 10

    # Import the data 
    data = read_csv("GMM_dataset.csv")

    # Randomly initialize the centroids
    max_val = int(data.max())
    centroids = np.array([[random.randint(0, max_val), random.randint(0, max_val)]for i in range(k)])

    centroid_list = []
    error_list = []
    cluster_list = []
    for _ in range(r):
        prev_centroids = np.zeros(centroids.shape)
    
        distances = calculate_distances(data, centroids, k)
        clusters = group_data(distances)
        prev_centroids = copy.deepcopy(centroids)

        for i in range(k):
            data_pts = [data[j] for j in range(data.shape[0]) if clusters[j] == i]
            if data_pts != []:
                centroids[i] = np.mean(data_pts, axis=0)
    
        error = sum_of_squares_error(prev_centroids, centroids)

        error_list.append(error)
        centroid_list.append(prev_centroids)
        cluster_list.append(clusters)

    # Find the centroids with the smallest sum of squares error
    min_idx = error_list.index(min(error_list))
    best_pick = centroid_list[min_idx]
    targets = cluster_list[min_idx]

    # Split data up in order to plot it
    x_vals = np.array(data[:, 0])
    y_vals = np.array(data[:, 1])
    cent_x = np.array(best_pick[:, 0])
    cent_y = np.array(best_pick[:, 1])
    plot_data(x_vals, y_vals, cent_x, cent_y, targets, k)

if __name__ == '__main__':
    main()
