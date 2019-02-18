from read_in_csv import read_csv

import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Hannah Galbraith
# CS546
# Program 1
# 2/17/19

def calculate_distances(data, centroids, k):
    """ :param data: (1500, 2) matrix of floats
        :param centroids: (k, 2) matrix of floats
        :param k: integer
        
        Calculates the euclidian distance between each data point and each
        centroid. Returns a (1500, k) matrix of all the distances. """

    distances = np.zeros((data.shape[0], k))
    
    for i in range(centroids.shape[0]):
        c = centroids[i]
        dist = (data - c)**2
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)
        distances[:, i] = dist.T
    
    return distances


def group_data(distances):
    """ :param distances: (1500, k) matrix of floats
        Finds the index of the smallest distance for each row in distances and assigns the 
        corresponding data point to the cluster represented by that index.
        
        Returns a (1500,) vector of integers representing the targets for each data point. """

    clusters = np.zeros((distances.shape[0], 1))
    for i in range(distances.shape[0]):
        row = distances[i]
        label = np.argmin(row)
        clusters[i] = label
    
    return clusters


def plot_k_means_data(x_vals, y_vals, cent_x, cent_y, targets, k):
    """ :param x_vals: (1500,) vector of floats
        :param y_vals: (1500,) vector of floats
        :param cent_x: (k,) vector of floats
        :param cent_y: (k,) vector of floats
        :param targets: (1500,) vector of floats
        :param k: integer
        
        Plots each data point according to which target it has been assigned to.
        Saves the graph as a png. """

    if k > 3:
        plt.clf()   # Clear figure so graphs don't overlap

    for i in range(targets.shape[0]):
        if targets[i] == 0:
            plt.scatter(x_vals[i], y_vals[i], c='c', s=7)
        elif targets[i] == 1:
            plt.scatter(x_vals[i], y_vals[i], c='m', s=7)
        elif targets[i] == 2:
            plt.scatter(x_vals[i], y_vals[i], c='y', s=7)
        elif targets[i] == 3:
            plt.scatter(x_vals[i], y_vals[i], c='g', s=7)
        elif targets[i] == 4:
            plt.scatter(x_vals[i], y_vals[i], c='r', s=7)
    
    plt.scatter(cent_x, cent_y, marker='*', s=200, c='b', )
    plt.title("K-Means Clustering with {0} Clusters".format(k))
    plt.savefig("best-k-means_{0}-clusters.png".format(k))


def run_k_means_for_gmm(data, k):
    """ :param data: (1500, 2) matrix of floats
        :param k: integer
        
        This algorithm is called by the gmm_algorithm() function. The function passes in the 
        data set and an integer representing the number of desired clusters. run_k_means_for_gmm() then
        iterates the k-means algorithm 'r' times and returns the centroids and target labels associated with
        the smallest sum-of-squares error. """

    r = 10 # Sets the number of trials to be performed

    # Randomly initialize the centroids
    max_val = data.max()
    min_val = data.min()
    centroids = np.array([[random.uniform(min_val, max_val), random.uniform(min_val, max_val)] for i in range(k)])

    centroid_list = []
    error_list = []
    cluster_list = []
    for _ in range(r):
        centroid_list.append(centroids)

        distances = calculate_distances(data, centroids, k)
        clusters = group_data(distances)

        error = np.zeros((k, 1))
        for i in range(k):
            data_pts = [data[j] for j in range(data.shape[0]) if clusters[j] == i]
            if data_pts != []:
                # Get sum of squares error between all of the data points assigned to a cluster and the centroid
                error[i] = np.sum((data_pts - centroids[i])**2)

                # Get new centroid positions by calculating the mean of all the data points in the cluster
                centroids[i] = np.mean(data_pts, axis=0)
            
        sum_of_squares_error = np.sum(error, axis=0)

        error_list.append(sum_of_squares_error)
        cluster_list.append(clusters)

    # Find the centroids and targets with the smallest sum of squares error
    min_idx = error_list.index(min(error_list))
    best_pick = centroid_list[min_idx]
    targets = cluster_list[min_idx]

    results = {
        "centroids": best_pick,
        "targets": targets,
    }

    return results


def k_means_algorithm():
    """ This algorithm is called in run_k_means_gmm.py. This executes the k-means algorithm 'r' times per
        cluster size, finds the centroids and targets associated with the smallest sum-of-squares error for 
        each cluster size, and plots the clusters and centroids. Prints out the best sum-of-squares value 
        for each cluster size. """

    r = 10 # Sets the number of trials to be performed

    # Import the data 
    data = read_csv("GMM_dataset.csv")
    max_val = data.max()
    min_val = data.min()

    # Randomly initialize the centroids
    for k in [3, 4, 5]:
        centroids = np.array([[random.uniform(min_val, max_val), random.uniform(min_val, max_val)] for i in range(k)])

        centroid_list = []
        error_list = []
        cluster_list = []
        for _ in range(r):
            centroid_list.append(centroids)

            distances = calculate_distances(data, centroids, k)
            clusters = group_data(distances)

            error = np.zeros((k, 1))
            for i in range(k):
                data_pts = [data[j] for j in range(data.shape[0]) if clusters[j] == i]
                if data_pts != []:
                    # Get sum of squares error between all of the data points assigned to a cluster and the centroid
                    error[i] = np.sum((data_pts - centroids[i])**2)

                    # Get new centroid positions by calculating the mean of all the data points in the cluster
                    centroids[i] = np.mean(data_pts, axis=0)
            
            sum_of_squares_error = np.sum(error, axis=0)

            error_list.append(sum_of_squares_error)
            cluster_list.append(clusters)

        # Find the centroids and targets with the smallest sum of squares error
        min_idx = error_list.index(min(error_list))
        best_pick = centroid_list[min_idx]
        targets = cluster_list[min_idx]
        print("[K-Means] Best sum-of-squares error for {0} clusters: {1}\n".format(k, error_list[min_idx]))

        # Split data up in order to plot it
        x_vals = np.array(data[:, 0])
        y_vals = np.array(data[:, 1])
        cent_x = np.array(best_pick[:, 0])
        cent_y = np.array(best_pick[:, 1])
        plot_k_means_data(x_vals, y_vals, cent_x, cent_y, targets, k)

