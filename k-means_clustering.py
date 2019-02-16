from read_in_csv import read_csv

import numpy as np
import matplotlib.pyplot as plt

import random
from scipy.stats import multivariate_normal

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


def plot_k_means_data(x_vals, y_vals, cent_x, cent_y, targets, k):
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
    #plt.show()
    plt.savefig("best-k-means_{0}-clusters.png".format(k))

def expectation_step(data, centroids, covariance_matrices, priors):
    responsibilities = np.zeros((centroids.shape[0], data.shape[0]))

    for j in range(data.shape[0]):
        for i in range(centroids.shape[0]):
            var = multivariate_normal(mean=centroids[i], cov=covariance_matrices[i])
            responsibilities[i,j] = priors[i] * var.pdf(data[j])

    responsibilities = responsibilities / np.sum(responsibilities, axis=0)
    return responsibilities


def main():
    k = 3  # Sets the number of clusters for the data
    r = 10 # Sets the number of trials to be performed

    # Import the data 
    data = read_csv("GMM_dataset.csv")

    # Randomly initialize the centroids
    max_val = data.max()
    centroids = np.array([[random.uniform(0.0, max_val), random.uniform(0.0, max_val)]for i in range(k)])

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

    # Find the centroids with the smallest sum of squares error
    min_idx = error_list.index(min(error_list))
    best_pick = centroid_list[min_idx]
    targets = cluster_list[min_idx]

    # Split data up in order to plot it
    x_vals = np.array(data[:, 0])
    y_vals = np.array(data[:, 1])
    cent_x = np.array(best_pick[:, 0])
    cent_y = np.array(best_pick[:, 1])
    plot_k_means_data(x_vals, y_vals, cent_x, cent_y, targets, k)


    # GMMs
    centroids = best_pick
    covariance_matrices = []
    priors = np.array([1/k for i in range(k)])

    targets = cluster_list[min_idx]
    for i in range(k):
            data_pts = [data[j] for j in range(data.shape[0]) if targets[j] == i]
            if data_pts != []:
                data_pts = np.array(data_pts)
                cov = np.cov(data_pts.T)
                print("Covariance shape: {0}".format(cov.shape))
                covariance_matrices.append(cov)

    responsibilities = expectation_step(data, centroids, covariance_matrices, priors)

    N_k = np.sum(responsibilities, axis=1)
    print(N_k)
    print(N_k.shape)

    new_mu = 

if __name__ == '__main__':
    main()
