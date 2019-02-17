from read_in_csv import read_csv
from k_means_clustering import run_k_means_algorithm

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import random

def expectation_step(data, centroids, covariance_matrices, priors):
    responsibilities = np.zeros((centroids.shape[0], data.shape[0]))

    for j in range(data.shape[0]):
        for i in range(centroids.shape[0]):
            dist = multivariate_normal(mean=centroids[i], cov=covariance_matrices[i])
            responsibilities[i,j] = priors[i] * dist.pdf(data[j])

    # Normalize the responsibilities by the sum of the responsibilities of a given data point x_i for each cluster
    # Essentially, this sums all the columns of the matrix
    responsibilities = responsibilities / np.sum(responsibilities, axis=0)
    return responsibilities


def maximization_step(data, responsibilities, N_k, k, priors):
    covariance_matrices = []
    centroids = np.zeros((k, 2))
    for i in range(k):
        new_mu = np.dot(responsibilities[i,:], data) / N_k[i]
        centroids[i] = new_mu
        sigma = np.zeros((2,2))

        for j in range(data.shape[0]):
            sigma += responsibilities[i][j] * np.outer((data[j,:] - new_mu), (data[j,:] - new_mu))

        sigma /= N_k[i]
        covariance_matrices.append(sigma)
        priors[i] = N_k[i] / np.sum(N_k)

    return centroids, covariance_matrices, priors


def compute_log_likelihood(data, centroids, covariance_matrices, priors):
    responsibilities = np.zeros((centroids.shape[0], data.shape[0]))

    for j in range(data.shape[0]):
        for i in range(centroids.shape[0]):
            dist = multivariate_normal(mean=centroids[i], cov=covariance_matrices[i])
            responsibilities[i,j] = priors[i] * dist.pdf(data[j])

    summed_responsibilities = np.sum(responsibilities, axis=0)
    log_of_responsibilities = np.log(summed_responsibilities)
    log_likelihood = np.sum(log_of_responsibilities)
    #print("Log likelihood: {0}".format(log_likelihood))
    return log_likelihood


def plot_gmm_data(data, centroids, covariance_matrices, k):
    xlist = np.linspace(-4.0, 4.0)
    ylist = np.linspace(-4.0, 4.0)
    X, Y = np.meshgrid(xlist, ylist)

    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], s=7)
    for i in range(k):
        mu = centroids[i]
        sigma = covariance_matrices[i]

        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        Z = multivariate_normal.pdf(pos, mu, sigma)
        CS = ax.contour(X, Y, Z)

    plt.title("EM Algorithm for GMM Clustering with {0} Clusters".format(k))
    #plt.show()
    plt.savefig("best-gmm_{0}-clusters.png".format(k))



def main():
    r = 10                   # Number of random restarts to be done
    num_iterations = 20      # Maximum number of times E- and M-steps will be iterated
    gmm_mean_list = []       # Stores the list of final means from each restart
    cov_matrix_list =[]      # Stores the list of final covariance matrices from each restart
    log_likelihood_list = [] # Stores the list of final log likelihoods
    prior_list = []          # Stores the list of final priors

    # Import the data 
    data = read_csv("GMM_dataset.csv")

    for _ in range(r):
        # Run k-means in order to obtain initial means and targets
        results = run_k_means_algorithm()
        centroids = results["centroids"]
        targets = results["targets"]
        k = results["k"]

        # Initialize list of covariance matrices, initialize uniform priors, and initialize log likelihood
        covariance_matrices = []
        priors = np.array([1/k for i in range(k)])
        old_log_likelihood = float('-inf')

        for i in range(k):
            data_pts = [data[j] for j in range(data.shape[0]) if targets[j] == i]
            if data_pts != []:
                data_pts = np.array(data_pts)
                cov = np.cov(data_pts.T)
                covariance_matrices.append(cov)

        for _ in range(num_iterations):
            responsibilities = expectation_step(data, centroids, covariance_matrices, priors)

            N_k = np.sum(responsibilities, axis=1)

            centroids, covariance_matrices, priors = maximization_step(data, responsibilities, N_k, k, priors)

            new_log_likelihood = compute_log_likelihood(data, centroids, covariance_matrices, priors)
            #plot_gmm_data(data, centroids, covariance_matrices, k)
            if abs(new_log_likelihood - old_log_likelihood) < 0.0001:
                break
            else:
                old_log_likelihood = new_log_likelihood

        gmm_mean_list.append(centroids)
        cov_matrix_list.append(covariance_matrices)
        log_likelihood_list.append(new_log_likelihood)
        prior_list.append(priors)

    print("GMM mean list: {0}".format(gmm_mean_list))
    print("Length of GMM mean list: {0}\n".format(len(gmm_mean_list)))
    print("Covariance matrix list: {0}".format(cov_matrix_list))
    print("Length of covariance matrix list: {0}\n".format(len(cov_matrix_list)))
    print("Log likelihood list: {0}".format(log_likelihood_list))
    print("Length of log likelihood list: {0}\n".format(len(log_likelihood_list)))
    print("Prior list: {0}".format(prior_list))
    print("Length of prior list: {0}\n".format(len(prior_list)))

if __name__ == '__main__':
    main()