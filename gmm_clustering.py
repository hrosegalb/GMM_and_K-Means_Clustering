from read_in_csv import read_csv
from k_means_clustering import run_k_means_for_gmm

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import random

# Hannah Galbraith
# CS546
# Program 1
# 2/17/19

def expectation_step(data, means, covariance_matrices, priors):
    """ :param data: (1500, 2) matrix of floats
        :param means: (k, 2) matrix of floats
        :param covariance_matrices: list of k (2,2) matrices of floats
        :param priors: (k,) vector of floats
        
        Executes the expectation step of the EM algorithm. Calculates the responsibilities
        of each data point for each cluster, normalizes the values, and returns a (k, 1500)
        matrix of responsibilities. """

    responsibilities = np.zeros((means.shape[0], data.shape[0]))

    # Calculate the pdf for each data point for each Gaussian and multiply it by the prior for that
    # Gaussian
    for j in range(data.shape[0]):
        for i in range(means.shape[0]):
            dist = multivariate_normal(mean=means[i], cov=covariance_matrices[i])
            responsibilities[i,j] = priors[i] * dist.pdf(data[j])

    # Normalize the responsibilities by the sum of the responsibilities of a given data point x_i for each cluster
    # by summing all of the columns of the matrix
    responsibilities = responsibilities / np.sum(responsibilities, axis=0)
    return responsibilities


def maximization_step(data, responsibilities, N_k, k, priors):
    """ :param data: (1500, 2) matrix of floats
        :param responsibilities: (k, 1500) matrix of floats
        :param N_k: (3,) vector of floats
        :param k: integer
        :param priors: (k,) vector of floats
        
        Executes the maximization step of the EM algorithm. Updates the means, 
        the sigmas, and the priors for each Gaussian and returns them. """

    covariance_matrices = []
    means = np.zeros((k, 2))

    for i in range(k):
        new_mean = np.dot(responsibilities[i,:], data) / N_k[i] # Gets the sum of the responsibilities * data points, normalized by N_k
        means[i] = new_mean
        sigma = np.zeros((2,2))

        # Gets the sum of the responsibility for each data point multiplied by the transpose of (x_n - mu) * (x_n - mu)
        for j in range(data.shape[0]):
            sigma += responsibilities[i][j] * np.outer((data[j,:] - new_mean), (data[j,:] - new_mean))

        # Normalize sigma_k by N_k and append the matrix to the list of covariance matrices
        sigma /= N_k[i]
        covariance_matrices.append(sigma)
        priors[i] = N_k[i] / np.sum(N_k) # Update prior

    return means, covariance_matrices, priors


def compute_log_likelihood(data, means, covariance_matrices, priors):
    """ :param data: (1500, 2) matrix of floats
        :param means: (k, 2) matrix of floats
        :param covariance_matrices: list of k (2, 2) matrices of floats
        :param priors: (k,) matrix of floats
        
        Computes the log likelihood by summing the responsibilities for each Gaussian, taking the
        log of them, and then summming all the logs for each data point. Returns the log likelihood
        (a float). """

    responsibilities = np.zeros((means.shape[0], data.shape[0]))

    for j in range(data.shape[0]):
        for i in range(means.shape[0]):
            dist = multivariate_normal(mean=means[i], cov=covariance_matrices[i])
            responsibilities[i,j] = priors[i] * dist.pdf(data[j])

    summed_responsibilities = np.sum(responsibilities, axis=0)
    log_of_responsibilities = np.log(summed_responsibilities)
    log_likelihood = np.sum(log_of_responsibilities)
    return log_likelihood


def plot_gmm_data(data, means, covariance_matrices, k):
    """ :param data: (1500, 2) matrix of floats
        :param means: (k, 2) matrix of floats
        :param covariance_matrices: list of k (2, 2) matrices of floats
        :param k: integer
        
        Plots the data points and layers the contours of the Gaussians over them.
        Saves the graph as a png. """

    if k > 3:
        plt.clf()   # Clear figure so graphs don't overlap

    xlist = np.linspace(-4.0, 4.0)
    ylist = np.linspace(-4.0, 4.0)
    X, Y = np.meshgrid(xlist, ylist)

    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], s=7)
    for i in range(k):
        mu = means[i]
        sigma = covariance_matrices[i]

        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        Z = multivariate_normal.pdf(pos, mu, sigma)
        CS = ax.contour(X, Y, Z)

    plt.title("EM Algorithm for GMM Clustering with {0} Clusters".format(k))
    plt.savefig("best-gmm_{0}-clusters.png".format(k))



def gmm_algorithm():
    """ Performs the EM algorithm for GMMs. For each number of clusters, performs r random restarts and
        iterates through the E- and M-steps until there is either convergence or the steps have been iterated
        100 times. Prints out the best final log likelihood from the attempts and the associated parameters.
        Plots the data points and the Gaussians. """

    r = 10                   # Number of random restarts to be done
    num_iterations = 100     # Maximum number of times E- and M-steps will be iterated
    gmm_mean_list = []       # Stores the list of final means from each restart
    cov_matrix_list =[]      # Stores the list of final covariance matrices from each restart
    log_likelihood_list = [] # Stores the list of final log likelihoods
    prior_list = []          # Stores the list of final priors

    # Import the data 
    data = read_csv("GMM_dataset.csv")

    for k in [3, 4]: 
        for _ in range(r):
            # Run k-means in order to obtain initial means and targets
            results = run_k_means_for_gmm(data, k)
            means = results["centroids"]
            targets = results["targets"]

            # Initialize list of covariance matrices, initialize uniform priors, and initialize log likelihood
            covariance_matrices = []
            priors = np.array([1/k for i in range(k)])
            old_log_likelihood = float('-inf')

            # Calculate the initial covariance matrices based on the targets for each data point given by
            # run_k_means_for_gmm
            for i in range(k):
                data_pts = [data[j] for j in range(data.shape[0]) if targets[j] == i]
                if data_pts != []:
                    data_pts = np.array(data_pts)
                    cov = np.cov(data_pts.T)
                    covariance_matrices.append(cov)

            # Iterate through E- and M-steps
            for _ in range(num_iterations):
                responsibilities = expectation_step(data, means, covariance_matrices, priors)

                N_k = np.sum(responsibilities, axis=1)

                means, covariance_matrices, priors = maximization_step(data, responsibilities, N_k, k, priors)

                # Check if the log likelihood has converged. If not, replace the old log likelihood
                # with the new log likelihood and iterate through the E- and M-steps again
                new_log_likelihood = compute_log_likelihood(data, means, covariance_matrices, priors)
                if abs(new_log_likelihood - old_log_likelihood) < 0.00001:
                    break
                else:
                    old_log_likelihood = new_log_likelihood

            # Append final means, covariance matrices, log likelihoods, and priors to their respective lists
            gmm_mean_list.append(means)
            cov_matrix_list.append(covariance_matrices)
            log_likelihood_list.append(new_log_likelihood)
            prior_list.append(priors)

        # Find the largest log likelihood and get the means, covariance matrices, and priors associated 
        # with that iteration of the algorithm
        max_idx = log_likelihood_list.index(max(log_likelihood_list))
        best_means = gmm_mean_list[max_idx]
        best_cov_matrices = cov_matrix_list[max_idx]
        best_priors = prior_list[max_idx]

        # Print out parameters and plot the data
        print("[GMM] Best log likelihood out of {0} clusters and {1} restarts: {2}\n".format(k, r, log_likelihood_list[max_idx]))
        print("[GMM] Best means out of {0} clusters and {1} restarts: {2}\n".format(k, r, best_means))
        print("[GMM] Best covariance matrices out of {0} clusters and {1} restarts: {2}\n".format(k, r, best_cov_matrices))
        print("[GMM] Best priors out of {0} clusters and {1} restarts: {2}\n".format(k, r, best_priors))
        plot_gmm_data(data, best_means, best_cov_matrices, k)