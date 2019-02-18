from k_means_clustering import k_means_algorithm
from gmm_clustering import gmm_algorithm

# Hannah Galbraith
# CS546
# Program 1
# 2/17/19

def main():
    """ Runs through both the k-means algorithm and the EM algorithm for GMMs. """
    k_means_algorithm()
    gmm_algorithm()

if __name__ == '__main__':
    main()