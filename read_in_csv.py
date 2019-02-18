import pandas as pd

# Hannah Galbraith
# CS546
# Program 1
# 2/17/19


def read_csv(filename):
    """:param filename: string
       Reads in a csv file given by filename and returns the data. Assumes there is no header. """
    datareader = pd.read_csv(filename, header=None)
    data = datareader.values

    return data

