import pandas as pd

def read_csv(filename):
    datareader = pd.read_csv(filename, header=None)
    data = datareader.values

    return data

