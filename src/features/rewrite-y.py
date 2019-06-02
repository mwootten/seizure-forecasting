import numpy as np
import os

def rewrite(array):
    probabilities = array[np.logical_and(array != 1, array != -1)]
    probabilities = 2 * probabilities - 1
    return array

for file in os.listdir():
    np.load(file).dump(file)
