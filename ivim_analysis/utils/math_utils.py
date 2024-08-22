import numpy as np

def sum_intensity(data, mask):
    return np.sum(data * mask)

def calculate_mean_intensity(data, mask):
    return sum_intensity(data, mask) / mask.sum()