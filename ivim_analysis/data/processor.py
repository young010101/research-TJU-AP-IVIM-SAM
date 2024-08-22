import numpy as np

def calculate_b0_volume(data):
    b0_slice = data[..., -1].copy()
    b0_slice[b0_slice == 0] = 1
    return b0_slice

def calculate_s0(data, b0_volume):
    return data[..., 1] / b0_volume