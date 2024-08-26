import numpy as np

from dipy.io.gradients import read_bvals_bvecs
from dipy.data import get_fnames

_, fbval, fbvec = get_fnames('ivim')

_, bvecs = read_bvals_bvecs(fbval, fbvec)
bvals = np.array([20, 50, 80, 150, 200, 500, 800, 1000, 1500, 0])
bvecs = bvecs[0:10,:]

def load_b():
    return bvals, bvecs