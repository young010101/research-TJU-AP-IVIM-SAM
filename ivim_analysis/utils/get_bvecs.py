from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs

def get_bvecs():
    _, fbval, fbvec = get_fnames('ivim')
    _, bvecs = read_bvals_bvecs(fbval, fbvec)
    ap_bvecs = bvecs[0:10,:]
    return ap_bvecs