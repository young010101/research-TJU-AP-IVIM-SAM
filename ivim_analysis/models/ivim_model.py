from dipy.reconst.ivim import IvimModel
from dipy.core.gradients import gradient_table

def fit_ivim_model(data_slice, bvals, bvecs):
    gtab = gradient_table(bvals, bvecs, b0_threshold=0)
    ivim_model = IvimModel(gtab, fit_method='trr')
    return ivim_model.fit(data_slice)

def extract_model_params(ivim_fit, x_roi, y_roi):
    # Code to extract model parameters for a specific ROI
    pass