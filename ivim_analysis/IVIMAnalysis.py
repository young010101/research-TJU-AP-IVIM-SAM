import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from math import pi
import pickle
import os 

from dipy.reconst.ivim import IvimModel
from dipy.core.gradients import gradient_table

from ivim_analysis.gen_mask import generate_mask, plot_map_roi, plot_map

class IVIMAnalysis:
    """Given a nii file, ROI coordinates and radius, this class fits the IVIM model to the data and plots the maps.
    
    Attributes:
    """
    def __init__(self, nii_path: str, x_roi: int, y_roi: int, rad: int, patient_id: str):
        self.nii_path = nii_path
        self.x_roi = x_roi
        self.y_roi = y_roi
        self.rad = rad
        self.patient_id = patient_id
        self.img = nib.load(self.nii_path)
        self.img_data = self.img.get_fdata()
        self.mask_roi = None
        self.ivimfit = None
        self.bvals =  None
        self.bvecs = None
        self.data_slice = None
        self.estimated_params_roi = None
        self.ivim_params_maps = None
        self.pickle_ivim_path = None

    class param_maps:
        def __init__(self, S0, f, D_star, D):
            self.S0 = S0
            self.f = f
            self.D_star = D_star
            self.D = D

    def preprocess_data(self):
        """!Don't RUN!!! No pass the test
        Normalize the data. Warning: This function modifies the original data."""
        avol_ = self.img_data[:, :, :, -1].copy()
        avol_[avol_ == 0] = 1
        tmp = self.img_data[:, :, :, 1] / avol_
        self.img_data[:, :, :, 1] = tmp

    def generate_roi_mask(self):
        i_h, i_w = self.img_data.shape[0], self.img_data.shape[1]
        self.mask_roi = generate_mask(i_h, i_w, self.rad, self.x_roi, self.y_roi)

    def fit_ivim_model(self, bvals, bvecs, pancreas_slice = 9):
        gtab = gradient_table(bvals, bvecs, b0_threshold=0)
        ivimmodel = IvimModel(gtab, fit_method='trr')
        # !Only one slice is used for now
        self.data_slice = self.img_data[:,:,pancreas_slice,:]
        self.ivimfit = ivimmodel.fit(self.data_slice)
        self.ivim_params_maps = self.param_maps(self.ivimfit.S0_predicted, self.ivimfit.perfusion_fraction, self.ivimfit.D_star, self.ivimfit.D)

    def save_ivim_params(self, output_path: str = "../output/pickles/ivim"):
        """Save the IVIM parameters to a pickle file"""
        assert self.ivim_params_maps is not None, "IVIM parameters not found. Run the IVIM model first."
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        pickle_ivim_path = os.path.join(output_path, 'ivim_maps'+self.patient_id+'.pkl')
        with open(pickle_ivim_path, 'wb') as f:
            pickle.dump(self.ivim_params_maps, f)
        
        self.pickle_ivim_path = pickle_ivim_path

    def plot_map(self, map_data: np.ndarray, title:str ="Predicted S0", lim=(0, 10000), filename: str = "predicted_S0"):
        plot_map(map_data, title, lim, filename)

    def plot_maps(self):
        lim = [(0, 10000), (0, 1), (0, 0.01), (0, 0.001)]
        for key, value, lim in zip(self.ivim_params_maps.__dict__.keys(), self.ivim_params_maps.__dict__.values(), lim):
            plot_map(value, key, lim, key)
        
            # plot_map(self.ivimfit.S0_predicted, "Predicted S0", (0, 10000), "predicted_S0")
            # plot_map(self.data_slice[:, :, 0], "Measured S0", (0, 10000), "measured_S0")
            # plot_map(self.ivimfit.perfusion_fraction, "f", (0, 1), "perfusion_fraction")
            # plot_map(self.ivimfit.D_star, "D*", (0, 0.01), "perfusion_coeff")
            # plot_map(self.ivimfit.D, "D", (0, 0.001), "diffusion_coeff")

    def plot_maps_roi(self):
        plot_map_roi(self.ivimfit.S0_predicted, "Predicted S0", (0, 10000), "predicted_S0", self.x_roi, self.y_roi)
        plot_map_roi(self.data_slice[:, :, 0], "Measured S0", (0, 10000), "measured_S0", self.x_roi, self.y_roi)
        plot_map_roi(self.ivimfit.perfusion_fraction, "f", (0, 1), "perfusion_fraction", self.x_roi, self.y_roi)
        plot_map_roi(self.ivimfit.D_star, "D*", (0, 0.01), "perfusion_coeff", self.x_roi, self.y_roi)
        plot_map_roi(self.ivimfit.D, "D", (0, 0.001), "diffusion_coeff", self.x_roi, self.y_roi)
    
    def check_pickle_and_load(self, pickle_ivim_path: str):
        """If `pickle` file found, load the data from it."""
        if os.path.exists(pickle_ivim_path):
            with open(pickle_ivim_path, 'rb') as file:
                self.ivim_params_maps = pickle.load(file)
            assert self.ivim_params_maps is not None
            self.plot_maps()
            return True
        return False

    def run_analysis(self, bvals, bvecs, load_from_pickle: bool = False, pickle_ivim_path: str = None, save_ivim_params: bool = False): 
        self.generate_roi_mask()
        self.bvals = bvals
        self.bvecs = bvecs
        if load_from_pickle:
            print("Loading data from pickle file")
            if self.check_pickle_and_load(pickle_ivim_path):
                print("Data loaded successfully")
            else:
                print("Pickle file not found. Running the analysis.")
                self.fit_ivim_model(bvals, bvecs)  # Consumes a lot of time
        else:
            self.fit_ivim_model(bvals, bvecs)

        # self.preprocess_data()
        self.plot_maps()
        self.estimated_params_of_roi()
        if save_ivim_params and self.ivim_params_maps is not None and load_from_pickle == False:
            self.save_ivim_params()
        
    def plt_circle_roi(self):
        """plot the ROI circle on the image"""
        x_roi = self.x_roi
        y_roi = self.y_roi
        radius = self.rad
        plt.scatter(x_roi,y_roi,marker='o',s=radius**2*pi,facecolors='None',edgecolors='r')  # circle ROI

    def b0_plot(self, pancreas_slice = 9):
        """plot the b0 image"""
        plt.imshow(self.img_data[:,:,pancreas_slice,1],'gray')

    def b0_plot_roi(self):
        self.b0_plot()
        self.plt_circle_roi(x_roi, y_roi, rad)
        plt.show()

    def plot_b_intensities(self, pancreas_slice = 9):
        """plot the b-value intensities"""
        if self.mask_roi is None:
            self.generate_roi_mask()
            print("!Warning! ROI mask generated")
        assert self.mask_roi is not None, "Generate the ROI mask first"
        mask_roi = self.mask_roi
        num_bval = self.img_data.shape[3]
        print(num_bval)
        intensive_of_10b = np.zeros(num_bval)
        for i_bval in range(num_bval):
            intense_roi = np.sum(self.img_data[:,:,pancreas_slice,i_bval]*mask_roi)/mask_roi.sum()
            # intensive_of_10b[i_bval-9] = intense_roi
            intensive_of_10b[i_bval-num_bval+1] = intense_roi
        
        bvals = self.bvals
        assert bvals is not None, "b-values not found"
        assert len(bvals) == num_bval, "b-values length mismatch"
        assert len(bvals) == len(intensive_of_10b), "b-values length mismatch"
        if len(bvals) == 10:
            plt.scatter([0, 20, 50, 80, 150, 200, 500, 800, 1000, 1500],intensive_of_10b)
        else:
            # !This part is broken
            plt.scatter([0, 10, 20, 50, 80, 150, 200, 500, 800, 1000, 1500],intensive_of_10b)
        # plt.scatter(bvals.sort(),intensive_of_10b)

    def estimated_params_of_roi(self, is_print = False):
        """List all poisition of where value is true"""
        estimated_params_roi = []

        if self.ivimfit is None:
            print("Run the IVIM model first.")
            # estimated_params_roi.append(np.sum(np.nan_to_num(self.ivimfit.model_params[:,:,i])*mask_roi)/mask_roi.sum())
            return None

        shape_ivim_params = self.ivimfit.model_params.shape

        mask_roi = self.mask_roi

        for i in range(shape_ivim_params[2]):
            estimated_params_roi.append(np.sum(np.nan_to_num(self.ivimfit.model_params[:,:,i])*mask_roi)/mask_roi.sum())

        if is_print:
            for param in estimated_params_roi:
                print(f"Estimated parameter: {param}")
            
        self.estimated_params_roi = estimated_params_roi

        return estimated_params_roi


if __name__ == "__main__":
    nii_path = 'ST0_A01-AB+MRCP+CE-Q_20240120092423_11.nii.gz'
    x_roi = 129
    y_roi = 156
    rad = 4
    # bvals, bvecs = read_bvals_bvecs('bvals.txt', 'bvecs.txt')

    analysis = IVIMAnalysis(nii_path, x_roi, y_roi, rad)
    # analysis.run_analysis(bvals, bvecs)
