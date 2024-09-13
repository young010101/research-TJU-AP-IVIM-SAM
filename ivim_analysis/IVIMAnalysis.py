from typing import Any, Dict, List
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from math import pi
import pickle
import os

from dipy.reconst.ivim import IvimModel
from dipy.core.gradients import gradient_table

from ivim_analysis.gen_mask import generate_mask, plot_map_roi, plot_map, plot_ivim


class IVIMAnalysis:
    """Given a nii file, ROI coordinates and radius, this class fits the IVIM model to the data and plots the maps.

    Attributes:
    """

    def __init__(
        self,
        nii_path: str,
        pancreas_slice_idx: int,
        x_roi: int,
        y_roi: int,
        rad: int,
        patient_id: str,
    ):
        self.nii_path: str = nii_path
        self.pancreas_slice_idx: int = pancreas_slice_idx
        self.x_roi: int = x_roi
        self.y_roi: int = y_roi
        self.rad: int = rad
        self.patient_id: str = patient_id
        self.img: nib.Nifti1Image = nib.load(self.nii_path)
        self.img_data: np.ndarray = self.img.get_fdata()
        # self.mask_roi: np.ndarray = None
        self.ivimfit: Any = None
        self.bvals: np.ndarray = None
        self.bvecs: np.ndarray = None
        # self.estimated_params_roi: np.ndarray = None
        self.ivim_params_maps: Any = None
        self.pickle_ivim_path: str = None
        self.dict_ivim_params: Dict[str, np.ndarray] = None
        self.circles: List[tuple] = None

    class param_maps:
        """Also can use dictionary to store the parameters"""

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

    @DeprecationWarning
    def generate_roi_mask(self):
        i_h, i_w = self.img_data.shape[0], self.img_data.shape[1]
        self._mask_roi = generate_mask(i_h, i_w, self.rad, self.x_roi, self.y_roi)

    @property
    def mask_roi(self) -> np.ndarray:
        i_h, i_w = self.img_data.shape[0], self.img_data.shape[1]
        self._mask_roi = generate_mask(i_h, i_w, self.rad, self.x_roi, self.y_roi)
        return self._mask_roi

    # @mask_roi.setter
    @property
    def mask_multi_roi(self) -> List[np.ndarray]:
        """Generate mask for multiple ROIs"""
        circles: list = None
        if circles is None:
            circles = [
                (self.x_roi, self.y_roi, self.rad),
                (100, 100, 5),
                (200, 200, 5),
            ]
        i_h, i_w = self.img_data.shape[0], self.img_data.shape[1]
        mask_roi = []
        for circle in circles:
            x_roi, y_roi, rad = circle
            mask_roi.append(generate_mask(i_h, i_w, rad, x_roi, y_roi))

        self._mask_multi_roi = mask_roi
        return self._mask_multi_roi

    def plot_mask_roi(self):
        fig, ax = plt.subplots(1, len(self.mask_multi_roi), figsize=(15, 4))
        for i, mask in enumerate(self.mask_multi_roi):
            cax = ax[i].imshow(mask, "gray")
            fig.colorbar(cax, ax=ax[i])

    def fit_ivim_model(self, bvals, bvecs):
        gtab = gradient_table(bvals, bvecs, b0_threshold=0)
        ivimmodel = IvimModel(gtab, fit_method="trr")
        # !Only one slice is used for now
        self.pancreas_slice = self.img_data[:, :, self.pancreas_slice_idx, :]
        self.ivimfit = ivimmodel.fit(self.pancreas_slice)
        self.ivim_params_maps = self.param_maps(
            self.ivimfit.S0_predicted,
            self.ivimfit.perfusion_fraction,
            self.ivimfit.D_star,
            self.ivimfit.D,
        )
        self.dict_ivim_params = {
            "S0": self.ivimfit.S0_predicted,
            "f": self.ivimfit.perfusion_fraction,
            "D_star": self.ivimfit.D_star,
            "D": self.ivimfit.D,
        }

    def save_ivim_params(self, output_path: str = "../output/pickles/ivim"):
        """Save the IVIM parameters to a pickle file"""
        assert (
            self.ivim_params_maps is not None
        ), "IVIM parameters not found. Run the IVIM model first."
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        pickle_ivim_path = os.path.join(
            output_path, "ivim_maps" + self.patient_id + ".pkl"
        )
        with open(pickle_ivim_path, "wb") as f:
            pickle.dump(self.ivim_params_maps, f)

        self.pickle_ivim_path = pickle_ivim_path

    # getter data_slice by property
    @property
    def pancreas_slice(self) -> np.ndarray:
        self._pancreas_slice = self.img_data[:, :, self.pancreas_slice_idx, :]
        return self._pancreas_slice

    def plot_ivim(self) -> plt.Figure:
        return plot_ivim(self.dict_ivim_params)

    def plot_map(
        self,
        map_data: np.ndarray,
        title: str = "Predicted S0",
        lim=(0, 10000),
        filename: str = "predicted_S0",
    ):
        plot_map(map_data, title, lim, filename)

    @DeprecationWarning
    def plot_maps(self):
        lim = [(0, 10000), (0, 1), (0, 0.01), (0, 0.001)]
        for key, value, lim in zip(
            self.ivim_params_maps.__dict__.keys(),
            self.ivim_params_maps.__dict__.values(),
            lim,
        ):
            plot_map(value, key, lim, key)

            # plot_map(self.ivimfit.S0_predicted, "Predicted S0", (0, 10000), "predicted_S0")
            # plot_map(self.data_slice[:, :, 0], "Measured S0", (0, 10000), "measured_S0")
            # plot_map(self.ivimfit.perfusion_fraction, "f", (0, 1), "perfusion_fraction")
            # plot_map(self.ivimfit.D_star, "D*", (0, 0.01), "perfusion_coeff")
            # plot_map(self.ivimfit.D, "D", (0, 0.001), "diffusion_coeff")

    @DeprecationWarning
    def plot_maps_roi(self):
        plot_map_roi(
            self.ivimfit.S0_predicted,
            "Predicted S0",
            (0, 10000),
            "predicted_S0",
            self.x_roi,
            self.y_roi,
        )
        plot_map_roi(
            self.pancreas_slice[:, :, 0],
            "Measured S0",
            (0, 10000),
            "measured_S0",
            self.x_roi,
            self.y_roi,
        )
        plot_map_roi(
            self.ivimfit.perfusion_fraction,
            "f",
            (0, 1),
            "perfusion_fraction",
            self.x_roi,
            self.y_roi,
        )
        plot_map_roi(
            self.ivimfit.D_star,
            "D*",
            (0, 0.01),
            "perfusion_coeff",
            self.x_roi,
            self.y_roi,
        )
        plot_map_roi(
            self.ivimfit.D, "D", (0, 0.001), "diffusion_coeff", self.x_roi, self.y_roi
        )

    def check_pickle_and_load(self, pickle_ivim_path: str):
        """If `pickle` file found, load the data from it.
        a cache idea to save time"""
        if os.path.exists(pickle_ivim_path):
            with open(pickle_ivim_path, "rb") as file:
                self.ivim_params_maps = pickle.load(file)
            assert self.ivim_params_maps is not None
            self.dict_ivim_params = {
                "S0": self.ivim_params_maps.S0,
                "f": self.ivim_params_maps.f,
                "D_star": self.ivim_params_maps.D_star,
                "D": self.ivim_params_maps.D,
            }
            # self.plot_maps()
            return True
        return False

    def run_analysis(
        self,
        bvals,
        bvecs,
        load_from_pickle: bool = False,
        pickle_ivim_path: str = None,
        save_ivim_params: bool = False,
        is_plot: bool = True,
    ):
        # self.generate_roi_mask()
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
        if is_plot:
            # self.plot_maps()
            self.plot_ivim()
        # self.estimated_params_roi()
        if (
            save_ivim_params
            and self.ivim_params_maps is not None
            and load_from_pickle == False
        ):
            self.save_ivim_params()

    def plot_circle_roi(self):
        """plot the ROI circle on the image"""
        x_roi = self.x_roi
        y_roi = self.y_roi
        radius = self.rad
        plt.scatter(
            x_roi,
            y_roi,
            marker="o",
            s=radius**2 * pi,
            facecolors="None",
            edgecolors="r",
        )  # circle ROI

    def plot_multi_roi(self, ax, circles: list):
        """plot multiple ROIs on the image

        for simple, circles is a list of tuples (x_roi, y_roi, rad),
        where x_roi, y_roi are the center of the circle and rad is the radius
        Note: size of list is the number of ROIs, which is 3
        """
        for circle in circles:
            x_roi, y_roi, rad = circle
            ax.scatter(
                x_roi,
                y_roi,
                marker="o",
                s=rad**2 * pi,
                facecolors="None",
                edgecolors="r",
            )

    def plot_pancreas_slice(self, ax, plot_roi: bool = False, circles: list = None):
        ax.imshow(self.pancreas_slice[:, :, 0], "gray")

        if plot_roi:
            # self.plot_circle_roi()
            # TODO: circle should be a parameter while class initialization
            if circles is None:
                circles = [
                    (self.x_roi, self.y_roi, self.rad),
                    (130, 130, 5),
                    (130, 110, 5),
                ]
            self.plot_multi_roi(ax, circles)

    @DeprecationWarning
    def b0_plot(self):
        """plot the b0 image"""
        plt.imshow(self.img_data[:, :, self.pancreas_slice_idx, 1], "gray")

    @DeprecationWarning
    def b0_plot_roi(self):
        self.b0_plot()
        self.plot_circle_roi(self.x_roi, self.y_roi, self.rad)
        plt.show()

    def plot_b_intensities(self):
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
            intense_roi = (
                np.sum(self.img_data[:, :, self.pancreas_slice_idx, i_bval] * mask_roi)
                / mask_roi.sum()
            )
            # intensive_of_10b[i_bval-9] = intense_roi
            intensive_of_10b[i_bval - num_bval + 1] = intense_roi

        bvals = self.bvals
        assert bvals is not None, "b-values not found"
        assert len(bvals) == num_bval, "b-values length mismatch"
        assert len(bvals) == len(intensive_of_10b), "b-values length mismatch"
        if len(bvals) == 10:
            plt.scatter(
                [0, 20, 50, 80, 150, 200, 500, 800, 1000, 1500], intensive_of_10b
            )
        else:
            # !This part is broken
            plt.scatter(
                [0, 10, 20, 50, 80, 150, 200, 500, 800, 1000, 1500], intensive_of_10b
            )
        # plt.scatter(bvals.sort(),intensive_of_10b)

    def plot_log_b(self):
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
            intense_roi = (
                np.sum(self.img_data[:, :, self.pancreas_slice_idx, i_bval] * mask_roi)
                / mask_roi.sum()
            )
            # intensive_of_10b[i_bval-9] = intense_roi
            intensive_of_10b[i_bval - num_bval + 1] = np.log(intense_roi)

        bvals = self.bvals
        assert bvals is not None, "b-values not found"
        assert len(bvals) == num_bval, "b-values length mismatch"
        assert len(bvals) == len(intensive_of_10b), "b-values length mismatch"

        # TODO: Fix the x-axis
        if len(bvals) == 10:
            tmp_bvals = [0, 20, 50, 80, 150, 200, 500, 800, 1000, 1500]
        else:
            # !This part is broken
            tmp_bvals = [0, 10, 20, 50, 80, 150, 200, 500, 800, 1000, 1500]
        plt.scatter(tmp_bvals, intensive_of_10b)

        from sklearn.linear_model import LinearRegression

        reg = LinearRegression().fit(
            np.array(tmp_bvals).reshape(-1, 1), intensive_of_10b
        )
        print(reg.coef_, reg.intercept_)
        x = np.array(tmp_bvals)
        y = reg.coef_ * x + reg.intercept_
        plt.plot(x, y, "-r")

    @property
    def estimated_params_roi(self):
        """List all poisition of where value is true"""
        estimated_params_roi_all = {}
        list_roi_name = ["head", "body", "tail"]

        multi_roi = self.mask_multi_roi

        for idx, j in enumerate(list_roi_name):
            estimated_params_roi_all[j] = {}
            roi = multi_roi[idx]
            for i in self.dict_ivim_params:
                estimated_params_roi_all[j][i] = (
                    np.sum(np.nan_to_num(self.dict_ivim_params[i][:, :]) * roi)
                    / roi.sum()
                )

        # for i in self.dict_ivim_params:
        #     estimated_params_roi[i] = (
        #         np.sum(np.nan_to_num(self.dict_ivim_params[i][:, :]) * mask_roi)
        #         / mask_roi.sum()
        #     )

        self._estimated_params_roi = estimated_params_roi_all

        return self._estimated_params_roi

    def print_estimated_params_roi(
        self, save_ivim_params: bool = False, output_path: str = None
    ):
        e_p = self.estimated_params_roi
        print(f"{self.patient_id} estimated parameter:")
        # for param in e_p:
        #     print(f"{param}: {e_p[param]}")
        for roi in e_p:
            print(f"{roi}:")
            for param in e_p[roi]:
                print(f"{param}: {e_p[roi][param]}")

        if output_path is None:
            from datetime import date

            output_path = os.path.join(
                "..", "output", str(date.today()), self.patient_id
            )
            txt_path = os.path.join(output_path, "ivim_params.txt")
            json_path = os.path.join(output_path, "ivim_params.json")
        if save_ivim_params:
            with open(txt_path, "w") as f:
                for roi in e_p:
                    f.write(f"{roi}:\n")
                    for param in e_p[roi]:
                        f.write(f"{param}: {e_p[roi][param]}\n")

        # save the estimated parameters to json file
        if save_ivim_params:
            with open(json_path, "w") as f:
                import json

                json.dump(e_p, f)


if __name__ == "__main__":
    print("Hello")
    # nii_path = "ST0_A01-AB+MRCP+CE-Q_20240120092423_11.nii.gz"
    # x_roi = 129
    # y_roi = 156
    # rad = 4
    # # bvals, bvecs = read_bvals_bvecs('bvals.txt', 'bvecs.txt')

    # analysis = IVIMAnalysis(nii_path, x_roi, y_roi, rad)
    # # analysis.run_analysis(bvals, bvecs)
