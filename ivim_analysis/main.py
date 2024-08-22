import time
import os
import numpy as np
import matplotlib.pyplot as plt
from data.data_loader import load_nii, get_data_paths, extract_layer
from roi_selector import select_circle_roi, display_roi
from masks.generator import generate_circle_mask
from models.ivim_model import fit_ivim_model
from visualization.plots import plot_slice
from visualization.maps import create_intensity_map
from utils.math_utils import calculate_mean_intensity
from visualization.plots import plot_specific_layer
from utils.other_utils import get_bvals_bvecs
from gen_mask import plot_map_roi, plot_map, save_figure, generate_mask

# 定义装饰器
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to run.")
        return result
    return wrapper

def mask_esti(ivimfit, mask_roi):
# list all poisition of where value is true
    estimated_params_roi = []

    shape_ivim_params = ivimfit.model_params.shape

    for i in range(shape_ivim_params[2]):
        estimated_params_roi.append(np.sum(np.nan_to_num(ivimfit.model_params[:,:,i])*mask_roi)/mask_roi.sum())


    return np.array(estimated_params_roi)

# 使用装饰器
@timing_decorator
def main():
    # Load data paths
    data_folder = '/data/users/cyang/acute_pancreatitis/unprocess/nii/pantient2/'
    data_paths = get_data_paths(data_folder, '.nii.gz')

    # data = load_nii()

    # # Load and process data
    for path in data_paths:
        if "IVIM_Multi_b" in path:
            # Process files that have "IVIM_Multi_b" in their filename
            data = load_nii(path)
            print(path, "hello")
            
        # b0_volume = calculate_b0_volume(data)
        # s0 = calculate_s0(data, b0_volume)
    
    pancreas_slice, b_value_index = 12, 1
    plot_specific_layer(data, pancreas_slice, b_value_index, f"Layer {pancreas_slice} of {os.path.basename(path)}")

    # Select ROI
    x_center, y_center, radius = 125, 180, 4  # cyang: Set by myself temp.
    x_roi, y_roi, radius = select_circle_roi(data, x_center, y_center, radius)
    display_roi(x_roi, y_roi, radius)

    plt.show()

    # Define the output directory
    output_dir = 'output/'

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"Layer {pancreas_slice}.png")
    plt.savefig(output_path)

    # # Generate mask
    # mask = generate_circle_mask(data.shape[0], data.shape[1], 125, 180, 4)

    # 指定JSON文件的路径
    json_file_path = 'json_data/b_values.json'

    # 调用函数并提取b值数组
    bvals, bvecs = get_bvals_bvecs(json_file_path)  # Assume this function is defined elsewhere
    if bvals is not None:
        print("Extracted b-values:", bvals)
        # 这里可以添加更多的处理逻辑
    else:
        print("Failed to extract b-values.")
     
    data_slice_of_10b = data[:,:,pancreas_slice,:]

    # Fit IVIM model  
    ivimfit = fit_ivim_model(data_slice_of_10b, bvals, bvecs)

    plot_map(ivimfit.S0_predicted, "Predicted S0", (0, 10000), "predicted_S0.png")
    # plot_map(data_slice[:, :, 0], "Measured S0", (0, 10000), "measured_S0.png")
    plot_map(ivimfit.perfusion_fraction, "f", (0, 1), "perfusion_fraction.png")
    plot_map(ivimfit.D_star, "D*", (0, 0.01), "perfusion_coeff.png")
    fig_d = plot_map(ivimfit.D, "D", (0, 0.001), "diffusion_coeff.png")
    save_figure(fig_d, "diffusion_coeff.png")
    
    i_h, i_w = data.shape[0], data.shape[1]
    mask_roi = generate_mask(i_h, i_w, radius, x_roi, y_roi)
    tmp = mask_esti(ivimfit, mask_roi)
    print("hello", tmp)
    # save to txt

    # # Visualize results
    # plot_slice(s0, 125, 180, 4)
    # create_intensity_map(ivim_fit.S0_predicted, (0, 10000), 'predicted_S0.png')

if __name__ == '__main__':
    main()
