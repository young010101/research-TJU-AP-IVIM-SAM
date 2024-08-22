import os
from scripts.ivim_analysis.data.data_loader import load_image, get_image_data
from roi_selector import select_circle_roi, display_roi
from scripts.ivim_analysis.models.ivim_model import fit_ivim_model, extract_model_params
from scripts.ivim_analysis.visualization.visualization import visualize_data, save_figure

def main():
    # Define paths and filenames
    image_path = 'path/to/image.nii'
    # ...ROI selection and other parameters...

    # Load and process data
    image = load_image(image_path)
    data = get_image_data(image_path)

    # Select ROI
    x_roi, y_roi, radius = select_circle_roi(data, x_center, y_center)
    display_roi(data, x_roi, y_roi, radius)

    # Fit IVIM model
    gtab = gradient_table(bvals, bvecs)
    ivim_fit = fit_ivim_model(data, gtab)

    # Extract and visualize model parameters
    params = extract_model_params(ivim_fit, x_roi, y_roi)
    visualize_data(params, 'IVIM Model Parameters')
    save_figure(plt, 'ivim_params.png')

if __name__ == '__main__':
    main()