import nibabel as nib
import os
import numpy as np

def load_image(filename):
    return nib.load(filename)

def get_image_data(image_path):
    img = load_image(image_path)
    return img.get_fdata()

def load_nii(filename):
    return nib.load(filename).get_fdata()

def get_data_paths(data_folder, pattern):
    return [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(pattern)]

def extract_layer(image_input, layer_index, b_value_index):
    """
    Extract a specific layer from a 4D NIfTI image or a 4D NumPy array.
    NOTE: 注意性能开销，可以使用装饰器或工厂模式

    Parameters:
    - image_input (str or np.ndarray): The path to the NIfTI image file or a 4D NumPy array.
    - layer_index (int): The index of the layer to extract.

    Returns:
    - A NumPy array representing the extracted layer.
    """
    if isinstance(image_input, str):
        # Load the image from the file path
        image = nib.load(image_input)
        data = image.get_fdata()
    elif isinstance(image_input, np.ndarray):
        # Assume image_input is already a 4D NumPy array
        data = image_input
    else:
        raise ValueError("Unsupported input type. Expected a file path or a 4D NumPy array.")

    return data[:, :, layer_index, b_value_index]