from matplotlib import pyplot as plt
from math import pi

def select_circle_roi(image_data, x_center, y_center, radius):
    # Code to select ROI using a circular mask
    return x_center, y_center, radius

def display_roi(x_roi, y_roi, radius):
    # Code to display the ROI on the image
    plt.scatter(x_roi, y_roi, s=radius**2*pi, edgecolors='r', facecolors='none')
    